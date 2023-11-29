import logging
import os

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter


class PartialFC(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./"):
        """
        rank: int
            Unique process(GPU) ID from 0 to world_size - 1.
        local_rank: int
            Unique process(GPU) ID within the server from 0 to 7.
        world_size: int
            Number of GPU.
        batch_size: int
            Batch size on current rank(GPU).
        resume: bool
            Select whether to restore the weight of softmax.
        margin_softmax: callable
            A function of margin softmax, eg: cosface, arcface.
        num_classes: int
            The number of class center storage in current rank(CPU/GPU), usually is total_classes // world_size,
            required.
        sample_rate: float
            The partial fc sampling rate, when the number of classes increases to more than 2 millions, Sampling
            can greatly speed up training, and reduce a lot of GPU memory, default is 1.0.
        embedding_size: int
            The feature dimension, default is 512.
        prefix: str
            Path for save checkpoint, default is './'.
        """
        super(PartialFC, self).__init__()
        #
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device: torch.device = torch.device(f"cuda:{self.local_rank}")
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate   # todo 这是什么
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix   # todo 这是什么
        # 每个gpu上的负责的类别数. 均分到每个gpu上
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        # 分布式系统上每个gpu上负责的类别开始编号, 如104类, start=78,指这个gpu负责的类别从78开始, 后延13个
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        # 当前GPU负责的总类别个数
        self.num_sample: int = int(self.sample_rate * self.num_local)
        # 这两个是干嘛的
        self.weight_name = os.path.join(self.prefix, f"rank_{self.rank}_softmax_weight.pt")
        self.weight_mom_name = os.path.join(self.prefix, "rank_{self.rank}_softmax_weight_mom.pt")

        if resume:  # 是否加载预训练权重
            try:
                self.weight: torch.Tensor = torch.load(self.weight_name)
                self.weight_mom: torch.Tensor = torch.load(self.weight_mom_name)
                if self.weight.shape[0] != self.num_local or self.weight_mom.shape[0] != self.num_local:
                    raise IndexError
                logging.info("softmax weight resume successfully!")
                logging.info("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
                self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
                logging.info("softmax weight init!")
                logging.info("softmax weight mom init!")
        else:
            self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
            logging.info("softmax weight init successfully!")
            logging.info("softmax weight mom init successfully!")
        # todo 配置cuda流, 将数据载入流文件?
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)

        self.index = None   # todo 这是什么
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))

    def save_params(self):
        """ Save softmax weight for each rank on prefix
        """
        torch.save(self.weight.data, self.weight_name)
        torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        """
        Sample all positive class centers in each rank, and random select neg class centers to filling a fixed
        `num_sample`.

        total_label: tensor
            Label after all gather, which cross all GPUs.
        """
        # 当前gpu负责的正类,[self.class_start,self.class_start+self.num_local)
        # 如果total_label中某个label在这个区间中,设为True, 否则置位false. index_positive shape与total_label的shape一样
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        # ~ 取反后, 不属于这个区间内的label对应的位置都为True, 是不属于当前GPU负责的正类, 置位-1, 不做处理
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start     # 当前GPU的类标号从0开始
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_label[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_label[index_positive] = torch.searchsorted(index, total_label[index_positive])
            self.sub_weight = Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        """ Partial fc forward, `logits = X * sample(W)`
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        """ Set updated weight and weight_mom to memory bank.
        """
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        """
        get sampled class centers for cal softmax.

        label: tensor
            Label tensor on each rank.
        optimizer: opt
            Optimizer for partial fc, which need to get weight mom.
        """
        with torch.cuda.stream(self.stream):
            # 创建聚合所有rank上label 之前的全0占位空壳
            total_label = torch.zeros(size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            # 将不同rank上的值合并在一起, 最后每个rank使用的值都是合并的值. label的值整合到total_label中
            # dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            # total_label, tensor, shape=(bath_size*self.world_size,)
            self.sample(total_label)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight

    def forward_backward(self, label, features, optimizer):
        """
        Partial fc forward and backward with model parallel

        label: tensor
            Label tensor on each rank(GPU)
        features: tensor
            Features tensor on each rank(GPU)
        optimizer: optimizer
            Optimizer for partial fc

        Returns:
        --------
        x_grad: tensor
            The gradient of features.
        loss_v: tensor
            Loss value for cross entropy.
        """
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        # dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True

        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)

        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            # dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            # dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            # dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        # feature gradient all-reduce
        # dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        x_grad = x_grad * self.world_size
        # backward backbone
        return x_grad, loss_v
