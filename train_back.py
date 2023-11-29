import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from util_funcs.losses import get_loss
from backbones import get_model
from util_funcs.dataset import MXFaceDataset, DataLoaderX
from util_funcs.partial_fc import PartialFC  # 这可是个神奇的模块
from util_funcs.util_config import get_config, init_logging, AverageMeter
from util_funcs.callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from util_funcs.utils_amp import MaxClipGradScaler


def init_rank(cfg):
    try:
        world_size = int(os.environ['WORLD_SIZE'])  # 分布式系统上所有节点上所有进程数总和, 一般有多少gpu就有多少进程数
        rank = int(os.environ['RANK'])  # 分布式系统上当前进程号,[0,word_size)
        dist_url = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
        # dist.init_process_group('nccl', init_method=dist_url, rank=rank, world_size=world_size)
    except KeyError:
        world_size = 1
        rank = 0
        # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12584', rank=rank, world_size=world_size)

    # args.local_rank接受的是分布式launch自动传入的参数local_rank, 针对当前节点来说, 指每个这个节点上gpu编号
    # 不同节点上gpu编号相同
    # local_rank=args.local_rank
    local_rank = 0
    torch.cuda.set_device(local_rank)  # 设置当前使用的gpu编号
    # 递归创建多级目录, 若目录已经存在,则不再重复创建, 也不抛异常
    if rank == 0:  # 只在分布式系统第0个进程上创建记录日志文件
        os.makedirs(cfg.output, exist_ok=True)
        init_logging(cfg.output)

    return world_size, rank, local_rank


def get_dataset(cfg, world_size, local_rank):
    train_set = MXFaceDataset(cfg.dataset_path)
    # 数据并行, 将数据集加载到分布式系统上
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set, shuffle=True)
    # todo 制作数据集 ?☹☀ \(^o^)/~
    # pin_memory, 锁页内存,将内存的Tensor转义到GPU的显存会更快一些
    train_loader = DataLoaderX(local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
                               sampler=None, num_workers=2, pin_memory=True, drop_last=True)  # todo 分布式要改为train_sampler
    train_nums = len(train_set)
    total_batch_size = cfg.batch_size * world_size  # 配置文件中的batch_size,是分布式系统上每个gpu的batch_size.
    # todo 这是什么？
    cfg.warmup_step = train_nums // total_batch_size * cfg.warmup_epoch
    cfg.total_step = train_nums // total_batch_size * cfg.num_epoch
    train_sampler = None
    return train_sampler, train_loader, train_nums, total_batch_size


def get_backbone_fc(cfg, world_size, rank, local_rank):
    '''
    获得特征提取网络,并处理是否要载入预训练模型
    '''
    backbone = get_model(cfg.network, dropout=0.0,
                         fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
    if cfg.resume:  # 导入预训练好的模型, 或从断点处导入模型,继续训练.
        try:
            backbone_path = os.path.join(cfg.output, 'backbone.pth')
            backbone.load_state_dict(torch.load(backbone_path, map_location=torch.device(local_rank)))
            if rank == 0:  # 只打印一次
                logging.info('backbone resume successfully!')
        except Exception as e:
            if rank == 0:
                logging.info(f'resume fail,backbone init successfully')
    # 将模型加载到分布式系统当前节点当前gpu上
    # backbone = torch.nn.parallel.DistributedDataParallel(module=backbone,
    #                                                      broadcast_buffers=False, device_ids=[local_rank])
    # 获得损失函数
    margin_softmax = get_loss(cfg.loss)
    # 获得分布式系统的
    fc_model = PartialFC(rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
                         batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
                         sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    return backbone, fc_model


def main(args):
    ''''''
    '''
    启用分布式进程后,会在每一个GPU上都启动一个main函数
    '''
    cfg = get_config(args.config)
    # 初始化分布式系统
    world_size, rank, local_rank = init_rank(cfg)
    # 预处理数据集,并载入分布式系统
    train_sampler, train_loader, train_nums, total_batch = get_dataset(cfg, world_size, local_rank)
    # 获得主干提取网络,并载入分布式系统
    backbone, fc_model = get_backbone_fc(cfg, world_size, rank, local_rank)
    backbone.train()

    def lr_step_func(cur_step):
        cfg.decay_step = [x * train_nums // total_batch for x in cfg.decay_epoch]

        if cur_step < cfg.warmup_step:
            return cur_step / cfg.warmup_step
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= cur_step])

    # 主干网络和分布式模型部分优化器参数一样
    params_dict = {"lr": cfg.lr / 512 * cfg.batch_size * world_size,
                   "momentum": 0.9, "weight_decay": cfg.weight_decay}
    backbone_optimizer = torch.optim.SGD(params=[{'params': backbone.parameters()}], **params_dict)
    fc_optimizer = torch.optim.SGD(params=[{'params': fc_model.parameters()}], **params_dict)

    # 自定义调整,优化器中学习率的大小,学习率=初始学习率*lr_lambda函数的值
    backbone_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=backbone_optimizer, lr_lambda=lr_step_func)
    fc_sheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=fc_optimizer, lr_lambda=lr_step_func)

    # 打印配置文件基本信息
    for key, value in cfg.items():
        space_num = 25 - len(key)
        # 必须用加号,不能用逗号分割
        logging.info(': ' + key + ' ' * space_num + str(value))

    val_target = cfg.val_targets  # 这是测试数据
    # 各种回调函数
    callback_verify = CallBackVerification(2000, rank, val_target, cfg.dataset_path)
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()  # todo 这个损失函数是什么
    start_epoch = 0
    global_step = 0
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None

    for epoch in range(start_epoch, cfg.num_epoch):
        # 使DistributedSampler中的shuffle=True生效,使每个epoch数据都不同.不然每个epoch数据都一样了
        # train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            global_step += 1
            backbone_optimizer.zero_grad()
            fc_optimizer.zero_grad()

            # 对提取的特征进行L2正则化,限制在0~之间,注意和不为1
            features = F.normalize(backbone(img))
            x_grad, loss_v = fc_model.forward_backward(label, features, fc_optimizer)
            fc_model.update()  # 分布式模型中,自己实现的函数
            # 下述判断是对主干网络的权重更新
            if cfg.fp16:  # 16位于32位混合训练   todo 一般会这样玩吗?
                features.backward(grad_amp.scale(x_grad))  # 分布式模型梯度不自动更新,手动放大到32再更新
                grad_amp.unscale_(backbone_optimizer)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)  # 限制梯度大小
                grad_amp.step(backbone_optimizer)  # 更新梯度
                grad_amp.update()  # 对自己状态清0
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)  # 限制梯度大小
                backbone_optimizer.step()

            # 更新分布式模型权重
            fc_optimizer.step()
            backbone_scheduler.step()
            fc_sheduler.step()

            loss.update(loss_v, 1)
            # 验证部分
            callback_verify(global_step, backbone)  # todo 还是不懂干嘛的!
            callback_logging(global_step, loss, epoch, cfg.fp16, backbone_scheduler.get_last_lr()[0], grad_amp)
        callback_checkpoint(global_step, backbone, fc_model)
    # dist.destroy_process_group()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
