import argparse
from loguru import logger
# import logging as logger
import os

import numpy as np
import torch
from typing import List
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from util_funcs.lr_func import PolyScheduler
from pathlib2 import Path

from util_funcs.losses import get_loss, CombinedMarginLoss
from backbones import get_model
from util_funcs.dataset import MXFaceDataset, DataLoaderX, get_dataloader
from util_funcs.partial_fc import PartialFC, PartialFCAdamW  # 这可是个神奇的模块
from util_funcs.util_config import get_config, init_logging, AverageMeter
from util_funcs.callbacks import CallBackVerification, CallBackLogging
from util_funcs.utils_amp import MaxClipGradScaler

# 初始化分布式系统
try:
    world_size = int(os.environ['WORLD_SIZE'])  # 分布式系统上所有节点上所有进程数总和, 一般有多少gpu就有多少进程数
    rank = int(os.environ['RANK'])  # 分布式系统上当前进程号,[0,word_size)
    dist_url = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    # dist.init_process_group(backend='nccl',
    #                         init_method=dist_url,
    #                         rank=rank,
    #                         world_size=world_size)
    # 使用环境变量法不会报错
    dist.init_process_group('nccl')
except KeyError as k:
    world_size = 1
    rank = 0
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:12584',
                            rank=rank,
                            world_size=world_size)


def get_backbone_fc_optimizer(cfg, rank, local_rank):
    '''
    获得特征提取网络,并处理是否要载入预训练模型
    '''
    backbone = get_model(cfg.network, dropout=0.0,
                         fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    # 导入预训练好的模型, 或从断点处导入模型,继续训练.
    if cfg.resume:
        try:
            backbone_path = os.path.join(cfg.output, 'backbone.pth')
            backbone.load_state_dict(torch.load(backbone_path, map_location=torch.device(local_rank)))
            if rank == 0:  # 只打印一次
                logger.info('backbone resume successfully!')
        except Exception as e:
            if rank == 0:
                logger.info(f'resume fail,backbone init successfully,error:{e}')
    # 将模型加载到分布式系统当前节点当前gpu上
    backbone = torch.nn.parallel.DistributedDataParallel(module=backbone,
                                                         broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    # 获得损失函数
    margin_loss = CombinedMarginLoss(64, cfg.margin_list[0], cfg.margin_list[1],
                                     cfg.margin_list[2], cfg.interclass_filtering_threshold)
    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(margin_loss=margin_loss,
                                      embedding_size=cfg.embedding_size,
                                      num_classes=cfg.num_classes,
                                      sample_rate=cfg.sample_rate,
                                      fp16=cfg.fp16)
        module_partial_fc.train().cuda()
        optimizer = torch.optim.SGD(params=[{'params': backbone.parameters()},
                                            {'params': module_partial_fc.parameters()}],
                                    lr=cfg.lr,
                                    momentum=0.9,
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        module_partial_fc = PartialFCAdamW(margin_loss=margin_loss,
                                           embedding_size=cfg.embedding_size,
                                           num_classes=cfg.num_classes,
                                           sample_rate=cfg.sample_rate,
                                           fp16=cfg.fp16)
        module_partial_fc.train().cuda()
        optimizer = torch.optim.AdamW(params=[{'params': backbone.parameters()},
                                              {'params': module_partial_fc.parameters()}],
                                      lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
    else:
        raise ValueError('optimizer error')

    # 返回时转为train
    return backbone, module_partial_fc, optimizer


def main(args):
    '''
    启用分布式进程后,会在每一个GPU上都启动一个main函数
    '''
    # 不同节点上gpu编号相同,设置当前使用的gpu编号
    # args.local_rank接受的是分布式launch自动传入的参数local_rank, 针对当前节点来说, 指每个这个节点上gpu编号
    # todo  新版本中,使用os.environ['LOCAL_RANK']替代分布式系统自动传入的--local_rank
    # local_rank = int(os.environ['LOCAL_RANK'])
    seed = 2333
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    # 配置文件中的batch_size,是分布式系统上每个gpu的batch_size
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // total_batch_size * cfg.num_epoch

    # 递归创建多级目录, 若目录已经存在,则不再重复创建, 也不抛异常
    if rank == 0:  # 只在分布式系统第0个进程上创建记录日志文件
        os.makedirs(cfg.output, exist_ok=True)
        # init_logging(cfg.output)
        # 打印配置文件基本信息,只在0号gpu打印一次. 使用loguru时必须指定0号gpu
        for key, value in cfg.items():
            space_num = 20 - len(key)
            # 必须用加号,不能用逗号分割
            # logger.info(': ' + key + ' ' * space_num + str(value))
            logger.info(f"': {key}{' ' * space_num}{str(value)}")

    summary_writer = SummaryWriter(log_dir=str(Path(cfg.output) / 'tensorboard')) if rank == 0 else None
    train_loader = get_dataloader(root_dir=cfg.dataset_path,
                                  local_rank=args.local_rank,
                                  batch_size=cfg.batch_size,
                                  dali=cfg.dali)
    # 获得主干提取网络,并载入分布式系统
    backbone, module_partial_fc, optimizer = get_backbone_fc_optimizer(cfg, rank, args.local_rank)

    lr_scheduler = PolyScheduler(optimizer=optimizer,
                                 base_lr=cfg.lr,
                                 max_steps=cfg.total_step,
                                 warmup_steps=cfg.warmup_step)
    # 各种回调函数
    callback_verify = CallBackVerification(val_targets=cfg.val_targets,
                                           rec_prefix=cfg.dataset_path, summary_writer=summary_writer)
    callback_logging = CallBackLogging(frequent=cfg.frequent, total_step=cfg.total_step,
                                       batch_size=cfg.batch_size, writer=summary_writer)

    loss_am = AverageMeter()  # todo 这个损失函数是什么
    start_epoch = 0
    global_step = 0
    # todo 这是干嘛的?
    grad_amp = torch.cuda.amp.grad_scaler.GradScaler(growth_factor=100)
    for epoch in range(start_epoch, cfg.num_epoch):
        # 使DistributedSampler中的shuffle=True生效,使每个epoch数据都不同.不然每个epoch数据都一样了
        if isinstance(train_loader, torch.utils.data.DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()

            features = backbone(img)
            loss = module_partial_fc(features, label, optimizer)

            # 下述判断是对主干网络的权重更新
            if cfg.fp16:  # 16位于32位混合训练
                # 分布式模型梯度不自动更新,手动放大到32再更新
                grad_amp.scale(loss).backward()
                grad_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5)  # 限制梯度大小
                grad_amp.step(optimizer)  # 更新梯度
                grad_amp.update()  # 对自己状态清0
            else:
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5)  # 限制梯度大小
                optimizer.step()  #

            # 更新分布式模型权重
            lr_scheduler.step()
            with torch.no_grad():
                loss_am.update(loss.item(), 1)  # 干嘛的?
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], grad_amp)

                if global_step % cfg.verbose == 0 and global_step > 200:
                    print(f'global_step={global_step}')
                    callback_verify(global_step, backbone)

        path_pfc = Path(cfg.output) / f"softmax_fc_gpu_{rank}.pt"
        # 保存并行计算的模型
        torch.save(module_partial_fc.state_dict(), str(path_pfc))
        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()
    # 为何最外层还要再重复一次?
    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    dist.destroy_process_group()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument('--config', type=str, default="configs/ms1v3_r18", help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
