from easydict import EasyDict
from .data_base import dataset_info

config = EasyDict()

config.margin_list = (1.0, 0.5, 0.0)
config.loss = 'arcface'
config.network = 'r18'
config.resume = None  # 不需要加载模型权重
config.output = None  # 不设置输出文件夹

config.embedding_size = 512  # 设置模型输出向量的维度,通常为256,512

config.fp16 = True  # 设置与float32进行混合精度训练,加快训练速度
config.batch_size = 4

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

# For SGD
config.optimizer = "sgd"
config.lr = 0.1  # 设置学习率
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10

# 数据集相关
# For Large Sacle Dataset, such as WebFace42M
config.dali = False
config.dataset = 'webface'
config = dataset_info(config)
