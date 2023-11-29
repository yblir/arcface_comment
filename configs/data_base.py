import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def dataset_info(config):
    if config.dataset == "emore":
        config.rec = "/train_tmp/faces_emore"
        config.num_classes = 85742
        config.num_image = 5822653
        config.num_epoch = 16
        config.warmup_epoch = -1
        config.decay_epoch = [8, 14, ]
        config.val_targets = ["lfw", ]

    elif config.dataset == "ms1m-retinaface-t1":
        config.rec = "/train_tmp/ms1m-retinaface-t1"
        config.num_classes = 93431
        config.num_image = 5179510
        config.num_epoch = 25
        config.warmup_epoch = -1
        config.decay_epoch = [11, 17, 22]
        config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    elif config.dataset == "glint360k":
        config.rec = "/train_tmp/glint360k"
        config.num_classes = 360232
        config.num_image = 17091657
        config.num_epoch = 20
        config.warmup_epoch = -1
        config.decay_epoch = [8, 12, 15, 18]
        config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    elif config.dataset == "webface":
        config.dataset_path = "/home/train_tmp/faces_webface_112x112"
        config.num_classes = 93431
        config.num_image = 5179510
        config.num_epoch = 25
        config.warmup_epoch = 2
        config.decay_epoch = [10, 16, 22]
        # config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
        config.val_targets = ["lfw"]

    return config
