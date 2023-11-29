import importlib
import os
import logging
import sys


class AverageMeter:
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    module_name = config_file.split('/')[1]
    job_config = importlib.import_module(f'configs.{module_name}').config

    if job_config.output is None:
        job_config.output = os.path.join('work_dirs', module_name)

    return job_config


def init_logging(models_root):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)
    log_root.info(f'rank_id: {0}')
