import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        '''
        max_prefetch:指队列长度, 指预先载入显存的数据量
        '''
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        # todo train中已经执行过这个命令, 为何这个步骤还要再次执行?
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:  # 队列取完, 结束迭代
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


class DataLoaderX(DataLoader):
    '''
    重构dataloader, 在内部开起新线程异步不断加载数据到Queue ,
    外部使用相同接口迭代数据时，就直接从Queue中取，而不需要取得时候才着手数据前处理
    '''

    def __init__(self, local_rank, **kwargs):
        '''
        local_rank: 当前节点,正在使用的gpu编号. rank是分布式系统上全局进程编号,与local rank不同
        '''
        super(DataLoaderX, self).__init__(**kwargs)
        # todo 创建流文件 ?
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def preload(self):
        '''
        预加载,将数据读取到显存中
        '''
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None

        with torch.cuda.stream(self.stream):
            for i in range(len(self.batch)):
                # todo non_blocking这个参数什么意思 ?
                self.batch[i] = self.batch[i].to(device=self.local_rank, non_blocking=True)

    def __iter__(self):
        # todo 为什么iter中也要重载父类?
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()

        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        # 提取当前batch数据. 再调用preload方法,更新下一batch的数据
        # cur_batch是个list,有两个元素. 每个元素是个tensor长度=batch_size.
        # 0:shape=(128,3,112,112),1:shape=(128,)
        cur_batch = self.batch
        if cur_batch is None:
            raise StopIteration
        self.preload()

        return cur_batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # MXNet的rec文件能够用来保存任意的二进制数据, idx是每个二进制文件的索引编号
        rec_path = os.path.join(root_dir, 'train.rec')
        idx_path = os.path.join(root_dir, 'train.idx')
        # 打乱次序读取数据
        self.img_rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

        # 获取指定idx的数据
        s = self.img_rec.read_idx(0)
        # 解压rec文件, header应该是标签信息等, 里面的flag是标签的数量?
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:  # todo 这是干嘛?
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.img_idx = np.array(range(1, int(header.label[0])))
        else:
            self.img_idx = np.array(list(self.img_rec.keys))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        idx = self.img_idx[index]
        s = self.img_rec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label

        # todo 这又是干嘛的?
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        img_arr = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            img_arr = self.transform(img_arr)
        return img_arr, label


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(batch_size: int, rec_file: str, idx_file: str, num_threads: int,
                   initial_fill=32768, random_shuffle=True,
                   prefetch_queue_depth=1, local_rank=0, name="reader",
                   mean=(127.5, 127.5, 127.5),
                   std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=local_rank,
                    prefetch_queue_depth=prefetch_queue_depth)
    condition_flip = fn.random.coin_flip(probability=0.5)

    with pipe:
        jpegs, labels = fn.readers.mxnet(path=rec_file,
                                         index_path=idx_file,
                                         initial_fill=initial_fill,
                                         num_shards=world_size,
                                         shard_id=rank,
                                         random_shuffle=random_shuffle,
                                         pad_last_batch=False,
                                         name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(images,
                                          dtype=types.FLOAT,
                                          mean=mean,
                                          std=std,
                                          mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()

    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name))


@torch.no_grad()
class DALIWarper:
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()


def get_dataloader(root_dir: str, local_rank: int, batch_size: int, dali=False) -> Iterable:
    if dali and root_dir != "synthetic":
        rec = os.path.join(root_dir, 'train.rec')
        idx = os.path.join(root_dir, 'train.idx')
        return dali_data_iter(batch_size=batch_size,
                              rec_file=rec,
                              idx_file=idx,
                              num_threads=2,
                              local_rank=local_rank)
    else:
        if root_dir == "synthetic":
            train_set = SyntheticDataset()
        else:
            train_set = MXFaceDataset(root_dir=root_dir)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoaderX(local_rank=local_rank,
                                   dataset=train_set,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)
        return train_loader
