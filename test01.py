import numpy as np
import torch
import numpy


# num_classes=104
# world_size=8
# rank=3
# total_label=torch.tensor([1,20,3,50,6])
# num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
# class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
# index_positive = (class_start <= total_label) & (total_label < class_start + num_local)
# print(num_local,class_start)
# print(index_positive)
# print(~index_positive)
#
# str1="Welcome to Netease Youdao, hope you have a good day!"
# str2=list(str1.split(' '))
# print(str2)
def ab(a=1, t=3, b=2, c=3, d=1):
    print(t, a, b, c)


if __name__ == '__main__':
    speed_total=0.1234
    avg=0.1234
    msg= f"Speed {speed_total:.2f} samples/sec Loss {avg:.4f}  "
    a=torch.tensor(3.0).cuda()
    print(a)
    print(float(a))