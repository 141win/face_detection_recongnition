# _*_ coding: UTF-8 _*_
# @Time : 2024/12/2 17:34
# @Author : yyj
# @Email : 1410959463@qq.com
# @File : face_detection.py
# @Project : test


from facenet_pytorch import MTCNN, training
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

data_dir = './data/wujing'  # './data/test_images'

batch_size = 1
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
    for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\r第 {} 批，共 {} 批'.format(i + 1, len(loader)), end='')

del mtcnn
