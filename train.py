# _*_ coding: UTF-8 _*_
# @Time : 2024/12/1 20:49
# @Author : yyj
# @Email : 1410959463@qq.com
# @File : test.py
# @Project : test

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

import matplotlib.pyplot as plt

data_dir = './data/test_images'

batch_size = 64
epochs = 20
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

resnet = InceptionResnetV1(
    classify=True,  # 使用分类器
    pretrained='vggface2',  # 预训练模型
    num_classes=500  # 分类数
).to(device)

# 优化器、调度器
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])
# 数据集和数据加载器
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]
# 训练数据加载器
train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
# 测试数据加载器
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

# 用于存储训练过程中的损失和准确率
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

print('\n\n初始化')
print('-' * 10)
resnet.eval()
val_metrics = training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

val_losses.append(val_metrics[0])
val_accuracies.append(val_metrics[1]['acc'])

for epoch in range(epochs):
    print('\n循环 {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()  # 训练模式
    train_metrics = training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    train_losses.append(train_metrics[0])
    train_accuracies.append(train_metrics[1]['acc'])

    resnet.eval()  # 评估模式
    val_metrics = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    val_losses.append(val_metrics[0])
    val_accuracies.append(val_metrics[1]['acc'])
model_path = './data/model/Face_vggface2_checkpoint.pth'
torch.save(resnet.state_dict(), model_path)  # 保存模型
writer.close()

# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 2), val_losses, label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 2), val_accuracies, label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
