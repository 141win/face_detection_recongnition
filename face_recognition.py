# _*_ coding: UTF-8 _*_
# @Time : 2024/11/25 13:14
# @Author : yyj
# @Email : 1410959463@qq.com
# @File : test.py
# @Project : test

from facenet_pytorch import MTCNN
# inception_resnet_V1是从facenet_pytorch中copy的，避免第一次使用从网上下载模型数据失败。
# 模型参数文件在./data/model/20180402-114759-vggface2.pt
from inception_resnet_v1 import InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import os


def collate_fn(x):
    return x[0]


def save_embeddings(embeddings, names, path='./data/embeddings/embeddings.pt'):
    """
    保存人脸数据
    :param embeddings:
    :param names:
    :param path:
    :return:
    """
    data = {'embeddings': embeddings, 'names': names}
    torch.save(data, path)


def load_embeddings(path='./data/embeddings/embeddings.pt'):
    """
    加载保存的人脸数据
    :param path:
    :return:
    """
    data = torch.load(path)
    return data['embeddings'], data['names']


def compare_with_saved(new_embedding, saved_embeddings, saved_names, threshold=1.0):
    """
    比较两张人脸相似度，小于threshold判定为同一个人
    :param new_embedding:
    :param saved_embeddings:
    :param saved_names:
    :param threshold:
    :return:
    """
    dists = [(new_embedding - emb).norm().item() for emb in saved_embeddings]
    closest_index = dists.index(min(dists))
    if dists[closest_index] < threshold:
        return saved_names[closest_index], dists[closest_index]
    else:
        return "未知", min(dists)


class face_recognition:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.workers = 0 if os.name == 'nt' else 4

    def load_dataset(self, path):
        """
        加载数据集
        :param path:
        :return: 返回加载器、数据集
        """
        # path='./data/test_images'
        dataset = datasets.ImageFolder(path)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=self.workers)
        return loader, dataset

    # 提取特征
    def extract_faces(self, loader, dataset):
        aligned = []
        names = []
        for x, y in loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                print('检测到的人脸及其概率: {:8f}'.format(prob))
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])
        return aligned, names

    def process(self, new_image):
        """
        与已保存数据比较
        Image.open(io.BytesIO(image_data)).convert('RGB')
        传入待比较的照片，且经过上述代码打开转换
        :param new_image:
        :return: name
        """
        # 读取保存的特征
        saved_embeddings, saved_names = load_embeddings()

        new_aligned, _ = self.mtcnn(new_image, return_prob=True)
        print('检测到的人脸及其概率: {:8f}'.format(_))
        if new_aligned is not None:
            new_aligned = new_aligned.unsqueeze(0).to(self.device)
            new_embedding = self.resnet(new_aligned).detach().cpu()
            name, distance = compare_with_saved(new_embedding, saved_embeddings, saved_names)
            print(f'识别结果: {name}, 距离: {distance}')
            return name

    # 比较同一个人不同时期照片
    def compare_process(self, image1, image2):
        """
        比较两人相似度
        Image.open(io.BytesIO(image_data)).convert('RGB')
        输入两张照片，且经过上面代码转换成RGB
        :param image1:
        :param image2:
        :return: 无
        """
        aligned1, _ = self.mtcnn(image1, return_prob=True)
        print('检测到的人脸及其概率：{:8f}'.format(_))
        aligned2, _, landmarks2 = self.mtcnn(image2, return_prob=True)

        if aligned1 is not None and aligned2 is not None:
            aligned1 = aligned1.unsqueeze(0)
            aligned2 = aligned2.unsqueeze(0)
            embedding1 = self.resnet(aligned1).detach().cpu()
            embedding2 = self.resnet(aligned2).detach().cpu()
            distance = (embedding1 - embedding2).norm().item()
            print(distance)
