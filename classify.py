# _*_ coding: UTF-8 _*_
# @Time : 2024/12/1 23:03
# @Author : yyj
# @Email : 1410959463@qq.com
# @File : test1.py
# @Project : test

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return image_tensor.to(device)


# 加载模型
def load_model(model_path, num_classes=500):
    model = InceptionResnetV1(classify=True, pretrained=None, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


# 进行分类
def classify_image(model, image_tensor, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
    return predicted_class


# 主函数
def main():
    # 模型路径
    model_path = './data/model/Face_vggface2_checkpoint_20epochs.pth'
    # 图片路径
    image_path = 'test_images/434.jpg'
    # 类别名称
    class_names = [f'class_{i}' for i in range(500)]  # 请根据实际情况更新类别名称

    # 加载模型
    model = load_model(model_path, num_classes=500)

    # 预处理图片
    image_tensor = preprocess_image(image_path)

    # 显示图片
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # 进行分类
    predicted_class = classify_image(model, image_tensor, class_names)
    print(type(predicted_class))
    print(f'Predicted class: {predicted_class}')


if __name__ == '__main__':
    main()
