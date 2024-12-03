# _*_ coding: UTF-8 _*_
# @Time : 2024/12/2 18:22
# @Author : yyj
# @Email : 1410959463@qq.com
# @File : val_test.py
# @Project : test
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def main():
    df = pd.read_csv('answer1.csv')
    # 模型路径
    model_path = 'Face_vggface2_checkpoint_20epochs.pth'
    # 加载模型
    model = load_model(model_path, num_classes=500)
    # pre_class = []
    # pre_id = []
    acc_num = 0
    for i in range(700):
        image_path = f"test_images1/{i}.jpg"
        # 预处理图片
        image_tensor = preprocess_image(image_path)
        class_names = [i for i in range(500)]
        # 进行分类
        predicted_class = classify_image(model, image_tensor, class_names)
        if predicted_class == df['category'][i]:
            acc_num += 1
    # print(test_acc(df, pre_df))
    print("准确率：", acc_num / 700)


if __name__ == "__main__":
    main()
