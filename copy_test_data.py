import numpy as np
import pandas as pd
import random
import os

category = []
id = []
for i in range(700):
    id.append(i)
    # 从0到499中随机选择一个数
    category.append(random.randint(0, 499))
df = pd.DataFrame({'id': id,
                   'category': category
                   })
print(df)
df.to_csv("./data/answer2.csv", index=False)


def copy_file(src, dst):
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        fdst.write(fsrc.read())


# 根据df中的category值从路径为“./data/test_images_cropped”的文件夹中选择文件夹名转换成int数据后与category值匹配的的文件夹中选择任选一张图片，然后将名字按照对应的id命名存入test_images文件夹中
for i in range(len(df)):
    name = df['category'][i]
    if name < 10:
        name_str = '00' + str(name)
    elif name < 100:
        name_str = '0' + str(name)
    else:
        name_str = str(name)
    path = f"./data/test_images_cropped/{name_str}"
    file_list = os.listdir(path)
    file_name = random.choice(file_list)
    print(f"{i}: {file_name}")
    copy_file(f"{path}/{file_name}", f"./data/test_images2/{i}.jpg")
