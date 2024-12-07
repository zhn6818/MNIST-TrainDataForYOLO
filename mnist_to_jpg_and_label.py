import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image, ImageOps
import os
import random

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def save_image(filename, data_array):
    bgcolor = (0x00, 0x00, 0xff)
    screen = (500, 375)

    img = Image.new('RGB', screen, bgcolor)
    mnist_img = Image.fromarray(data_array.astype('uint8'))
    mnist_img_invert = ImageOps.invert(mnist_img)

    w = int(mnist_img.width*10)
    mnist_img_invert = mnist_img_invert.resize((w,w))

    x = int((img.width-w)/2)
    y = int((img.height-w)/2)
    img.paste(mnist_img_invert, (x, y))
    img.save(filename)

    return convert((img.width,img.height), (float(x), float(x+w), float(y), float(y+w)))

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 创建必要的目录
DIR_NAME = "JPEGImages"
if not os.path.exists(DIR_NAME):
    os.mkdir(DIR_NAME)

LABEL_DIR_NAME = "labels"
if not os.path.exists(LABEL_DIR_NAME):
    os.mkdir(LABEL_DIR_NAME)

# 初始化计数器字典，用于追踪每个类别的图片数量
class_counter = {i: 0 for i in range(10)}
no = 0  # 文件编号

# 处理训练集和测试集
datasets = [(x_train, y_train), (x_test, y_test)]
for x_data, y_data in datasets:
    for x, y in zip(x_data, y_data):
        # 如果当前类别已经有20张图片，则跳过
        if class_counter[y] >= 20:
            continue
            
        # 生成图片和标签
        filename = f"{DIR_NAME}/{no:05d}.jpg"
        ret = save_image(filename, x)
        
        # 生成标签文件
        label_filename = f"{LABEL_DIR_NAME}/{no:05d}.txt"
        with open(label_filename, 'w') as f:
            f.write(f"{y} {ret[0]} {ret[1]} {ret[2]} {ret[3]}")
        
        print(f"生成类别 {y} 的第 {class_counter[y]+1} 张图片: {filename}")
        
        class_counter[y] += 1
        no += 1
        
        # 检查是否所有类别都已经达到20张
        if all(count >= 20 for count in class_counter.values()):
            break
    
    # 如果所有类别都已经有20张图片，退出外层循环
    if all(count >= 20 for count in class_counter.values()):
        break

# 打印最终统计信息
print("\n生成完成！每个类别的图片数量：")
for digit, count in class_counter.items():
    print(f"数字 {digit}: {count} 张图片")
print(f"总共生成了 {no} 张图片")
