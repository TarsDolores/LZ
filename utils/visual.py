import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 加载自定义Unet模型
from models import my_Unet

# 定义模型相关参数
num_classes = 6
resolutions = [(252, 252), (126, 126), (56, 56), (28, 28)]
model = my_Unet(3, num_classes, [32, 64, 128, 256, 512], input_resolutions=resolutions)
ckpt = "/home/gws/zdh/hzq/ResU-Former-Pytorch-master/weights/initial_experiments/2024-09-09-07-52-33_vaihingen/ckpt_SobelFormer_cur_epochs_100.pth"
checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])

# 设置设备为 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 注册钩子函数
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 为模型的特定层注册钩子
model.eval()
model.erb_db_1.register_forward_hook(get_activation('bn'))

# 定义图片的预处理函数
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 创建存储特征图的文件夹
save_dir = 'erb_trans_1_train'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义图片文件夹路径
image_folder = "/home/gws/zdh/hzq/ResU-Former-Pytorch-master/dataset/vaihingen/train/images"
image_list = [img for img in os.listdir(image_folder) if img.endswith(".png")]  # 获取所有.png格式的图片
total_images = len(image_list)  # 总图片数量

# 遍历图片文件夹中的所有图片
for idx, image_name in enumerate(image_list):
    # 加载图片
    img_path = os.path.join(image_folder, image_name)
    img = Image.open(img_path).convert('RGB')
    image = np.array(img)

    # 对图片进行预处理
    img = val_transform(image=image)
    img = img['image']
    img = torch.unsqueeze(img, dim=0)  # 增加batch维度
    img = img.to(device)  # 将图像数据传输到 GPU

    # 将图片输入模型并获取特定层的输出
    _ = model(img)
    bn = activation['bn']

    # 去掉 batch_size 和 channel 维度，获取二维特征图
    feature_map = bn[0, 0].cpu().numpy()  # 转换到CPU以保存

    # 使用matplotlib保存特征图
    plt.imshow(feature_map, cmap='gray')
    plt.title(f'{image_name} Feature Map')
    save_path = os.path.join(save_dir, image_name)  # 保存路径
    plt.savefig(save_path)  # 保存特征图
    plt.close()  # 关闭图像以释放内存

    # 计算并显示进度百分比
    percent_complete = (idx + 1) / total_images * 100
    print(f"Processed {image_name} ({idx + 1}/{total_images}) - {percent_complete:.2f}% complete")
