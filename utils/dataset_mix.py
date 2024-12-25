import os
import random
import shutil
from tqdm import tqdm


# 随机选择20%的图片和对应标签进行复制
def select_and_copy_files(val_image_dir, val_label_dir, train_image_dir, train_label_dir, percentage=0.5):
    # 获取所有图片和标签文件
    image_filenames = sorted(os.listdir(val_image_dir))
    label_filenames = sorted(os.listdir(val_label_dir))

    # 确保 image 和 label 文件名对应
    assert len(image_filenames) == len(label_filenames), "Image and label counts do not match."

    # 随机选择 20% 的文件
    num_to_select = int(len(image_filenames) * percentage)
    selected_indices = random.sample(range(len(image_filenames)), num_to_select)

    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)

    # 复制图片和标签到 train 文件夹
    for idx in tqdm(selected_indices, desc="Copying files from val to train"):
        img_file = image_filenames[idx]
        lbl_file = label_filenames[idx]

        # 确保图片与标签对应
        img_path = os.path.join(val_image_dir, img_file)
        lbl_path = os.path.join(val_label_dir, lbl_file)

        # 目标路径
        img_target = os.path.join(train_image_dir, img_file)
        lbl_target = os.path.join(train_label_dir, lbl_file)

        # 复制图片和标签
        shutil.copy(img_path, img_target)
        shutil.copy(lbl_path, lbl_target)


# 主处理函数
def process_dataset(input_root):
    val_image_dir = os.path.join(input_root, 'val', 'images')
    val_label_dir = os.path.join(input_root, 'val', 'labels')

    train_image_dir = os.path.join(input_root, 'train', 'images')
    train_label_dir = os.path.join(input_root, 'train', 'labels')

    # 执行文件选择和复制
    select_and_copy_files(val_image_dir, val_label_dir, train_image_dir, train_label_dir)


# 输入路径
input_root = r"/root/autodl-tmp/Unet/dataset/potsdam/"

# 执行文件选择和复制
process_dataset(input_root)
