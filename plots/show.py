import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def create_visual_anno(anno):
    """将0,1,2这样的标签，对应成RGB数组"""
    # 这个8是标签最大值
    assert np.max(anno) <= 5, "only 8 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        # 颜色自定义，这个颜色选的不好
        0: [0,     0, 255],
        1: [255,   0,   0],
        2: [128, 128, 128],
        3: [0,   255,   0],
        4: [192, 192,   0],
        5: [255,  255,  0],
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

def show_img(path):
    """ 读取并展示图片 """
    img = Image.open(path)
    img = np.array(img)
    print('图片的shape:', img.shape)

    img = create_visual_anno(img)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    show_img("D:\1\baseballdiamond45.png")
