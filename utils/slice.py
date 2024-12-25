# -*- coding:utf-8 -*-
# @Time : 2022/12/27 16:16
# @Author : Lei Li


import torch
import math
from torchvision import transforms
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_slice(model, images):
    preds = []
    model.eval()
    for i, img in enumerate(images):
        pred = predict_patches2(model, img)
        preds.append(torch.from_numpy(pred))
    return torch.tensor([item.cpu().detach().numpy() for item in preds]).cuda()

def predict_patches2(model, img_,
                     threshold = 200,
                     dilate = True, dilate_size=64,
                     target_h = 128, target_w = 128,
                     crop_space = 64,
                     target_h_dilate=128, target_w_dilate=128
                    ):
    """
        切片预测函数，先切片，再预测，再拼接还原。

        Args:
            threshold:             分辨率阈值
            target_h:              自定义裁剪尺寸
            target_w:              自定义裁剪尺寸
            dilate_size:           上下左右膨胀像素

        Returns: 图片最终预测结果

        """
    with torch.no_grad():
        # 1.得到数据shape
        bands, h, w = 0, 0, 0
        if len(img_.shape) == 3:
            bands, h, w = img_.shape
        elif len(img_.shape) == 2:
            img_ = np.array([img_])
            bands, h, w = img_.shape

        image = img_.detach().cpu().numpy()

        # threshold = THRESHOLD
        if w <= threshold and h <= threshold:
            dilate_size = 0
            dilate = False
            target_h = h
            target_w = w
        else:
            # 裁剪大小自适应地从(5~7)*crop_space选取,(5~7)可更改这里自定义
            crop_size = np.array([crop_space * i for i in range(9, 4, -1)])
            pad_h = (crop_size - h % crop_size) % crop_size
            pad_w = (crop_size - w % crop_size) % crop_size
            left_h = crop_size - pad_h
            left_w = crop_size - pad_w
            score_h = list(zip(crop_size, left_h, pad_h))
            score_w = list(zip(crop_size, left_w, pad_w))
            score_h_sort = sorted(score_h, key=lambda x: (-x[1], x[2]))
            score_w_sort = sorted(score_w, key=lambda x: (-x[1], x[2]))
            best_h = score_h_sort[0][0]
            best_w = score_w_sort[0][0]

            target_h_dilate = best_h
            target_w_dilate = best_w

            # # 自定义了
            # target_h_dilate = 64
            # target_w_dilate = 64

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dilate:
            cropped, rows, cols = crop_with_dilate_(image, dilate_size, target_h_dilate, target_w_dilate)
        else:
            cropped, rows, cols = crop_without_dilate_(image, target_h, target_w)

        if h <= threshold and w <= threshold:
            image = torch.from_numpy(cropped).to(device, dtype=torch.float32)

            # 模型预测
            pred = model(image)
            pred = pred.cpu().detach().numpy()
        else:
            images = torch.from_numpy(cropped)
            all_pred = None
            # print(images.shape)
            for image in images:
                bands, h_1, w_1 = image.shape
                _biliner_resize = transforms.Resize(size=(256,256), interpolation=transforms.InterpolationMode.BILINEAR)
                image = _biliner_resize(image)

                # 模型预测
                pred = model(image.to(device, dtype=torch.float32).unsqueeze(0))
                torch.cuda.empty_cache()

                _biliner_resize_ = transforms.Resize(size=(h_1, w_1), interpolation=transforms.InterpolationMode.BILINEAR)
                pred = _biliner_resize_(pred)

                pred = pred.cpu().detach().numpy()

                all_pred = np.concatenate((all_pred, pred), axis=0) if all_pred is not None else pred

            pred = all_pred

        if dilate:
            final = concat_with_dilate_(h, w, cols, pred, dilate_size, target_h_dilate, target_w_dilate)
        else:
            final = concat_without_dilate_(h, w, cols, pred, target_h, target_w)

    return final


def crop_with_dilate_(img, dilate_size, target_h, target_w):
    _, h, w = img.shape
    container = []
    # fill right and bottom
    pad_w = target_w - (w % target_w) if (w % target_w) != 0 else 0
    pad_h = target_h - (h % target_h) if (h % target_h) != 0 else 0

    pad_img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0)

    # dilate all top/left/right/bottom
    pad_img = np.pad(pad_img, ((0, 0), (dilate_size, dilate_size), (dilate_size, dilate_size)), 'constant', constant_values=0)
    pad_img = torch.from_numpy(pad_img.astype(np.float32))

    # container: B C H W  ndarray
    for i in range(math.ceil(h / target_h)):
        for j in range(math.ceil(w / target_w)):
            crop = pad_img[:, i * target_h: (i + 1) * target_h + 2 * dilate_size,
                   j * target_w: (j + 1) * target_w + 2 * dilate_size]
            container.append(crop.numpy())
    return np.array(container), math.ceil(h / target_h), math.ceil(w / target_w)


def crop_without_dilate_(img, target_h, target_w):
    _, h, w  = img.shape
    container = []

    # fill right and bottom
    pad_w = target_w - (w % target_w) if (w % target_w) != 0 else 0
    pad_h = target_h - (h % target_h) if (h % target_h) != 0 else 0
    pad_img = np.pad(img, ((0, 0),(0, pad_h), (0, pad_w)), 'constant', constant_values=0)  # 每一维度如何填充
    pad_img = torch.from_numpy(pad_img.astype(np.float32))

    # container: B C H W  ndarray
    for i in range(math.ceil(h / target_h)):
        for j in range(math.ceil(w / target_w)):
            crop = pad_img[:, i * target_h: (i + 1) * target_h, j * target_w: (j + 1) * target_w]
            container.append(crop.numpy())
    return np.array(container), math.ceil(h / target_h), math.ceil(w / target_w)


def concat_with_dilate_(h, w, cols, pred, dilate_size, target_h, target_w):
    bs, c, _, _ = pred.shape
    final = np.zeros((c, math.ceil(h / target_h) * target_h, math.ceil(w / target_w) * target_w))
    row, col = 0, 0
    for i in range(bs):
        crop = pred[i, :, dilate_size:dilate_size + target_h, dilate_size:dilate_size + target_w]
        final[:, row * target_h: (row + 1) * target_h, col * target_w: (col + 1) * target_w] = crop
        col = col + 1
        if col % cols == 0:
            row += 1
            col = 0
    final = final[:, 0: h, 0: w]

    return final


def concat_without_dilate_(h, w, cols, pred, target_h, target_w):
    bs, c, _, _ = pred.shape
    final = np.zeros((c, math.ceil(h / target_h) * target_h, math.ceil(w / target_w) * target_w))
    row, col = 0, 0
    for i in range(bs):
        crop = pred[i, :, :, :]
        final[:, row * target_h: (row + 1) * target_h, col * target_w: (col + 1) * target_w] = crop
        col = col + 1
        if col % cols == 0:
            row += 1
            col = 0
    final = final[:, 0: h, 0: w]

    return final
