import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

train_transform = A.Compose([
    A.Resize(256, 256),
    #A.Resize(224, 224),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    #A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

