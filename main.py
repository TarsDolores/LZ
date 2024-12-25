# -*- coding:utf-8 -*-
# @Time : 2023/5/23 15:30
# @Author : Lei Li
from models import DAnet
from models import fcn_resnet50
from models import SwinTransformerSys
from models import VisionTransformer
from models import brats
from models import ours
from models import sobel_swint
import utils
import os
import random
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import logging
from models import my_Unet
import ml_collections
from torch.utils import data
from datasets import Remote
from metrics import StreamSegMetrics
from models import sobel_swint

from utils import train_transform, val_transform, validate, SaveLogs, save_ckpt
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
from pytorch_toolbelt import losses as L
import torch.optim.lr_scheduler as lr_scheduler

numclasses = 6


def get_argparser():
    parser = argparse.ArgumentParser()
    # 数据集地址
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/Unet/dataset/potsdam/",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='potsdam_SIM_FCM',
                        help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=numclasses,
                        help="num classes (default: None)")
    # 测试
    parser.add_argument("--test_only", action='store_true', default=False)

    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument('--num_epochs', type=int, default=160,
                        # 55
                        help='total training epochs.')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')

    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='val batch size (default: 16)')

    # 加载已经训练好的权重
    parser.add_argument("--ckpt", default="", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=20,
                        help="print interval of loss (default: 10)")

    # 自定义每多少个inters进行一次val
    parser.add_argument("--val_epochs", type=int, default=20,
                        help="epoch interval for eval (default: 100)")

    # parser.add_argument("--logs_items", type=str,
    #                     default=['Epoch', 'loss', 'Overall Acc', 'Mean IoU', 'Avg F1'] + [str(i) for i in
    #                                                                                       range(numclasses)],
    #                     help="logs items")

    parser.add_argument("--logs_items", type=str,
                        default=['Epoch', 'loss', 'Overall Acc', 'Mean IoU', 'Avg F1'] +
                                [f'IoU_{i}' for i in range(numclasses)] +  # 每个类别的 IoU
                                [f'F1_{i}' for i in range(numclasses)],  # 每个类别的 F1
                        help="logs items")

    return parser


def get_dataset(opts):
    train_dst = Remote(root=opts.data_root,
                       split='train', transform=train_transform)
    val_dst = Remote(root=opts.data_root,
                     split='val', transform=val_transform)
    return train_dst, val_dst


def setup_logger(log_dir, opts):
    # 获取当前时间戳，用于生成唯一日志文件名
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_filename = os.path.join(log_dir, f"log_{timestamp}.log")

    # 创建日志记录器
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    # 创建文件处理器，将日志写入文件
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)

    # 记录超参数（opts）
    logger.info("Training started with the following options:")
    for key, value in opts.items():
        logger.info(f"{key}: {value}")

    return logger


def main():
    opts = get_argparser().parse_args()
    timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    log = SaveLogs(opts.logs_items, timer, opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
        drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    def get_b16_config():
        """Returns the ViT-B/16 configuration."""
        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict({'size': (16, 16)})
        config.hidden_size = 768
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 3072
        config.transformer.num_heads = 12
        config.transformer.num_layers = 1
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.1
        # config.resnet.att_type = 'CBAM'
        config.classifier = 'seg'
        config.representation_size = None
        config.resnet_pretrained_path = None
        # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
        config.patch_size = 16

        config.decoder_channels = (256, 128, 64, 16)
        config.n_classes = 2
        config.activation = 'softmax'
        return config

    def get_r50_b16_config():
        """Returns the Resnet50 + ViT-B/16 configuration.-------------------------wo yong de """
        config = get_b16_config()
        config.data = ml_collections.ConfigDict()
        config.data.img_size = 256  # 6144
        config.data.in_chans = 3
        config.n_classes = 6
        config.patches.grid = (4, 4)
        config.resnet = ml_collections.ConfigDict()
        config.resnet.num_layers = (3, 4, 6, 3)
        config.resnet.width_factor = 0.5

        config.classifier = 'seg'
        config.trans = ml_collections.ConfigDict()
        config.trans.num_heads = [3, 6, 12, 24]
        config.trans.depths = [2, 2, 6, 2]
        config.trans.embed_dim = 96
        config.trans.window_size = 8

        # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz' #yuxunlian
        config.decoder_channels = (
            512, 256, 128, 64)  # (256,128,64,16)##1024,512,256,128,64)#(2048,1024,512,256,128)#(256, 128, 64, 16)
        config.skip_channels = [512, 256, 128,
                                64]  # [256,128,64,16]#[512,256,128,64,16]#[512,256,128,64,32]#[1024,512,256,128,64]#[512, 256, 64, 16]
        config.n_classes = 6
        config.n_skip = 4
        config.activation = 'softmax'

        return config

    config_vit = get_r50_b16_config()
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    model = VisionTransformer(config_vit, img_size=256, num_classes=6)

    #model = my_U_Net(3, opts.num_classes)

    # model = SwinTransformerSys(
    #     img_size=224,  # 输入图像的尺寸
    #     patch_size = 4,  # 每个patch的尺寸
    #     in_chans = 3,  # 输入图像的通道数（例如RGB图像）
    #     num_classes = 6, # 类别数，通常为数据集中类别的总数
    #     embed_dim = 96,  # 嵌入维度
    #     depths = [2, 2, 2, 2],  # 每层的深度
    #     num_heads = [3, 6, 12, 24],  # 每层的注意力头数
    #     window_size = 7,
    #     mlp_ratio = 4,
    #     qkv_bias = True,
    #     qk_scale = None,
    #     drop_rate = 0.0,
    #     drop_path_rate = 0.1,
    #     ape = False,
    #     patch_norm = None,
    #     use_checkpoint = False,
    # )
    #model = fcn_resnet50(6, False)
    #model = UNetFormer(pretrained=False)
    #model = DAnet(6)
    #model = brats()
    # model = smp.Unet(
    #      encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #      encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #      in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #      classes=6,  # model output channels (number of classes in your dataset)
    #  )

   # resolutions = [(252, 252), (126, 126), (56, 56), (28, 28)]
   # model = sobel_swint(3, 6, [32, 64, 128, 256, 512], input_resolutions=resolutions)
    metrics = StreamSegMetrics(opts.num_classes)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.num_epochs, eta_min=0)

    loss = "SoftCE_dice1"
    if loss == "SoftCE_dice":
        DiceLoss_fn = DiceLoss(mode='multiclass')
        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
        criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                                first_weight=0.5, second_weight=0.5).cuda()
    else:
        LovaszLoss_fn = LovaszLoss(mode='multiclass')
        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
        criterion = L.JointLoss(first=LovaszLoss_fn, second=SoftCrossEntropy_fn,
                                first_weight=0.5, second_weight=0.5).cuda()
    #criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()

    utils.mkdir('checkpoints')

    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epochs = checkpoint["cur_epochs"]
            print("Training state restored from %s" % opts.ckpt)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print("Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        model = nn.DataParallel(model)
        model.to(device)

    if opts.test_only:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:
        model.train()
        cur_epochs += 1
        for (images, labels, _) in train_loader:
            since = time.time()
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            # print(loss)

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % opts.print_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                time_elapsed = time.time() - since
                interval_loss = interval_loss / opts.print_interval
                print("Epoch %d/%d, Itrs %d/%d, Loss=%f ,time=%.2f, LR=%.6f" %
                      (cur_epochs, opts.num_epochs, cur_itrs % (len(train_dst) // opts.batch_size),
                       len(train_dst) // opts.batch_size, interval_loss, time_elapsed, current_lr))
                interval_loss = 0.0

        if (cur_epochs) % opts.val_epochs == 0:
            save_ckpt("weights", timer, 'SobelFormer', cur_epochs, model, optimizer, scheduler, opts)
            print("validation...")
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            model.eval()
            val_score = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            # logs_list = [cur_epochs, np_loss, val_score['Overall Acc'],
            #              val_score['Mean IoU'], val_score['Avg F1']] + \
            #             [val_score['Class IoU'][i] for i in range(numclasses)]
            # log.update(logs_list)

            logs_list = [cur_epochs, np_loss, val_score['Overall Acc'],
                         val_score['Mean IoU'], val_score['Avg F1']] + \
                        [val_score['Class IoU'][i] for i in range(numclasses)] + \
                        [val_score['Class F1'].get(i,0) for i in range(numclasses)]  # 新增：每类 F1 值
            log.update(logs_list)

            model.train()
        scheduler.step()

        if cur_epochs >= opts.num_epochs:
            return


if __name__ == '__main__':
    main()
