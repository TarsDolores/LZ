# -*- coding:utf-8 -*-
# @Time : 2023/6/21 10:35
# @Author : Lei Li


import pandas as pd
from collections import OrderedDict
import os


class SaveLogs(object):
    def __init__(self, logs_items, timer,args):
        super(SaveLogs, self).__init__()
        self.items = logs_items
        self.log = OrderedDict()
        self.logs_dir = 'logs/%s_%s/' % (timer,args.dataset)
        self.add_items()
        self.args = args
        self.save_hyperparameters()

    def add_items(self):
        for k in self.items:
            self.log[k] = []

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def update(self, logs_list):
        for i, k in enumerate(self.log):
            if k == 'Epoch':
                self.log[k].append(logs_list[i])
            else:
                self.log[k].append(f'{logs_list[i]:.4f}')
            #self.log[k].append(logs_list[i])
        # Save
        pd.DataFrame(self.log).to_csv(self.logs_dir+'log.csv', index=False)

    def save_hyperparameters(self):
        hyperparams = {
            "data_root": self.args.data_root,
            "dataset": self.args.dataset,
            "num_classes": self.args.num_classes,
            "test_only": self.args.test_only,
            "num_epochs": self.args.num_epochs,
            "lr": self.args.lr,
            "batch_size": self.args.batch_size,
            "ckpt": self.args.ckpt,
            "gpu_id": self.args.gpu_id,
            "val_epochs": self.args.val_epochs,
            "continue_trianing":self.args.continue_training
        }
        # 保存为TXT文件
        hyperparams_filename = os.path.join(self.logs_dir, 'opts.txt')
        with open(hyperparams_filename, 'w') as f:
            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")  # 每行写入一个键值对

        print(f"Hyperparameters saved to {hyperparams_filename}")
