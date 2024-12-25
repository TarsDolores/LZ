# -*- coding:utf-8 -*-
# @Time : 2023/6/21 10:32
# @Author : Lei Li

import os
import torch


def save_ckpt(save_dir, timer, model_name, cur_epochs, model, optimizer, scheduler,opts):
#def save_ckpt(save_dir, timer, model_name, cur_epochs, model, optimizer, opts):
    save_dir = save_dir + "/{}_{}/".format(timer, opts.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = save_dir + '/ckpt_{}'.format(model_name) + \
               "_cur_epochs_{}".format(cur_epochs) + ".pth"

    torch.save({
        "cur_epochs": cur_epochs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, save_dir)
    print("Model saved as %s" % save_dir)
