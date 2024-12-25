# -*- coding:utf-8 -*-
# @Time : 2023/6/07 15:30
# @Author : Lei Li

from tqdm import tqdm
import os
import torch
from PIL import Image
import numpy as np
from .slice import infer_slice


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        results_dir = os.path.join('results', opts.dataset)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    with torch.no_grad():
        for i, (images, labels, file_names) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            #outputs2 = infer_slice(model, images)
            #outputs = outputs1 + outputs2
            preds = outputs.detach().max(dim=1)[1].cpu().numpy().astype(np.uint8)
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                for i, nm in enumerate(file_names):
                    Image.fromarray(preds[i]).save(os.path.join(results_dir,'%s.png' % nm))

        score = metrics.get_results()
    return score
