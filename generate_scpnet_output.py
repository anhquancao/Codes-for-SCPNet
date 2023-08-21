# -*- coding:utf-8 -*-
# author: Xinge, Xzy
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

torch.set_grad_enabled(False)

# from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader_rec.dataloader.pc_dataset import get_SemKITTI_label_name, get_eval_mask, unpack

from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings
from utils.np_ioueval import iouEval
import yaml

warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    model_load_path += 'iou37.5557_epoch3.pth'
    # model_load_path += '0.pth'
    model_save_path += ''
    if os.path.exists(model_load_path):
        print('Load model from: %s' % model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        print('No existing model, training model from scratch...')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print(model_save_path)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader, train_pt_dataset, val_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  generate=True,
                                                                  use_tta=False,
                                                                  use_multiscan=True)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    # learning map
    with open("config/label_mapping/semantic-kitti.yaml", 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    class_strings = semkittiyaml["labels"]
    class_inv_remap = semkittiyaml["learning_map_inv"]

    dataloaders = [train_dataset_loader, val_dataset_loader]
    datasets = [train_pt_dataset, val_pt_dataset]
    
    output_path = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti/scpnet_output"
    my_model.eval()
    for i in range(2):
        dataset_loader = dataloaders[i]
        pt_dataset = datasets[i] 
        pbar = tqdm(total=len(dataset_loader))
        for i_iter_val, (_, val_vox_label, val_grid, _, val_pt_fea, val_index, origin_len) in tqdm(enumerate(dataset_loader)):
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                            val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]

            assert val_batch_size == 1
            for bat in range(val_batch_size):

                val_label_tensor = val_vox_label[bat,:].type(torch.LongTensor).to(pytorch_device)
                val_label_tensor = torch.unsqueeze(val_label_tensor, 0)
                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                

                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                predict_labels = np.squeeze(predict_labels).astype(np.uint8)
                
                save_dir = pt_dataset.im_idx[val_index[0]]
                _,dir2 = save_dir.split('/sequences/',1)
                new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne', 'predictions')[:-3]+'npy'
                
                os.makedirs(os.path.dirname(new_save_dir), exist_ok=True)
                np.save(new_save_dir, predict_labels)
                
                # val_vox_label0 = val_vox_label[bat, :].cpu().detach().numpy()
                # val_vox_label0 = np.squeeze(val_vox_label0)
                
                # val_name = val_pt_dataset.im_idx[val_index[0]]
                
                # invalid_name = val_name.replace('velodyne', 'voxels')[:-3]+'invalid'
                # invalid_voxels = unpack(np.fromfile(invalid_name, dtype=np.uint8))  # voxel labels
                # invalid_voxels = invalid_voxels.reshape((256, 256, 32))
                # masks = get_eval_mask(val_vox_label0, invalid_voxels)
                # predict_labels = predict_labels[masks]
                # val_vox_label0 = val_vox_label0[masks]
                
                pbar.update(1)
                
                
                
                

   


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti-multiscan.yaml')
    # parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
