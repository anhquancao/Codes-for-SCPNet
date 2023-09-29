# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py 

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea


def freeze_layer(layer):
    cnt = 0
    for param in layer.parameters():
        param.requires_grad = False
        cnt += 1
    print("Freeze {} layers".format(cnt))
    layer.eval()


def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']

    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)


    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)


    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    # freeze_layer(cy_fea_net)
    # freeze_layer(cylinder_3d_spconv_seg.downCntx)
    # freeze_layer(cylinder_3d_spconv_seg.resBlock2)
    # freeze_layer(cylinder_3d_spconv_seg.resBlock3)
    # freeze_layer(cylinder_3d_spconv_seg.resBlock4)
    # freeze_layer(cylinder_3d_spconv_seg.resBlock5)
    # freeze_layer(cylinder_3d_spconv_seg.upBlock0)
    # freeze_layer(cylinder_3d_spconv_seg.upBlock1)
    # freeze_layer(cylinder_3d_spconv_seg.upBlock2)
    # freeze_layer(cylinder_3d_spconv_seg.upBlock3)
    # freeze_layer(cylinder_3d_spconv_seg.ReconNet)
# Freeze performance
# Validation per class iou:                                                                                                                                                   
# IoU class 1 [car] = 0.008
# IoU class 2 [bicycle] = 0.000                                                                                                                                               
# IoU class 3 [motorcycle] = 0.000
# IoU class 4 [truck] = 0.000                                                                                                                                                 
# IoU class 5 [other-vehicle] = 0.000
# IoU class 6 [person] = 0.000                                                                                                                                                
# IoU class 7 [bicyclist] = 0.000
# IoU class 8 [motorcyclist] = 0.000                                                                                                                                          
# IoU class 9 [road] = 0.228
# IoU class 10 [parking] = 0.005
# IoU class 11 [sidewalk] = 0.092
# IoU class 12 [other-ground] = 0.000                                                                                                                                         
# IoU class 13 [building] = 0.049
# IoU class 14 [fence] = 0.006                                                                                                                                                
# IoU class 15 [vegetation] = 0.047
# IoU class 16 [trunk] = 0.001                                                                                                                                                
# IoU class 17 [terrain] = 0.040
# IoU class 18 [pole] = 0.001                                                                                                                                                 
# IoU class 19 [traffic-sign] = 0.000                                                                                                                                         
# Current val completion iou is 0.163
# Current val miou is 2.514 while the best val miou is 2.514
# Current val loss is 4.620


    return model
