# -*- coding: UTF-8 -*-
from __future__ import print_function
import pickle
import os
import sys

import torch
from util.vatex_dataloader import Dataset2BertI3d
import evaluation_vatex
from model_part.model_vatex import get_model
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder

import logging
import json
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from basic.constant import ROOT_PATH

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='/home/fengkai/PycharmProjects/dual_encoding/result/fengkai_vatex_bert/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_768_text_rnn_size_1024_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feat_dim_1024_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_False_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=1, help='number of captions of each image/video (default: 1)')

    args = parser.parse_args()
    return args

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    n_caption = opt.n_caption
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    # 模型加载
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    # 文件名称
    visual_feat_path = os.path.join(rootpath, 'vatex/video_embed_info/val_video') 
    caption_files = os.path.join(rootpath, 'vatex/text_embed_info/val')
    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # set data loader
    dset = {'val': Dataset2BertI3d(caption_files, visual_feat_path, videoEmbed_num = 32) }
    data_loaders_val = torch.utils.data.DataLoader(dataset=dset['val'],
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=opt.workers)
    video_embs, cap_embs, video_ids, ch_caps = evaluation_vatex.encode_data(model, data_loaders_val, opt.log_step, logging.info)
    #embedding 的可视化分析 
    tensor_show = torch.cat((video_embs.data, torch.ones(len(video_embs), 1)), 1)
    with SummaryWriter(log_dir='./results', comment='embedding——show') as writer: 
            writer.add_embedding(
                video_embs.data,
                label_img=cap_embs.data,
                global_step=1)
    c2i_all_errors = evaluation_vatex.cal_error(video_embs, cap_embs, options.measure)

     # caption retrieval
    (r1i, r5i, r10i, medri, meanri) = evaluation_vatex.t2i(c2i_all_errors, n_caption=n_caption)
    t2i_map_score = evaluation_vatex.t2i_map(c2i_all_errors, n_caption=n_caption)

    # video retrieval
    (r1, r5, r10, medr, meanr) = evaluation_vatex.i2t(c2i_all_errors, n_caption=n_caption)
    i2t_map_score = evaluation_vatex.i2t_map(c2i_all_errors, n_caption=n_caption)

    print(" * Text to Video:")
    print(" * r_1_5_10, medr, meanr: {}".format([round(r1i, 1), round(r5i, 1), round(r10i, 1), round(medri, 1), round(meanri, 1)]))
    print(" * recall sum: {}".format(round(r1i+r5i+r10i, 1)))
    print(" * mAP: {}".format(round(t2i_map_score, 3)))
    print(" * "+'-'*10)

    # caption retrieval
    print(" * Video to text:")
    print(" * r_1_5_10, medr, meanr: {}".format([round(r1, 1), round(r5, 1), round(r10, 1), round(medr, 1), round(meanr, 1)]))
    print(" * recall sum: {}".format(round(r1+r5+r10, 1)))
    print(" * mAP: {}".format(round(i2t_map_score, 3)))
    print(" * "+'-'*10)



if __name__ == '__main__':
    main()
