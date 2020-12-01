# -*- coding: UTF-8 -*-
from __future__ import print_function
import pickle
import os
import sys

import torch
from model_part.model_vatex_fine_classify import get_model
from util.data_classify import DatasetCorrelation, collate_data, DatasetCorrelationVal
import evaluation_vatex
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder
from sklearn import metrics
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
    parser.add_argument('--logger_name', default='/home/fengkai/PycharmProjects/dual_encoding/result/fengkai_vatex_classify/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_768_text_rnn_size_1024_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feat_dim_1024_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_False_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_1', help='Path to save the model and Tensorboard log.')
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
    flag_csv = os.path.join(rootpath, 'vatex/video_text_classify.csv')
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
    caption_files = os.path.join(rootpath, 'vatex/text_embed_info/val_mean_multi_np')
    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # set data loader
    dset = {'val': DatasetCorrelationVal(caption_files, visual_feat_path, flag_csv) }
    data_loaders_val = torch.utils.data.DataLoader(dataset=dset['val'],
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=opt.workers,
                                    collate_fn = collate_data)

    model.val_start()


    # numpy array to keep all the embeddings
    labels = np.zeros((1,1))
    logits = np.zeros((1,2))

    for i, (videoId, text_data, video_data, flag) in enumerate(data_loaders_val):
        # make sure val logger is used

        # compute the embeddings
        with torch.no_grad():
            logit, label= model.forward_emb(videoId, text_data, video_data, flag)

        # initialize the numpy arrays given the size of the embeddings

        labels = np.append(labels, label.data.cpu().numpy().copy())
        logits = np.vstack((logits, (logit.data.cpu().numpy().copy())))
        
        # measure elapsed time
        del video_data, text_data, videoId
    labels = labels[1:]
    logits = logits[1:]
    predict = logits.argmax(1)
    # predict = np.array(list((map(lambda x:np.where(x<0.5, 0, 1),logits))))
    # predict = np.array(list((map(lambda x:np.where(x[0]<x[1], 0, 1),logits))))
    acc = metrics.accuracy_score(labels, predict)
    pre = metrics.precision_score(labels, predict)
    recall = metrics.recall_score(labels, predict)
    f1 = metrics.f1_score(labels, predict)
    print(' * score: ', (acc, pre, recall, f1))



if __name__ == '__main__':
    main()
