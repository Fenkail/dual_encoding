# -*- coding: UTF-8 -*-
from __future__ import print_function
import pickle
import os
import sys
import time
import torch
from util.vatex_dataloader import Dataset2BertI3d, collate_data
import evaluation_vatex
from model_part.model_vatex_fine import get_model
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder
import random
import logging
import json
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from basic.constant import ROOT_PATH
from scipy.spatial import distance
from util.text2vec_bert import text_embed
from util.jieba_pos import PartOfSpeech
import sys
sys.path.append('/home/fengkai/PycharmProjects/kinetics-i3d-pytorch')
from extract_features_other import extract_video

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='/home/fengkai/PycharmProjects/dual_encoding/result/fengkai_vatex_multi-cyc/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_768_text_rnn_size_1024_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feat_dim_1024_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_False_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_23_gru+cluster/', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
   
    args = parser.parse_args()
    return args

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']

def evaluation(model, videoId, text_data, video_data):
    model.val_start()

    # compute the embeddings
    vid_emb, cap_emb, clip_gru, word_gru, vid_gru ,cap_gru  = model.forward_emb(videoId, text_data, video_data)

    # measure elapsed time
    end = time.time()

    print('Test:' 'Time  ({batch_time:.3f})\t'.format(batch_time= (time.time() - end)))
    del video_data, text_data, videoId
    
    return vid_emb, cap_emb

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    # 模型加载
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    # 判断options是都拥有concate这个属性
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    # 文件名称  地址
    # video_path = ''
    # text = '东方红一号里的中国故事'
    # text = '自制美食:烤牛肉丸串'
    # text = '山东费县果业微课系列(一):果园生草的好处'
    # text = '电影《风平浪静》“案”潮汹涌 演技派现场碰撞来真的'
    text = '我想去种苹果'
    
    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # ?*768
    text_data = get_text_feature(text)
    text_embeds = text_data[0]
    text_stence_embeds = torch.tensor([text_data[1]])
    #  ?*1024
    videoId = ['argi_grass.mp4']
    video_embed = get_video_feature(videoId, '/home/fengkai/dataset/video_c')
    text_tensor = torch.tensor([text_embeds])
    video_tensor = torch.tensor(video_embed[0])
    
    video_lengths = [min(64, len(frame)) for frame in video_tensor]
    video_tensor_dim = len(video_tensor[0][0])
    # # batch*i3d数目*1024
    vidoes = torch.zeros(len(video_tensor), max(video_lengths), video_tensor_dim)
    videos_mean = torch.zeros(len(video_tensor), video_tensor_dim)
    # # batc*i3d数目
    vidoes_mask = torch.zeros(len(video_tensor), max(video_lengths))
    for i, frames in enumerate(video_tensor):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end, :]
            videos_mean[i, :] = torch.mean(frames, 0)
            vidoes_mask[i, :end] = 1.0
    video_lengths = torch.tensor(np.array(video_lengths))
    video_data = (vidoes, videos_mean, vidoes_mask, video_lengths)

    # # 文本数据的处理
    text_lengths = [min(32, len(token)) for token in text_tensor]
    cap_tensor_dim = len(text_tensor[0][0])
    text = torch.zeros(len(text_tensor), max(text_lengths), cap_tensor_dim)
    texts_mask = torch.zeros(len(text_tensor), max(text_lengths))
    text_stence = torch.zeros(len(text_tensor), cap_tensor_dim)
    for i, batch in enumerate(text_tensor):
        end = text_lengths[i]
        text[i, :end, :] = batch[:end, :]
        texts_mask[i, :end] = 1.0
        text_stence[i,:] = text_stence_embeds[i]
    text_lengths = torch.tensor(np.array(text_lengths))
    text_data = (text, text_stence, texts_mask, text_lengths)

    # data compute
    video_embs, cap_embs = evaluation(model, videoId, text_data, video_data)
    with torch.no_grad():
        video_embs = torch.nn.functional.normalize(video_embs).cpu()
        cap_embs = torch.nn.functional.normalize(cap_embs).cpu()
    score = distance.cdist(video_embs, cap_embs, 'euclidean')
    # embedding 的可视化分析 
    
    # with SummaryWriter(log_dir='./result/visual', comment='embedding_show') as writer: 
    #     writer.add_embedding(
    #             video_embs,
    #             global_step=1,)
    
    # with SummaryWriter(log_dir='./result/visual', comment='embedding_show') as writer: 
    #     writer.add_embedding(
    #             cap_embs,
    #             global_step=1,
    #             tag='text')

    
    print(" *The distance of video & text:", score)


def get_video_feature(video_list,video_path):
    data = extract_video(video_list,video_path)
    return data

def get_text_feature(text):
    text_processer = text_embed('Chinese')
    ch_caps = [text]
    ch_pos = PartOfSpeech(ch_caps)
    features = text_processer.process_ch(ch_caps)
    word_embed = features[0][0]
    setence_embed = features[1][0]
    data = (word_embed, setence_embed, ch_pos)
    return data
    

if __name__ == '__main__':
    main()
