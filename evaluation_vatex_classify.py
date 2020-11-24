from __future__ import print_function
import os
import pickle
import time
import numpy as np
from scipy.spatial import distance
import torch
from torch.autograd import Variable
from basic.metric import getScorer
from basic.util import AverageMeter, LogCollector
from basic.generic_utils import Progbar
from sklearn import metrics

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

def cal_error(videos, captions, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*np.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    return errors


def encode_data(model, data_loader, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    labels = np.zeros((1,1))
    logits = np.zeros((1,2))

    for i, (videoId, text_data, video_data, flag) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            logit, label, vid_emb, cap_emb= model.forward_emb(videoId, text_data, video_data, flag)

        # initialize the numpy arrays given the size of the embeddings

        labels = np.append(labels, label.data.cpu().numpy().copy())
        logits = np.vstack((logits, (logit.data.cpu().numpy().copy())))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del video_data, text_data, videoId
    labels = labels[1:]
    logits = logits[1:]
    logits = logits.max(-1)
    predict = np.array(list((map(lambda x:np.where(x<0.5, 0, 1),logits))))
    # predict = np.array(list((map(lambda x:np.where(x[0]<x[1], 0, 1),logits))))
    acc = metrics.accuracy_score(labels, predict)
    pre = metrics.precision_score(labels, predict)
    recall = metrics.recall_score(labels, predict)
    f1 = metrics.f1_score(labels, predict)
    scores = (acc, pre, recall, f1)
    return scores
