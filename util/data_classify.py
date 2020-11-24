import os
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import json
import random
from multiprocessing import set_start_method
import csv
try:
    set_start_method('spawn')
except RuntimeError:
    pass

'''
文本与视频数据的相关性分数：
构造方法类似于bert模型的数据构造：
正负样本比例需要通过实验在业务数据集上多次尝试：
    之前做意图识别时，真实数据不同比例的类别，在训练时最好也采用略高于真实比例的比例。。。。。
    冷启动就1:1吧，看看效果，反正结果看准召
正样本<video，text>    1   50% 
负样本<video, text>   0   50%v与T不匹配，
其实只改dataloader即可，加载正确的样本和错误的样本，一个batch一半对错
'''
class DatasetCorrelation(data.Dataset):
    """
    Load captions and video features & process features for classification.
    """

    def __init__(self, cap_file_path, visual_file_path, n_caption=None):
        # 加载文件  文本最长定为32  视频最长64个I3D
        self.data = []
        self.length = 0
        self.cap_file_path = cap_file_path
        self.visual_file_path = visual_file_path
        self.text_id = []
        cap_files = os.listdir(cap_file_path)
        count = 0
        for caption in tqdm(cap_files):
            video_id = caption[:-4]
            ranint = random.random()
            if ranint < 0.4:
                text_id = random.sample(cap_files, 1)[0][:-4]
                self.length += 1
                flag = 0.0
                if text_id == video_id:
                    flag = 1.0
                self.data.append((video_id, text_id, flag))
            else:
                self.length += 1
                flag = 1.0
                count += 1
                text_id = video_id
                self.data.append((video_id, text_id, flag))
        print("the number of train positive sample is ",count)

    def __getitem__(self, index):
        video_id, text_id, flag = self.data[index]
        # assert video_id == text_id
        text_data = np.load(os.path.join(self.cap_file_path, text_id+'.npz'), allow_pickle=True)
        text_embeds = text_data['word_embed']
        text_stence_embeds = text_data['setence_embed']
        video_embed = (np.load(os.path.join(self.visual_file_path, video_id+'.npy')))[0]
        text_tensor = torch.tensor(text_embeds)
        video_tensor = torch.tensor(video_embed)
        text_stence_embeds = torch.tensor(text_stence_embeds)
        return video_id, text_tensor, video_tensor, text_stence_embeds, flag

    def __len__(self):
        return self.length


def collate_data(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length

    # 最长为64个I3D  video数据加载的预处理  dataloader只能加载方阵
    video_id, text_tensor, video_tensor, text_stence_embeds, flag = zip(*data)
    video_lengths = [min(64, len(frame)) for frame in video_tensor]
    video_tensor_dim = len(video_tensor[0][0])
    # batch*i3d数目*1024
    vidoes = torch.zeros(len(video_tensor), max(video_lengths), video_tensor_dim)
    videos_mean = torch.zeros(len(video_tensor), video_tensor_dim)
    # batc*i3d数目
    vidoes_mask = torch.zeros(len(video_tensor), max(video_lengths))
    for i, frames in enumerate(video_tensor):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end, :]
            videos_mean[i, :] = torch.mean(frames, 0)
            vidoes_mask[i, :end] = 1.0
    video_lengths = torch.tensor(np.array(video_lengths))
    video_data = (vidoes, videos_mean, vidoes_mask, video_lengths)

    # 文本数据的处理
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

    flag = torch.tensor(flag, dtype=torch.long)

    return video_id, text_data, video_data, flag



class DatasetCorrelationVal(data.Dataset):
    """
    Load captions and video features & process features for classification.
    """

    def __init__(self, cap_file_path, visual_file_path, flag_csv_path):
        # 加载文件  文本最长定为32  视频最长64个I3D
        self.data = []
        self.length = 0
        self.cap_file_path = cap_file_path
        self.visual_file_path = visual_file_path
        self.text_id = []
        cap_files = os.listdir(cap_file_path)
        with open(flag_csv_path, 'r') as r:
            lines = csv.reader(r)
            for k, line in enumerate(lines):
                if k == 0:
                    continue
                self.length += 1
                video_id, text_id, flag = line[0], line[1], float(line[2])
                self.data.append((video_id, text_id, flag))

    def __getitem__(self, index):
        video_id, text_id, flag = self.data[index]
        text_data = np.load(os.path.join(self.cap_file_path, text_id+'.npz'), allow_pickle=True)
        text_embeds = text_data['word_embed']
        text_stence_embeds = text_data['setence_embed']
        video_embed = (np.load(os.path.join(self.visual_file_path, video_id+'.npy')))[0]
        text_tensor = torch.tensor(text_embeds)
        video_tensor = torch.tensor(video_embed)
        text_stence_embeds = torch.tensor(text_stence_embeds)
        return video_id, text_tensor, video_tensor, text_stence_embeds, flag

    def __len__(self):
        return self.length