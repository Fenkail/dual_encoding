import os
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import json
import random


class Dataset2BertI3d(data.Dataset):
    """
    Load captions and video frame features by pre-trained I3D model and Bert.
    """

    def __init__(self, cap_file_path, visual_file_path, videoEmbed_num, n_caption=None):
        # 加载文件  文本最长定为32  视频最长64个I3D
        self.data = []
        self.length = 0
        self.cap_file_path = cap_file_path
        self.visual_file_path = visual_file_path
        self.text_id = []
        cap_files = os.listdir(cap_file_path)
        for caption in tqdm(cap_files):
            video_id = caption[:-4]
            text_id = caption[:-4]
            self.length += 1
            self.data.append((video_id, text_id))
  

    def __getitem__(self, index):
        video_id, text_id = self.data[index]
        text_embeds = np.load(os.path.join(self.cap_file_path, text_id+'.npy'), allow_pickle=True)
        video_embed = (np.load(os.path.join(self.visual_file_path, video_id+'.npy')))[0]
        text_tensor = torch.tensor(text_embeds)
        video_tensor = torch.tensor(video_embed)
        return video_id, text_id, text_tensor, video_tensor

    def __len__(self):
        return self.length

    def video_embed_process(self, embed, videoEmbed_num):
        # 处理视频为相同size的特征
        num = len(embed)
        if num == videoEmbed_num:
            return embed
        else:
            times = int(videoEmbed_num/num)
            embed_out = embed
            for i in range(times):
                embed_out = np.concatenate((embed_out, embed), axis=0)
            embed_out = random.sample(list(embed_out), videoEmbed_num)
            return np.array(embed_out)


def collate_data(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    video_id, text_v_id, text_tensor, video_tensor = zip(*data)
    assert video_id == text_v_id
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

    video_data = (vidoes, videos_mean, vidoes_mask)

    # 文本数据的处理

    # text_lengths = [min(32, len(token)) for token in text_tensor]
    text_lengths,texts_mask = [],[]
    cap_tensor_dim = len(text_tensor[0])
    texts_mean = torch.zeros(len(text_tensor), cap_tensor_dim)
    for i, batch in enumerate(text_tensor):
        texts_mean[i,:] = batch[:]
    text = texts_mean
    text_data = (text, texts_mean, texts_mask)

    return video_id, text_data, video_data