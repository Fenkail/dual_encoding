import os
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import json
import random
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

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
        assert video_id == text_id
        text_data = np.load(os.path.join(self.cap_file_path, text_id+'.npz'), allow_pickle=True)
        text_embeds = text_data['word_embed']
        text_stence_embeds = text_data['setence_embed']
        video_embed = (np.load(os.path.join(self.visual_file_path, video_id+'.npy')))[0]
        text_tensor = torch.tensor(text_embeds)
        video_tensor = torch.tensor(video_embed)
        text_stence_embeds = torch.tensor(text_stence_embeds)
        return video_id, text_tensor, video_tensor, text_stence_embeds

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

    # 最长为64个I3D  video数据加载的预处理  dataloader只能加载方阵
    video_id, text_tensor, video_tensor, text_stence_embeds = zip(*data)
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

    return video_id, text_data, video_data
