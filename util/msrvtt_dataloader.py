import os
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import json
import random


class Dataset2BertRes(data.Dataset):
    """
    Load captions and video frame features by pre-trained resnet model and Bert.
    """

    def __init__(self, cap_file, visual_feat, video_frames, videoEmbed_num, n_caption=None):
        # 加载文件  文本最长定为32  视频最长64个I3D
        self.data = []
        self.length = 0
        self.frame = []
        self.visual_feat = visual_feat
        self.video_frame = video_frames
        # TODO多线程加载吧  太慢了
        cap_files = os.listdir(cap_file)
        for caption in tqdm(cap_files):
            with open(os.path.join(cap_file, caption), 'r') as cap_reader:
                line = cap_reader.readlines()
                da_js = json.loads(line[0])
                ch_captions = da_js.get('Text', '')
                ch_embeds = da_js.get('Embed', '')
                video_id = da_js.get('videoID', '')

            for index, ch_embed in enumerate(ch_embeds):
                ch_caps_embed = np.array(ch_embed)
                cap_len = min(len(ch_captions[index][6:-6]) - 2, 32)
                self.length += 1
                self.data.append({
                    'videoID': video_id,
                    'text_length': cap_len,
                    'chEmbed': ch_caps_embed
                })

                # length = 
                # ch_caps_embed = np.array([ch_embed]*length)

    def __getitem__(self, index):
        da_js = self.data[index]
        # video
        video_id = da_js.get('videoID','')
        frame_list = self.video_frame[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        video_tensor = torch.Tensor(frame_vecs)

        # text
        cap_tensor = torch.Tensor(da_js['chEmbed'])
        text_length = da_js['text_length']

        return video_id, cap_tensor, video_tensor, text_length

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
    videoId, cap_tensor, video_tensor, text_length = zip(*data)
    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # 最长为64个I3D  video数据加载的预处理  dataloader只能加载方阵
    video_lengths = [min(32, len(frame)) for frame in video_tensor]
    frame_vec_len = len(video_tensor[0][0])
    vidoes = torch.zeros(len(video_tensor), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(video_tensor), frame_vec_len)
    vidoes_mask = torch.zeros(len(video_tensor), max(video_lengths))
    for i, frames in enumerate(video_tensor):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end, :]
            videos_origin[i, :] = torch.mean(frames, 0)
            vidoes_mask[i, :end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    # 文本数据的处理


    text_origin = torch.zeros(len(cap_tensor), len(cap_tensor[0]))
    text_mask = torch.zeros(len(cap_tensor), max(text_length))
    for i, batch in enumerate(cap_tensor):
        end = text_length[i]
        text_mask[i, :end] = 1.0
        text_origin[i,:] = batch[:]


    text_data = (text_origin, text_length, text_mask)

    return videoId, text_data, video_data
