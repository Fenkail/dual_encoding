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

    def __init__(self, cap_file, visual_feat, videoEmbed_num , n_caption=None):
        # 加载文件
        self.data = []
        self.length = 0
        self.frame = []
        cap_files = os.listdir(cap_file)
        for caption in tqdm(cap_files):
            with open(os.path.join(cap_file,caption), 'r') as cap_reader:
                line = cap_reader.readlines()
                da_js = json.loads(line[0])
                ch_captions = da_js.get('chCap','')
                ch_embeds = da_js.get('chEmbed','')
                video_id = da_js.get('videoID','')
                ch_embeds = np.array(ch_embeds)
            video_embed = np.load(os.path.join(visual_feat,video_id+'.npy'))[0]
            video_embed = self.video_embed_process(video_embed, videoEmbed_num)
            for ch_embed in ch_embeds:
                self.length += 1
                self.data.append({
                    'videoID':video_id,
                    'videoEmbed':video_embed,
                    'videoChCaption':ch_captions,
                    'chEmbed':ch_embed
                })

    def __getitem__(self, index):
        da_js = self.data[index]
        # video
        video_tensor = torch.Tensor(da_js['videoEmbed'])
        videoId = da_js['videoID']

        # text
        cap_tensor = torch.Tensor(da_js['chEmbed'])
        ch_cap = da_js['videoChCaption']

        return videoId, cap_tensor, video_tensor, ch_cap
    
    def __len__(self):
        return self.length

    def video_embed_process(self, embed, videoEmbed_num):
        num = len(embed)
        if num == videoEmbed_num:
            return embed
        else:
            times = int(videoEmbed_num/num)
            embed_out = embed
            for i in range(times):
                embed_out = np.concatenate((embed_out,embed),axis=0)
            embed_out = random.sample(list(embed_out), videoEmbed_num)
            return np.array(embed_out)
