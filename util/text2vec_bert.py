import bert_pytorch
import numpy as np
import sys,os
import json
from pytorch_transformers import  BertModel, BertConfig,BertTokenizer
from transformers import AutoTokenizer, AutoModel
import torch
import ijson
import torch.nn as nn
from tqdm import tqdm


class text_embed:
    '''
    1. 对中文文本和视频进行预处理的--一一对应
    2. 中文信息分词，提取bert向量（降维？）
    3. 输入训练模型
    '''
    def __init__(self, language = None):
        if language == 'Chinese':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert = BertModel.from_pretrained('bert-base-chinese').cuda()
        elif language == "English":
            # bert-base-uncased means english == English
            self.tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert = AutoModel.from_pretrained('bert-base-uncased').cuda()
        assert language != None
        self.bert.eval()
        # bert = BertModel.from_pretrained('bert-base-chinese')


    def process_ch(self,texts):
    # 加载分词，处理文本
        tokens, segments, input_masks = [], [], []
        for text in texts:
            tokenized_text = self.tokenizer.tokenize(text) #用tokenizer对句子分词
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)#索引列表
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))
        max_len = max([len(single) for single in tokens]) #最大的句子长度
        
        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        # 加载数据 to GPU
        if torch.cuda.is_available():
            tokens_tensor = torch.tensor(tokens).cuda()
            segments_tensors = torch.tensor(segments).cuda()
            input_masks_tensors = torch.tensor(input_masks).cuda()
        else:
            print('cuda 有点问题呀')
            tokens_tensor = torch.tensor(tokens)
            segments_tensors = torch.tensor(segments)
            input_masks_tensors = torch.tensor(input_masks)
            
        # 特征获取
        with torch.no_grad():
            features = self.bert(tokens_tensor, input_masks_tensors, segments_tensors)
            # 每一个token有一个维度的特征，batch*token_max_len*768，特征的第二个维度上为batch*768
            text_embed = features[1]
        
        return text_embed.cpu().numpy().tolist()


    def process_en(self, texts):
    # 加载分词，处理文本
        tokens, segments, input_masks = [], [], []
        for text in texts:
            tokenized_text = self.tokenizer.tokenize(text) #用tokenizer对句子分词
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)#索引列表
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))
        max_len = max([len(single) for single in tokens]) #最大的句子长度
        
        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        # 加载数据 to GPU
        if torch.cuda.is_available():
            tokens_tensor = torch.tensor(tokens).cuda()
            segments_tensors = torch.tensor(segments).cuda()
            input_masks_tensors = torch.tensor(input_masks).cuda()
        else:
            print('cuda 有点问题呀')
            tokens_tensor = torch.tensor(tokens)
            segments_tensors = torch.tensor(segments)
            input_masks_tensors = torch.tensor(input_masks)
            
        # 特征获取
        with torch.no_grad():
            features = self.bert(tokens_tensor, input_masks_tensors, segments_tensors)
            # 每一个token有一个维度的特征，batch*token_max_len*768，特征的第二个维度上为batch*768
            text_embed = features[1]
        
        return text_embed.cpu().numpy().tolist()

    def save_embeds(self, path_dir, da_js):
        path = os.path.join(path_dir, 'bert_text/')
        videoID = da_js.get('videoID','')
        with open(path+videoID+'.txt','w') as w:
            data = json.dumps(da_js , ensure_ascii=False)
            w.writelines(data)

def vatex_process():

    with open('/home/fengkai/dataset/vatex/vatex_training_v1.0.json', 'r') as r:
        rr = r.readlines()
        line = rr[0]
        # line = line.split( ' {"videoID": ')
        da_js = json.loads(line)

    text_processer = text_embed()
    text_language = 'Chinese'
    if text_language == 'English':
        for video_txt_info in da_js:
            videoID = video_txt_info.get('videoID','')
            en_caps = video_txt_info.get('enCap','')
            en_embeds = []
            for en_cap in en_caps:
                en_embed = process_en(en_cap)
                en_embeds.append(en_embed)
            video_txt_info['enEmebd'] = en_embeds
    elif text_language == 'Chinese':
        for video_txt_info in tqdm(da_js):
            videoID = video_txt_info.get('videoID','')
            ch_caps = video_txt_info.get('chCap','')
            ch_embeds = []
            for key ,ch_cap in enumerate(ch_caps):
                ch_caps[key] = "[CLS] "+ch_cap+" [SEP]"
            ch_embeds = text_processer.process_ch(ch_caps)
            ## 向量str化
            # for key, ch_embed in  enumerate(ch_embeds):
            video_txt_info['chEmbed'] = ch_embeds
            text_processer.save_embeds(video_txt_info)

if __name__ == "__main__":
    path = {x:'/home/fengkai/dataset/msrvtt/msrvtt10k'+x+'/TextData/'+'msrvtt10k'+x+'.caption.txt' for x in ['val', 'test']}
    path_dir =  {x:'/home/fengkai/dataset/msrvtt/msrvtt10k'+x+'/TextData/' for x in ['val', 'test']}
    for key, path_value in path.items():
        with open(path_value, 'r') as r:
            rr = r.readlines()
            info = {}
            data = []
            for line in rr:
                line = line.strip().split(' ', maxsplit=1)
                ID = line[0].split('#', maxsplit=1)
                setenceID = ID[1]
                videoID = ID[0]
                text = line[1]
                data.append([videoID, text])
            data.sort()
            for da in data:
                id = da[0]
                text = da[1]
                if id in info:
                    info[id]['Text'].append(text)
                else:
                    info[id] = {'videoID':id,
                                'Text':[text],
                                'Embed':[]}
            text_processer = text_embed('English')
            for k,v in tqdm(info.items()):
                if len(v['Text']) != 20:
                    print('Wrong')
                    assert False
                embed = text_processer.process_en(v["Text"])
                info[k]['Embed'] = embed
                text_processer.save_embeds(path_dir[key],v)
                    
            

            #     for en_cap in en_caps:
            #         en_embed = process_en(en_cap)
            #         en_embeds.append(en_embed)
            #     video_txt_info['enEmebd'] = en_embeds

            #     ## 向量str化
            #     # for key, ch_embed in  enumerate(ch_embeds):


            #         video_txt_info['chEmbed'] = ch_embeds
            #         text_processer.save_embeds(video_txt_info)