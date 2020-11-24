import bert_pytorch
import numpy as np
import sys,os
import json
from transformers import AutoTokenizer, AutoModel
import torch
import ijson
import torch.nn as nn
from tqdm import tqdm
import random
from multiprocessing import set_start_method


try:
    set_start_method('spawn')
except RuntimeError:
    pass


part_list = ['v','vd','vn','n','nz','nt']
part_list_back = ['a','ad','an','d','i','l','s']


class text_embed:
    '''
    1. 对中文文本和视频进行预处理的--一一对应
    2. 中文信息分词，提取bert向量（降维？）
    3. 输入训练模型
    '''
    def __init__(self, language = None):
        if language == 'Chinese':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.bert = AutoModel.from_pretrained('bert-base-chinese').cuda()
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
            embed_out = [features[0].cpu().numpy(), features[1].cpu().numpy()]
        return embed_out

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
            embed = features[1]
        
        return embed.cpu().numpy()

    def save_embeds(self, path_dir, da_js):
        with open(path_dir+'.txt','w') as w:
            data = json.dumps(da_js , ensure_ascii=False)
            w.writelines(data)


    def save_embeds_ndarray(self, path_dir, features):
        word_embed, setence_embed, ch_pos = features
        np.savez(path_dir+'.npz', 
                    word_embed=word_embed, setence_embed=setence_embed, 
                    text_pos = ch_pos)

def vatex_process():
    with open('/home/fengkai/dataset/vatex/vatex_training_v1.0.json', 'r') as r:
        rr = r.readlines()
        line = rr[0]
        # line = line.split( ' {"videoID": ')
        da_js = json.loads(line)
    text_processer = text_embed('Chinese')
    path_dir = '/home/fengkai/dataset/vatex/text_embed_info/train_mean_multi_np'
    for video_txt_info in tqdm(da_js):
        info = {}
        videoID = video_txt_info.get('videoID','')
        ch_caps = video_txt_info.get('chCap','')
        # ch_embeds = []
        # for key ,ch_cap in enumerate(ch_caps):
        ch_caps = [ch_caps[4]]
        ch_pos = PartOfSpeech(ch_caps)
        features = text_processer.process_ch(ch_caps)
        word_embed = features[0][0]
        setence_embed = features[1][0]
        path_dir_id = os.path.join(path_dir, videoID)
        data = (word_embed, setence_embed, ch_pos)
        # for index, chembed in enumerate(ch_embeds):
            # path_dir_out = path_dir_id+'_'+str(index)
            # text_processer.save_embeds_ndarray(path_dir_out, chembed)
        # ch_mean_embeds = np.mean(ch_embeds, axis=0)
        text_processer.save_embeds_ndarray(path_dir_id, data)


def msrvtt_process():
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


def pos_embed(embeddings_index, words, word_parts):
    """
    docstring
    """
    word_tem,text_part = [],[]
    for word, part in zip(words, word_parts):
        if part in part_list:
            word_tem.append(word)
            text_part.append(part)
    text_part_outs = []
    embed_part_outs = []
    for word, part in zip(word_tem,text_part):
        if word in embeddings_index:
            word_vector = embeddings_index[word]
            text_part_outs.append(word+' '+part)
            embed_part_outs.append(word_vector)
    if len(text_part_outs) < 2:
        for word, part in zip(words, word_parts):
            if part in part_list_back:
                word_tem.append(word)
                text_part.append(part)
        for word, part in zip(word_tem,text_part):
            if word in embeddings_index:
                word_vector = embeddings_index[word]
                text_part_outs.append(word+' '+part)
                embed_part_outs.append(word_vector)
    if len(text_part_outs) < 2:
        for word, part in zip(words, word_parts):
            word_tem.append(word)
            text_part.append(part)
        for word, part in zip(word_tem,text_part):
            if word in embeddings_index:
                word_vector = embeddings_index[word]
                text_part_outs.append(word+' '+part)
                embed_part_outs.append(word_vector)
    text_part_outs = np.array(text_part_outs[:2])
    embed_part_outs = np.array(embed_part_outs[:2])
    
    return text_part_outs, embed_part_outs


def word2vector() -> dict:
    """
    加载词向量的模型
    return 
    """
    path = '/home/fengkai/model/word2vector_chinese'
    r = open(path, 'r', encoding='utf-8')
    line = r.readline()
    # word_num, word_dim = map(int, line.split())
    embeddings_index = {}
    lines = r.readlines()
    for data in tqdm(lines):
        data_list = data.strip().split(' ')
        word = data_list[0].strip()
        embeddings_index[word] = np.asarray(data_list[1:], dtype='float32')
    return embeddings_index


def word_pipeline():
    wrong = 0
    with open('/home/fengkai/dataset/vatex/vatex_training_v1.0.json', 'r') as r:
        rr = r.readlines()
        line = rr[0]
        # line = line.split( ' {"videoID": ')
        da_js = json.loads(line)
    path_dir = '/home/fengkai/dataset/vatex/text_embed_info/train_wordvector_pos'
    embeddings_index = word2vector()
    for video_txt_info in tqdm(da_js):
        videoID = video_txt_info.get('videoID','')
        ch_caps = video_txt_info.get('chCap','')
        ch_caps = [ch_caps[4]]
        words, flags = PartOfSpeech(ch_caps)
        path_dir_id = os.path.join(path_dir, videoID)
        text_part_outs, embed_part_outs = pos_embed(embeddings_index, words, flags)
        if len(text_part_outs) != 2:
            wrong += 1
            print(videoID)
            print(ch_caps)
            print(text_part_outs)
            print(embed_part_outs)
        np.savez(path_dir_id+'.npz', 
                    text_part_outs=text_part_outs,
                     embed_part_outs=embed_part_outs)
    print('不满两个词向量的共这么多：')
    print(wrong)

def PartOfSpeech(sentence):
    '''
    1. 文本读取，每一个视频ID对应10条句子，数据整理
    2. 文本词性提取，整理为{non： verb：，adj：}
    3. 向量提取，chembed_fine:[{non： verb：，adj：},{}。。。{}]
    :return:
    '''
    word_out,flag_out = [],[]
    for sent in sentence:
        sentence_seged = jieba.posseg.cut(sent.strip())
        outstr = ''
        for x in sentence_seged:
            outstr+="{}/{},".format(x.word,x.flag)
            word_out.append(x.word)
            flag_out.append(x.flag)
   
    return word_out, flag_out
if __name__ == "__main__":
    word_pipeline()

                    
            

            #     for en_cap in en_caps:
            #         en_embed = process_en(en_cap)
            #         en_embeds.append(en_embed)
            #     video_txt_info['enEmebd'] = en_embeds

            #     ## 向量str化
            #     # for key, ch_embed in  enumerate(ch_embeds):


            #         video_txt_info['chEmbed'] = ch_embeds
            #         text_processer.save_embeds(video_txt_info)