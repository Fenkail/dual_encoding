
import jieba
import jieba.analyse
import jieba.posseg
import json
from tqdm import tqdm
import os
 
jieba.add_word('拼多多') 

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
    #上面的for循环可以用python递推式构造生成器完成
    # outstr = ",".join([("%s/%s" %(x.word,x.flag)) for x in sentence_seged])
    return word_out, flag_out




def process_ch(self,texts):
# 加载分词，处理文本
    
    return None


if __name__ == "__main__":
    xx = PartOfSpeech(["市长江大桥" for _ in range(10)])
    print(xx)
    # path = '/home/fengkai/dataset/vatex/text_embed_info/train'
    # files = os.listdir(path)
    # for f_index, file in enumerate(files):

    #     with open(file, 'r') as r:
    #         rr = r.readlines()
    #         line = rr[0]
    #         # line = line.split( ' {"videoID": ')
    #         da_js = json.loads(line)


    #     for video_txt_info in tqdm(da_js):
    #         videoID = video_txt_info.get('videoID','')
    #         ch_caps = video_txt_info.get('chCap','')
    #         ch_embeds = []
    #         ch_embeds = text_processer.process_ch(ch_caps)
    #         ## 向量str化
    #         # for key, ch_embed in  enumerate(ch_embeds):
    #         video_txt_info['chEmbed'] = ch_embeds
    #         text_processer.save_embeds(video_txt_info)
