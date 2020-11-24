import os
import random
import csv


ROOT_PATH=os.path.join(os.environ['HOME'], 'dataset')
val_path = 'vatex/text_embed_info/val_mean_multi_np'
dir_list = os.listdir(os.path.join(ROOT_PATH, val_path))
if len(dir_list) != 3000:
    print('数据有点问题')
'''
构建数据对<true_label. false_label>
[
   [t,f,label],
   [t,f,label]。。。。。
]
'''
label_1000 = random.sample(dir_list, 1000)
label_1000 = list(map(lambda x: x[:-4], label_1000))
out = []
c = 0
for index, di in enumerate(dir_list):
    text = di[:-4]
    if index % 3 == 0:
        out.append([text, label_1000[int(index/3)], 0])
        c += 1
    else:
        out.append([text, text, 1])


# print(c)
path = dir_list = os.path.join(ROOT_PATH, 'vatex/video_text_classify.csv')
# with open(path, 'w') as w:
#     w = csv.writer(w)
#     w.writerow(['Video','Text','Flag'])
#     w.writerows(out)



    