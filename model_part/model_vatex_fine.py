from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_part.loss import TripletLoss
from model_part.cycleloss import CycleConsistencyLoss


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features



class Video_multilevel_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.concate

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.visual_kernel_sizes
            ])

        # visual mapping
        self.visual_mapping = MFC([1024+2048+2048,1024], opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, videos):
        """Extract video feature vectors."""
        
        # Level 1. Global Encoding by Mean Pooling According
        # 处理平均 batch*x*2048--> batch*2048
        videos, videos_mean, videos_mask, videos_length = videos
        if torch.cuda.is_available():
            videos = videos.cuda()
            videos_mean = videos_mean.cuda()
            videos_mask = videos_mask.cuda()

        level1 = videos_mean

        # Level 2. Temporal-Aware Encoding by biGRU
        # RNN : batch*x*2048 --> batch*x*2048
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:], 0)
        gru_out = mean_gru
        # batch*2048
        level2 = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        # batch*1*x*2048  -->  batch*
        videos_mask = videos_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * videos_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        level3 = self.dropout(con_out)

        # concatenation
        features = torch.cat((level1, level2, level3), 1)
        # features = torch.cat((level1, level2), 1)
        
        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features, gru_init_out, level2

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)



class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        
        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.text_kernel_sizes
            ])
        
        # multi fc layers
        self.text_mapping = MFC([768+2048+768*2,1024], opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, text, *args):
        # Embed word ids to vectors
        texts, texts_mean, texts_mask, texts_length = text
        if torch.cuda.is_available():
            texts = texts.cuda()
            texts_mean = texts_mean.cuda()
            texts_mask = texts_mask.cuda()

        # Level 1. attention  batch*768
        level1 = texts_mean
        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(texts)
        # Reshape *final* output to (batch_size, hidden_size)
        gru_out = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            gru_out[i] = torch.mean(batch[:], 0)
        level2 = self.dropout(gru_out)
        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        texts_mask = texts_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * texts_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        level3 = self.dropout(con_out)

        # concatenation
        features = torch.cat((level1, level2, level3), 1)
        # features = torch.cat((level1, level2), 1)
        
        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        return features, gru_init_out, level2




class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()


    def forward_loss(self, cap_emb, vid_emb, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss = self.criterion(cap_emb, vid_emb)
        if torch.__version__ == '0.3.1':  # loss.item() for 0.4.0, loss.data[0] for 0.3.1
            self.logger.update('Le', loss.data[0], vid_emb.size(0)) 
        else:
            self.logger.update('Le', loss.item(), vid_emb.size(0)) 
        return loss

    def compute_cluster_loss(self, v_embedding, p_embedding):
        v_embedding_normal = F.normalize(v_embedding)
        p_embedding_normal = F.normalize(p_embedding)
        return (self.criterion(v_embedding_normal, v_embedding_normal)
                + self.criterion(p_embedding_normal, p_embedding_normal)) / 2

    def train_emb(self, videoId, cap_tensor, video_tensor, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb, clip_gru, word_gru, vid_gru ,cap_gru  = self.forward_emb(videoId, cap_tensor,video_tensor ,False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, vid_emb)
        
         # add cluster loss
        # loss += self.compute_cluster_loss(vid_gru ,cap_gru)

        # add cycle loss
        # clip_clip_loss, sent_sent_loss = self.loss_cyclecons(
        #         clip_gru, video_tensor[2],video_tensor[3],
        #          word_gru, cap_tensor[2], cap_tensor[3])
        # loss += 0.001*(clip_clip_loss + sent_sent_loss )


        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
        else:
            loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value



class Dual_Encoding(BaseModel):
    """
    dual encoding network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.vid_encoding = Video_multilevel_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)
        print(self.vid_encoding)
        print(self.text_encoding)
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            # 输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation,
                                            cost_style=opt.cost_style,
                                            direction=opt.direction)
            
        # self.loss_cyclecons = CycleConsistencyLoss(
        #         num_samples=1, use_cuda=True)

        params = list(self.text_encoding.parameters())
        params += list(self.vid_encoding.parameters())
        self.params = params

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_emb(self, videoId, cap_tensor,video_tensor, volatile=False, *args):
        """
        Compute the video and caption embeddings
        """
        vid_emb,clip_gru,vid_gru = self.vid_encoding(video_tensor)
        cap_emb,word_gru,cap_gru = self.text_encoding(cap_tensor)
        return vid_emb, cap_emb, clip_gru, word_gru, vid_gru ,cap_gru 




NAME_TO_MODELS = {'dual_encoding': Dual_Encoding}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
