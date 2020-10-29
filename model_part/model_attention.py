from collections import OrderedDict
from model_part.attention import Transformer
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

        # Transformer
        self.transformer = Transformer({
            'is_decoder':False,
            'dim':1024,
            'p_drop_hidden':0.2,
            'p_drop_attn':0.2,
            'n_heads':8,
            'dim_ff':1024,
            'n_layers':4,
        })
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, 1024), padding=(window_size - 1, 0)) 
            for window_size in opt.visual_kernel_sizes
            ])
        #  mapping
        self.mapping = MFC([1024*4,1024], opt.dropout, have_bn=True, have_last_bn=True)
        # pooling



    def forward(self, videos):
        """Extract video feature vectors."""
        videos, videos_mean, lengths, videos_mask = videos
        if torch.cuda.is_available():
            videos = videos.cuda()
            videos_mean = videos_mean.cuda()
            videos_mask = videos_mask.cuda()
        # level 1. i3D平均特征 batch*1024
        level1 = videos_mean
        # Level 2. attention
        transformer_out_list = self.transformer(videos,videos_mask)
        transformer_out = transformer_out_list[-1]
        # batch*32*1024 --> batch*1*1024--> batch*1024
        pool = nn.MaxPool1d(32)
        transformer_out_p = transformer_out.permute(0,2,1)
        level2 = pool(transformer_out_p)
        level2 = level2.squeeze(2)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        # batch*1*x*1024  -->  batch*1024
        videos_mask = videos_mask.unsqueeze(2).expand(-1,-1,transformer_out.size(2)) # (N,C,F1)
        level3 = transformer_out * videos_mask
        con_out = level3.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        level3 = self.dropout(con_out)

        # mapping
        feature = torch.cat((level1, level2, level3), 1)
        feature = self.mapping(feature)

        if self.visual_norm:
            features = l2norm(feature)

        return features

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
        
        # self-attention
        self.transformer = Transformer({
            'is_decoder':False,
            'dim':768,
            'p_drop_hidden':0.2,
            'p_drop_attn':0.2,
            'n_heads':8,
            'dim_ff':768,
            'n_layers':4,
        })
        #  mapping
        self.mapping = MFC([768*4,1024], opt.dropout, have_bn=True, have_last_bn=True)

        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, 768), padding=(window_size - 1, 0)) 
            for window_size in opt.text_kernel_sizes
            ])


    def forward(self, text, *args):
        # Embed word ids to vectors
        text_origin, text_length, text_mask =  text
        if torch.cuda.is_available():
            text_origin = text_origin.cuda()
            text_mask = text_mask.cuda()

        # Level 1. attention
        text_origin_t = text_origin.unsqueeze(1)
        level1 = text_origin
        # Level 2. transformer
        transformer_out_list = self.transformer(text_origin_t, None)
        transformer_out = transformer_out_list[-1]
        level2 = transformer_out.squeeze(1)
        # Level 3. conv
        con_out = transformer_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        level3 = self.dropout(con_out)

        # mapping
        feature = torch.cat((level1, level2, level3), 1)
        features = self.mapping(feature)

        # concatenation
        # if self.concate == 'full': # level 1+2+3
        
        if self.text_norm:
            features = l2norm(features)

        return features




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

    def train_emb(self, videoId, cap_tensor, video_tensor, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb = self.forward_emb(videoId, cap_tensor,video_tensor ,False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, vid_emb)
        
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

        params = list(self.text_encoding.parameters())
        params += list(self.vid_encoding.parameters())
        self.params = params

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_emb(self, videoId, cap_tensor,video_tensor, volatile=False, *args):
        """Compute the video and caption embeddings
        """

        vid_emb = self.vid_encoding(video_tensor)
        cap_emb = self.text_encoding(cap_tensor)
        return vid_emb, cap_emb



NAME_TO_MODELS = {'dual_encoding': Dual_Encoding}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
