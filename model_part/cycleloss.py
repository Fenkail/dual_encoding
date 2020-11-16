import torch
from torch import nn


def compute_mean_distance_l2(c, s):
    return torch.mean((c - s) ** 2, dim=-1)


def compute_mean_distance_negative_l2(c, s):
    return -compute_mean_distance_l2(c, s)

class CycleConsistencyLoss(nn.Module):
    '''
    对抗生成网络的loss，目的是将A->B->A 保证视频embedding转换到文本后还能再次转换回来。
    来自于cycleGAN，其样本也不需要pix2pix是<图1，图2>这样的成对标签。直接给上两类图即可
    '''
    def __init__(self, num_samples=-1, use_cuda=True):
        super().__init__()
        self.num_samples = num_samples
        self.use_cuda = use_cuda
        self.num_samples_tensor = (torch.ones(1) * self.num_samples)
        if self.use_cuda:
            self.num_samples_tensor = self.num_samples_tensor.cuda(
                non_blocking=True)
        self.loss_distance_fn = compute_mean_distance_l2
        self.proximity_fn = compute_mean_distance_negative_l2
        self.proximity_mask_val = -1e18
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, clip_emb, clip_mask, clip_lens,
                sent_emb, sent_mask, sent_lens):
        clip_max_len, _ = torch.max(clip_lens, dim=-1)
        sent_max_len, _ = torch.max(sent_lens, dim=-1)
        # video clip processing
        clip_sent_nn, clip_alpha, clip_alpha_raw = self.get_soft_nn(
            clip_emb, clip_mask, sent_emb, sent_mask)
        del clip_alpha, clip_alpha_raw
        clip_clip_nn, clip_beta, clip_beta_raw = self.get_soft_nn(
            clip_sent_nn, clip_mask, clip_emb, clip_mask)
        del clip_clip_nn, clip_beta_raw
        # 只有beta会对loss产生影响
        clip_clip_loss = self.get_loss(
            clip_mask, clip_lens, clip_max_len, clip_beta)
        del  clip_lens, clip_max_len, clip_beta

        # text processing
        sent_clip_nn, sent_alpha, sent_alpha_raw = self.get_soft_nn(
            sent_emb, sent_mask, clip_emb, clip_mask)
        del sent_alpha, sent_alpha_raw
        sent_sent_nn, sent_beta, sent_beta_raw = self.get_soft_nn(
            sent_clip_nn, sent_mask, sent_emb, sent_mask)
        del sent_sent_nn, sent_beta_raw
        sent_sent_loss = self.get_loss(
            sent_mask, sent_lens, sent_max_len, sent_beta)
        del sent_mask, sent_lens, sent_max_len, sent_beta
        return clip_clip_loss, sent_sent_loss

    def get_mxn_repr(self, source_emb, source_mask, target_emb, target_mask):
        # 向量维度扩展
        source_rep = source_emb.unsqueeze(2)
        target_rep = target_emb.unsqueeze(1)
        # 相同才为一
        total_mask = source_mask.unsqueeze(2).bool() & target_mask.unsqueeze(
            1).bool()
        total_mask = total_mask.cuda()
        return source_rep, target_rep, total_mask

    def get_soft_nn(self, source_emb, source_mask, target_emb, target_mask):
        '''
        在已知目标向量的情况下做softmax，目标是target
        '''
        source_rep, target_rep, total_mask = self.get_mxn_repr(
            source_emb, source_mask, target_emb, target_mask)
        # 计算向量的L2距离（如果归一化也就是余弦距离）
        distance = self.proximity_fn(source_rep, target_rep)
        distance.masked_fill_(total_mask == 0, self.proximity_mask_val)
        weights_alpha = self.softmax(distance)
        soft_nn = target_emb.unsqueeze(dim=1) * weights_alpha.unsqueeze(dim=3)
        soft_nn = torch.sum(soft_nn, dim=2)
        return soft_nn, weights_alpha, distance

    def get_loss(self, emb_mask, emb_lens, emb_max_len, beta):
        idx_orig = torch.arange(emb_max_len)
        batch_size, _ = emb_mask.shape
        if self.use_cuda:
            idx_orig = idx_orig.cuda(non_blocking=True)
        idx_orig.unsqueeze_(0)
        # 索引上的一些权重求和
        index_nn = torch.sum(idx_orig.unsqueeze(1) * beta, dim=-1)
        idx_nn_rep, idx_orig_rep, emb_mask_rep = self.get_mxn_repr(
            index_nn, emb_mask, idx_orig, emb_mask)
        distance = self.loss_distance_fn(
            idx_nn_rep.unsqueeze(-1), idx_orig_rep.unsqueeze(-1))
        distance.masked_fill_(emb_mask_rep == 0, 0)
        l_seq = distance.diagonal(dim1=-2, dim2=-1)
        emb_lens = emb_lens.type_as(self.num_samples_tensor).cuda()
        if self.num_samples != -1:
            n_samp = torch.min(emb_lens, self.num_samples_tensor)
            total_loss = 0
            for batch, (c_loss, c_mask, c_nsamp) in enumerate(zip(
                    l_seq, emb_mask, n_samp)):
                idx = torch.multinomial(c_mask, int(c_nsamp))
                total_loss += c_loss[idx].mean()
            total_loss /= batch_size
        else:
            total_loss = (l_seq.sum(dim=-1) / emb_lens).mean(dim=-1)
        return total_loss
