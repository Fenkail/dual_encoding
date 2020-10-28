import torch.nn as nn
import torch
import torch.nn.functional as F


@torch.jit.script
def _split_last(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    b, s, d = x.size()
    head_size = int(d / n_heads)
    return x.view(b, s, n_heads, head_size)

# transfer function to Torch script
@torch.jit.script
def _merge_last(x: torch.Tensor) -> torch.Tensor:
    b, s, _, __ = x.size()
    return x.view(b, s, -1)

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, prop):
        """
        Args:
            prop:
                dim:
                p_drop_attn:
                n_heads:
        """
        super().__init__()
        self.proj_q = nn.Linear(prop['dim'], prop['dim'])
        self.proj_k = nn.Linear(prop['dim'], prop['dim'])
        self.proj_v = nn.Linear(prop['dim'], prop['dim'])
        self.drop = nn.Dropout(prop['p_drop_attn'])
        self.scores = None  # for visualization
        self.n_heads = prop['n_heads']

    def forward(self, x, mask, kv=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q = self.proj_q(x)
        k = self.proj_k(x) if kv is None else self.proj_k(kv)
        v = self.proj_v(x) if kv is None else self.proj_v(kv)
        q = _split_last(q, self.n_heads).transpose(1, 2)
        k = _split_last(k, self.n_heads).transpose(1, 2)
        v = _split_last(v, self.n_heads).transpose(1, 2)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / (k.size(-1) ** .5)
        if mask is not None:
            mask = mask[:, None, None, :]
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = _merge_last(h)
        self.scores = scores
        return h


class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, prop):
        """
        Args:
            prop:
                is_decoder:
                dim:
                p_drop_hidden:
                p_drop_attn:
                n_heads:
                dim_ff:
        """
        super().__init__()
        self.is_decoder: bool = prop.get('is_decoder', False)

        self.attn = MultiHeadedSelfAttention(prop)
        self.proj = nn.Linear(prop['dim'], prop['dim'])
        self.norm1 = LayerNorm(prop['dim'], 1e-12)
        self.pwff = PositionWiseFeedForward(prop)
        self.norm2 = LayerNorm(prop['dim'], 1e-12)
        self.drop = nn.Dropout(prop['p_drop_hidden'])
        if self.is_decoder:
            self.norm15 = LayerNorm(prop['dim'], 1e-12)
            self.encoder_attn = MultiHeadedSelfAttention(prop)
            self.encoder_proj = nn.Linear(prop['dim'], prop['dim'])

    def forward(self, x, mask, src_encoding=None, src_mask=None):
        x = x.type_as(self.proj.weight)
        if torch.is_tensor(mask) :
            mask = mask.type_as(self.proj.weight)
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))

        if self.is_decoder and src_encoding is not None:
            h2 = self.encoder_attn(h, src_mask, kv=src_encoding)
            h = self.norm15(h + self.drop(self.encoder_proj(h2)))

        h = self.norm2(h + self.drop(self.pwff(h)))
        return h



class LayerNorm(torch.nn.Module):
    # pylint: disable=line-too-long
    """
    An implementation of `Layer Normalization
    <https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Parameters
    ----------
    dimension : ``int``, required.
        The dimension of the layer output to normalize.
    eps : ``float``, optional, (default = 1e-6)
        An epsilon to prevent dividing by zero in the case
        the layer has zero variance.

    Returns
    -------
    The normalized layer output.
    """

    def __init__(self,
                 dimension: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, tensor: torch.Tensor):  # pylint: disable=arguments-differ
        new_tensor = tensor.type_as(self.gamma)
        mean = new_tensor.mean(-1, keepdim=True)
        s = (new_tensor - mean).pow(2).mean(-1, keepdim=True)
        x = (new_tensor - mean) / torch.sqrt(s + self.eps)
        res = self.gamma * x + self.beta
        res = res.type_as(tensor)
        return res



class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, prop):
        """
        Args:
            prop:
                dim:
                dim_ff:
        """
        super().__init__()
        self.fc1 = nn.Linear(prop['dim'], prop['dim_ff'])
        self.fc2 = nn.Linear(prop['dim_ff'], prop['dim'])
        self.act = nn.GELU()
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(self.act(self.fc1(x)))  # Gelu


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, option):
        """
        Args:
            option:
                is_decoder:
                dim:
                p_drop_hidden:
                p_drop_attn:
                n_heads:
                dim_ff:
                n_layers:
        """
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([Block(option) for _ in range(option['n_layers'])])

    def forward(self, h, mask, src_encoding=None, src_mask=None):
        all_layer_outputs = [h]
        for block in self.blocks:
            h = block(h, mask, src_encoding, src_mask)
            all_layer_outputs.append(h)
        return all_layer_outputs