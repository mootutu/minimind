import math
from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn

# 继承 nn.Module 类
class RMSNorm(nn.Module):
    # __init__ 初始化
    def __init__(self, dim:int, eps:float=1e-5):
        super().init()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward
    def forward(self, x):
        return self.weight * self._norm().type.type_as(x)

def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    预计算旋转位置编码（RoPE）的频率。

    参数:
        dim (int): 维度。
        end (int, 可选): 推入的序列长度，默认值为 32768。
        rope_base (float, 可选): 基础频率，默认值为 1e6。
        rope_scaling (dict, 可选): 缩放参数，用于 YaRN 缩放，默认值为 None。

    返回:
        freqs_cos (torch.Tensor): 余弦频率，形状为 (end, dim // 2)。
        freqs_sin (torch.Tensor): 正弦频率，形状为 (end, dim // 2)。
    """
    # 写出最初的 RoPE 式子
    """
    freqs_i = \frac{1}{rope_base^{\frac{2i}{dim}}}
    两两旋转： freqs 这里是 i，而指数那里是 2i，因为是两两一组进行旋转
    每个频率对应两个维度
    """
    freqs = 1.0 / rope_base ** torch.arange(0, dim, 2)[: dim // 2].float()/dim # 由于是两两一组进行旋转，所以以 2 为步长

    # YaRN
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        # 计算 corr_dim：从 0 开始，找第一个满足 波长 > 训练最大长度 的维度索引 i，也就是需要修正的维度，公式里面的 min 指的是最靠前的。为什么要找最靠前的？
        corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)

        # 计算 power
        power = torch.arange(0, dim//2, device=freqs.device).float() / max(dim//2 - 1, 1)

        # 计算 beta
        beta = beta_slow + (beta_fast - beta_slow) * power

        # 计算 scale
        scale = torch.where(
            torch.arange(0, dim//2, device=freqs.device).float() < corr_dim,
            (beta * factor - beta + 1)/beta * factor,
            1.0 / factor
        )

        # 应用 scale
        freqs = freqs * scale
    # 生成位置索引，与频率相乘
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float() # [end, dim//2] 的矩阵

    # 返回一个 cos 和 sin
    freqs_cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)
    freqs_sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    应用旋转位置编码（RoPE）到查询（q）和键（k）张量。

    参数:
        q (torch.Tensor): 查询张量，形状为 (..., seq_len, dim)。
        k (torch.Tensor): 键张量，形状为 (..., seq_len, dim)。
        cos (torch.Tensor): 余弦频率，形状为 (seq_len, dim // 2)。
        sin (torch.Tensor): 正弦频率，形状为 (seq_len, dim // 2)。
        unsqueeze_dim (int, 可选): 用于扩展维度的索引，默认值为 1。

    返回:
        q_embed (torch.Tensor): 编码后的查询张量，形状为 (..., seq_len, dim)。
        k_embed (torch.Tensor): 编码后的键张量，形状为 (..., seq_len, dim)。
    """

    # 实数域上的旋转：[a, b] => [-b, a]
    def rotate_half(x):
        """
        对输入张量 x 进行旋转，将最后一个维度的后半部分移动到前半部分，前半部分移动到后半部分。
        
        例如：
            x = [x1, x2, x3, x4]
            前半: [x1, x2]
            后半: [x3, x4]
            旋转后: [x3, x4, x1, x2]

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., dim)。

        返回:
            torch.Tensor: 旋转后的张量，形状为 (..., dim)。
        """ 
        # x.shape[-1] 取最后一个维度的中点
        # [-x[..., x.shape[-1] //2:] 取后半部分
        # [x[..., :x.shape[-1] //2] 取前半部分
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    
    # 应用旋转位置编码
    # x_rotated = x  * cos + rotate_half(x) * sin
    # unsqueeze 用于后续的维度扩展
    # 计算 q_embed, k_embed
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # bs: batch size, slen: sequence length, num_key_value_heads: number of key value heads, head_dim: head dimension
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:,:,:,None,:]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )