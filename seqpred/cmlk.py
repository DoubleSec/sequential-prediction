import torch
from torch import nn
from torch.nn import functional as F
import math


class RMSNorm(nn.Module):

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.size = size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.size))

    def forward(self, x):

        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.silu(self.linear(x)) * self.gate(x)


class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        input_dim,
        n_kv_heads,
        n_q_heads,
    ):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.n_q_heads = n_q_heads
        self.input_dim = input_dim

        self.negative_inf = float("-inf")

        # Everything needs to divide everything, basically.
        assert n_q_heads % n_kv_heads == 0
        assert input_dim % n_kv_heads == 0
        assert input_dim % n_q_heads == 0

        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = self.input_dim // self.n_q_heads

        self.wq = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.wk = nn.Linear(self.input_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.input_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.input_dim, self.input_dim, bias=False)

    def forward(self, x, mask=None):
        """Mask is additive ONLY."""

        # x is (n x s x e)
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose changes (n x s x h x e) to (n x h x s x e)
        xq = xq.transpose(1, 2)
        # Repeats k and v to match number of q heads
        exp_k = torch.repeat_interleave(xk, self.n_rep, dim=2).transpose(1, 2)
        exp_v = torch.repeat_interleave(xv, self.n_rep, dim=2).transpose(1, 2)

        # This is Scaled Dot-Product Attention
        attn_scores = torch.matmul(xq, exp_k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores + mask

        # n x h x s x s
        attn_scores = F.softmax(attn_scores, dim=-1)
        # n x h x s x e
        output = torch.matmul(attn_scores, exp_v)
        # n x s x (h x e)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # n x s x input_size
        return self.wo(output)


class TransformerLayer(nn.Module):

    def __init__(
        self,
        d_model,
        n_kv_heads,
        n_q_heads,
        ff_dim,
    ):
        super().__init__()

        self.input_dim = d_model
        self.n_kv_heads = n_kv_heads
        self.n_q_heads = n_q_heads
        self.ff_dim = ff_dim

        self.gq_attn = GroupedQueryAttention(
            input_dim=self.input_dim,
            n_kv_heads=self.n_kv_heads,
            n_q_heads=self.n_q_heads,
        )
        self.attn_norm = RMSNorm(d_model)
        self.linear_norm = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, ff_dim)
        self.linear = nn.Linear(ff_dim, d_model)

    def forward(self, x, mask=None):

        # Self attention
        a = x + self.gq_attn(self.attn_norm(x), mask=mask)
        # Linear layers
        h = self.swiglu(self.linear_norm(a))
        h = self.linear(h)
        return a + h


class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        layer_args: dict,
    ):
        super().__init__()

        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(**layer_args) for _ in range(n_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.transformer_layers:
            x = layer(x, mask)

        return x


if __name__ == "__main__":

    # attention = GroupedQueryAttention(
    #     n_kv_heads=4,
    #     n_q_heads=8,
    #     input_dim=32,
    # )

    transformer = Transformer(
        n_layers=4,
        layer_args=dict(d_model=32, n_kv_heads=4, n_q_heads=8, ff_dim=256),
    )

    x = torch.randn([128, 64, 32])
    mask = torch.ones([64, 64]).unsqueeze(0).expand([128, -1, -1])
    mask = torch.triu(mask).bool()

    y = transformer(x, mask)

    print(y.shape)
    print(sum(p.numel() for p in transformer.parameters()))
