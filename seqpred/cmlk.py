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


class RelativePositionBias(nn.Module):

    def __init__(
        self,
        n_buckets: int,
        max_distance: int,
        n_heads: int,
    ):
        super().__init__()

        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.biases = nn.Embedding(n_buckets, n_heads)

    def _make_buckets(self, relative_positions):
        """
        This only works with casually masked attention, it will be all f'd up otherwise
        """

        # Half of the buckets are reserved for the individual values closest to the query
        max_exact = self.n_buckets // 2
        # Clamp at 0 from below
        relative_positions = torch.maximum(
            relative_positions, torch.zeros_like(relative_positions)
        )
        is_smol = relative_positions < max_exact

        # Function to assign the remaining buckets, increasing in size as we get farther from the query
        val_if_large = (
            max_exact
            + (
                torch.log(relative_positions.float() / max_exact)
                / math.log(self.max_distance / max_exact)
                * (self.n_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.minimum(
            val_if_large, torch.full_like(val_if_large, self.n_buckets - 1)
        )

        return torch.where(is_smol, relative_positions, val_if_large)

    def forward(self, x):
        """Adds relative position bias to attention logits."""

        seq_len = x.shape[2]

        # s x 1
        qp = torch.arange(seq_len, device=x.device).unsqueeze(-1)
        # 1 x s
        kp = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # s x s
        relative_pos = qp - kp

        # convert positions to relative buckets, still s x s
        relative_buckets = self._make_buckets(relative_pos)
        # 1 x h x s x s
        biases = self.biases(relative_buckets).permute(2, 0, 1).unsqueeze(0)
        return x + biases


class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        input_dim: int,
        n_kv_heads: int,
        n_q_heads: int,
        position_bias: nn.Module,
    ):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.n_q_heads = n_q_heads
        self.input_dim = input_dim
        # This may be nn.Identity
        self.position_bias = position_bias

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

        # Apply relative position biases if needed
        attn_scores = self.position_bias(attn_scores)

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
        position_bias: nn.Module,
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
            position_bias=position_bias,
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
        use_relative_position_bias: bool = False,
        position_bias_args: dict = {},
    ):
        super().__init__()

        if use_relative_position_bias:
            self.position_biases = RelativePositionBias(
                **position_bias_args,
                n_heads=layer_args["n_q_heads"],
            )
        else:
            self.position_biases = nn.Identity()

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(**layer_args, position_bias=self.position_biases)
                for _ in range(n_layers)
            ]
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
