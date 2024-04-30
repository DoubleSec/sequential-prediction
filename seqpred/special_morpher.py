from morphers import Morpher
from morphers.nn import Unsqueezer
import morphers
import torch
import polars as pl
import numpy as np

def choose_options(p, temperature: float = 1.0):
    """Function used for generation
    Takes an array of logits, not probabilities, sorry."""
    # p is n x k
    p = torch.softmax(p / temperature, dim=-1)
    agg_p = p.cumsum(dim=1)
    rand = torch.rand(agg_p.shape[0], 1).to(agg_p)
    p_arrays = torch.cat([agg_p, rand], dim=1)
    # n x 1
    ranks = torch.argsort(torch.argsort(p_arrays, dim=-1), dim=-1)
    choices = ranks[:, -1]
    return choices

class MixtureLossNormalizer(morphers.Normalizer):
    """This is the first Generator Morpher, that provides a generate method."""

    N_KERNELS = 10

    def make_predictor_head(self, x):
        return torch.nn.Linear(in_features=x, out_features=self.N_KERNELS * 2)

    def make_criterion(self):
        def gaussian_mixture_loss(input, targets):
            # targets are n, input is n x (n_kernels * 2)

            distribution_params = input.view([-1, self.N_KERNELS, 2])
            mean = distribution_params[:, :, 0].view([-1])
            var = torch.nn.functional.softplus(distribution_params[:, :, 1].view([-1]))
            # TKTK is this right?
            exp_targets = targets.unsqueeze(-1).expand([-1, self.N_KERNELS]).reshape([-1])
            return torch.nn.functional.gaussian_nll_loss(
                mean, exp_targets, var, full=False, reduction="none",
            )
        return gaussian_mixture_loss

    def generate(self, x, **_):
        # Just eats args and kwargs besides x
        # x is output from the predictor head
        distribution_params = x.view([-1, self.N_KERNELS, 2])
        mean = distribution_params[:, :, 0]
        var = torch.nn.functional.softplus(distribution_params[:, :, 1])

        pos = torch.randint(0, self.N_KERNELS, [x.shape[0], 1]).to(x.device)
        selected_means = torch.gather(mean, 1, pos).squeeze(-1)
        selected_vars = torch.gather(var, 1, pos).squeeze(-1)

        draws = selected_means + torch.randn([x.shape[0]], device=x.device) * selected_vars
        return draws


class Integerizer(morphers.Integerizer):
    """The default morpher expects quite a weird shape,
    so fix it for a normal case."""
    def make_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction="none")

    def generate(self, x, temperature=1.0, **_):
        return choose_options(x, temperature=temperature)

class Quantiler(Morpher):

    N_QUANTILES = 128

    def __init__(self, quantiles):
        self.quantiles = quantiles

    @property
    def required_dtype(self):
        return torch.float32

    @property
    def missing_value(self):
        # Roughly the median
        # This may or may not be a unique value.
        return 0.5

    def __call__(self, x):
        q = pl.Series(self.quantiles[1:])
        # k means between the (k-1)th quantile and the kth quantile
        return (
            x.cut(q, labels=np.arange(self.N_QUANTILES).astype("str")).cast(pl.Float32)
            / self.N_QUANTILES
        )

    @classmethod
    def from_data(cls, x):
        q = np.linspace(0, 1, cls.N_QUANTILES)
        quantiles = np.nanquantile(x.to_numpy(), q).tolist()
        return cls(quantiles)

    def save_state_dict(self):
        return {"quantiles": self.quantiles}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __repr__(self):
        return f"Quantiler({self.quantiles})"

    def make_embedding(self, x):
        return torch.nn.Sequential(
            Unsqueezer(dim=-1),
            torch.nn.Linear(in_features=1, out_features=x),
        )

    def make_predictor_head(self, x):
        return torch.nn.Linear(in_features=x, out_features=self.N_QUANTILES)

    def make_criterion(self):
        # Each bucket means exactly the quantile value, so there's some 
        # quantization error.
        def quantile_bce(input, target):
            target = torch.round(target * self.N_QUANTILES).long()
            return torch.nn.functional.cross_entropy(input, target, reduction="none")

        return quantile_bce

    def generate(self, x, temperature = 1.0, **_):
        options =  choose_options(x, temperature=temperature)
        return options / self.N_QUANTILES


if __name__ == "__main__":

    rng = np.random.default_rng()
    some_data = rng.random([512])
    df = pl.DataFrame(
        {
            "a": some_data,
        }
    )
    testo = Quantiler.from_data(df["a"])
    df.select(a=testo(pl.col("a")))
    print(df.head())

    criterion = testo.make_criterion()

    inputs = torch.softmax(torch.rand([64, 32]), dim=1)
    targets = torch.tensor(df["a"].to_numpy()[:64])
    loss = criterion(inputs, targets)
    print(loss[:5])
