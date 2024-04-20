from morphers import Morpher
from morphers.nn import Unsqueezer
import morphers
import torch
import polars as pl
import numpy as np


class Integerizer(morphers.Integerizer):
    """The default morpher expects quite a weird shape,
    so fix it for a normal case."""
    def make_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction="none")

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
        q = pl.Series(self.quantiles[:-1])
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
        def ev_mse(input, target):
            ev = torch.sum(
                torch.softmax(input, dim=1) 
                * torch.linspace(0, 1, self.N_QUANTILES).to(input).view(1, -1),
                dim=-1
            )
            return torch.nn.functional.mse_loss(ev, target, reduction="none")

        return ev_mse


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
