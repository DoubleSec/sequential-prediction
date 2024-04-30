import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from .special_morpher import Quantiler


def _make_masked_sums(embeddings):
    """Make a sum mask for some set of embeddings"""
    mask = (
        torch.ones([embeddings.shape[1], embeddings.shape[1]])
        .to(embeddings)
        .triu(diagonal=1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand([embeddings.shape[0], -1, -1, embeddings.shape[-1]])
    )
    masked_sums = (
        embeddings
        .unsqueeze(dim=2)
        .expand([-1, -1, embeddings.shape[1], -1]) 
        * mask
    ).sum(dim=1)

    return masked_sums

class SumMarginalHead(nn.Module):

    def __init__(
        self,
        morphers: dict,
        hidden_size: int,
    ):
        """Should only expect morphers for the things to predict"""

        super().__init__()
        self.morphers = morphers
        self.embedders = nn.ModuleDict(
            {
                col: morpher.make_embedding(hidden_size)
                for col, morpher in morphers.items()
            }
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.predictors = nn.ModuleDict(
            {
                col: morpher.make_predictor_head(hidden_size)
                for col, morpher in morphers.items()
            }
        )

    def forward(self, input_embedding, features):
        # n x k x e
        embeddings = torch.stack(
            [input_embedding] + [
                embedder(features[col]) 
                for col, embedder in self.embedders.items()
            ],
            dim=1,
        )

        # n x k-1 x e (since the first dimension isn't predicted)
        masked_sums = _make_masked_sums(embeddings)[:, 1:, :]
        masked_sums = self.activation(self.norm(masked_sums))

        predictions = {
            col: predictor(masked_sums[:, i, :])
            for i, (col, predictor) in enumerate(self.predictors.items())
        }

        return predictions

    def generate(self, x, **kwargs):
        # x is context
        # embeddings is n x len(predictors) x e
        embeddings = torch.zeros(
            [x.shape[0], len(self.predictors) + 1, x.shape[-1]]
        ).to(x)
        preds = {}
        embeddings[:, 0, :] = x
        for i, (feat, predictor) in enumerate(self.predictors.items()):
            
            # Predict on the previous context
            total_context = torch.sum(embeddings[:, :i+1, :], dim=1)
            total_context = self.activation(self.norm(total_context))

            # Make a draw
            p_dist = predictor(total_context)
            generated_values = self.morphers[feat].generate(p_dist, **kwargs)

            preds[feat] = generated_values
            new_embedding = self.embedders[feat](generated_values)
            embeddings[:, i+1, :] = new_embedding

        return preds


class CatMarginalHead(nn.Module):

    def __init__(
        self,
        morphers: dict,
        hidden_size: int,
    ):
        """Should only expect morphers for the things to predict"""

        super().__init__()

        self.embedders = nn.ModuleDict(
            {
                col: morpher.make_embedding(hidden_size)
                for col, morpher in morphers.items()
            }
        )
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.activation = nn.GELU()
        self.predictors = nn.ModuleDict(
            {
                col: morpher.make_predictor_head(2 * hidden_size)
                for col, morpher in morphers.items()
            }
        )

    def forward(self, input_embedding, features):
        # now n x k x e
        repeated_context = input_embedding.unsqueeze(1).expand(-1, len(self.embedders), -1)

        # n x k x e
        embeddings = torch.stack(
            [
                embedder(features[col]) 
                for col, embedder in self.embedders.items()
            ],
            dim=1,
        )
        masked_sums = _make_masked_sums(embeddings)

        # Cat so they're side-by-side
        # n x k x 2e
        embeddings = torch.cat([repeated_context, masked_sums], dim=-1)
        masked_sums = self.activation(self.norm(embeddings))

        predictions = {
            col: predictor(masked_sums[:, i, :])
            for i, (col, predictor) in enumerate(self.predictors.items())
        }

        return predictions

    def generate(self, x, morphers):
        # x is context
        raise NotImplementedError("This doesn't work anyway so I didn't bother.")

class MargeNet(pl.LightningModule):

    def __init__(
        self,
        head_type: str,
        morphers: dict,
        hidden_size: int,
        initial_features: list,
        optim_lr: float,
        p_dropout: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optim_lr = optim_lr

        # Get the first feature's embedder
        self.init_features = initial_features
        self.init_embedders = nn.ModuleDict(
            {
                init_feat: morphers[init_feat].make_embedding(hidden_size)
                for init_feat in self.init_features
            }
        )
        
        self.dropout = nn.Dropout(p=p_dropout)

        if head_type == "sum":
            self.generator_head = SumMarginalHead(
                morphers={
                    col: morpher 
                    for col, morpher in morphers.items() 
                    if col not in self.init_features
                },
                hidden_size=hidden_size
            )
        elif head_type == "cat":
            self.generator_head = CatMarginalHead(
                morphers={
                    col: morpher 
                    for col, morpher in morphers.items() 
                    if col not in self.init_features
                },
                hidden_size=hidden_size
            )

        self.criteria = {
            col: morpher.make_criterion()
            for col, morpher in morphers.items()
            if col not in self.init_features
        }


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer


    def forward(self, x):
        context = sum(
            self.init_embedders[init_feat](x[init_feat])
            for init_feat in self.init_features
        )
        predictions = self.generator_head(context, x)
        return predictions


    def training_step(self, x):
        preds = self(x)
        loss_dict = {
            f"train_{col}_loss": criterion(preds[col], x[col]).mean()
            for col, criterion in self.criteria.items()
        }
        self.log_dict(loss_dict)
        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss)
        return total_loss


    def validation_step(self, x):
        preds = self(x)
        loss_dict = {
            f"validation_{col}_loss": criterion(preds[col], x[col]).mean()
            for col, criterion in self.criteria.items()
        }
        self.log_dict(loss_dict)
        total_loss = sum(loss_dict.values())
        self.log("validation_loss", total_loss)
        return total_loss

    def generate(self, x, **kwargs):
        """Generate pitches.

        x should contain values for all the context features."""

        context = sum(
            self.init_embedders[init_feat](x[init_feat])
            for init_feat in self.init_features
        )
        return self.generator_head.generate(context, **kwargs)