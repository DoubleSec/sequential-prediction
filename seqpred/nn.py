import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
        
# TKTK ACTIVATIONS AND NORMS, LOL
class MarginalHead(nn.Module):

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
        self.k = len(self.embedders) + 1
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.predictors = nn.ModuleDict(
            {
                col: morpher.make_predictor_head(hidden_size)
                for col, morpher in morphers.items()
            }
        )
        self.criteria = {
            col: morpher.make_criterion()
            for col, morpher in morphers.items()
        }

    def forward(self, input_embedding, features):
        # n x k x e
        embeddings = torch.stack(
            [input_embedding] + [
                embedder(features[col]) 
                for col, embedder in self.embedders.items()
            ],
            dim=1,
        )
        print(embeddings)

        # n x k x k x e
        mask = (
            torch.ones([self.k, self.k])
            .to(embeddings)
            .triu(diagonal=1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand([embeddings.shape[0], -1, -1, embeddings.shape[-1]])
        )

        # n x k-1 x e (since the first dimension isn't predicted)
        masked_sums = (
            embeddings
            .unsqueeze(dim=2)
            .expand([-1, -1, self.k, -1]) 
            * mask
        ).sum(dim=1)[:, 1:, :]
        masked_sums = self.activation(self.norm(masked_sums))

        predictions = {
            col: predictor(masked_sums[:, i, :])
            for i, (col, predictor) in enumerate(self.predictors.items())
        }

        return predictions

class MargeNet(pl.LightningModule):

    def __init__(self, morphers: dict, hidden_size: int, initial_feature: str):
        super().__init__()
        self.save_hyperparameters()

        # Get the first feature's embedder
        self.init_feature = initial_feature
        self.init_embedder = morphers[initial_feature].make_embedding(hidden_size)
        self.init_norm = nn.LayerNorm(hidden_size)
        self.init_activation = nn.GELU()

        self.generator_head = MarginalHead(
            morphers={
                col: morpher 
                for col, morpher in morphers.items() 
                if col != initial_feature
            },
            hidden_size=hidden_size
        )

    def forward(self, x):
        ie = self.init_embedder(x[self.init_feature])
        ie = self.init_activation(self.init_norm(ie))
        print(ie)
        predictions = self.generator_head(ie, x)
        return predictions