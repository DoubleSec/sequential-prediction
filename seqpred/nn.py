import math

import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl


class BoringPositionalEncoding(nn.Module):
    """
    Shamelessly "adapted" from a torch tutorial
    """

    def __init__(self, max_length: int, d_model: int):
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:, : x.shape[1], :]


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
        # n x s x k x e
        embeddings = torch.stack(
            [input_embedding]
            + [
                # We don't make a prediction for a context-free first pitch.
                embedder(features[col])[:, 1:, :]
                for col, embedder in self.embedders.items()
            ],
            dim=-2,
        )

        # n x s-1 x k-1 x e
        masked_sums = embeddings.cumsum(dim=-2)[:, :, :-1, :]
        masked_sums = self.activation(self.norm(masked_sums))

        predictions = {
            col: predictor(masked_sums[:, :, i, :])
            for i, (col, predictor) in enumerate(self.predictors.items())
        }

        return predictions

    def embed_inputs(self, features):
        # We actually need to use the input embeddings twice in the process,
        # once at the input to the transformer and once during generation.
        # n x s x k x e
        # (Probably just getting summed, but we'll see.)
        embeddings = torch.stack(
            [embedder(features[col]) for col, embedder in self.embedders.items()],
            dim=-2,
        )
        return embeddings

    def generate(self, x, **kwargs):
        # x is context, n x s x e
        # embeddings is n x len(predictors) x e
        embeddings = torch.zeros(
            [x.shape[0], x.shape[1], len(self.predictors) + 1, x.shape[-1]]
        ).to(x)
        preds = {}
        embeddings[:, :, 0, :] = x
        for i, (feat, predictor) in enumerate(self.predictors.items()):

            # Predict on the previous context
            total_context = torch.sum(embeddings[:, :, : i + 1, :], dim=-2)
            total_context = self.activation(self.norm(total_context))

            # Make a draw
            p_dist = predictor(total_context)
            generated_values = self.morphers[feat].generate(p_dist, **kwargs)

            preds[feat] = generated_values
            new_embedding = self.embedders[feat](generated_values)
            embeddings[:, :, i + 1, :] = new_embedding

        return preds


class SequentialMargeNet(pl.LightningModule):

    def __init__(
        self,
        morphers: dict,
        hidden_size: int,
        max_length,
        optim_lr: float,
        tr_args: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optim_lr = optim_lr
        self.max_length = max_length

        # This also includes input embedding layers.
        self.generator_head = SumMarginalHead(
            morphers={col: morpher for col, morpher in morphers.items()},
            hidden_size=hidden_size,
        )

        self.position_embedder = BoringPositionalEncoding(
            max_length=max_length, d_model=hidden_size
        )
        self.input_norm = nn.GELU()
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(max_length - 1),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=tr_args["nhead"],
                dim_feedforward=tr_args["dim_feedforward"],
                dropout=tr_args["dropout"],
                activation="gelu",
                batch_first=True,
            ),
            num_layers=tr_args["num_layers"],
        )

        self.criteria = {
            col: morpher.make_criterion() for col, morpher in morphers.items()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(self, x):
        # Sequences in x run from the first pitch to the last pitch.
        # Predictions run from the _second_ pitch to the last pitch.
        # n x s-1 x k x e
        tr_inputs = self.generator_head.embed_inputs(x)[:, :-1, :, :]
        # n x s-1 x e
        tr_inputs = tr_inputs.sum(dim=-2)
        tr_inputs = self.input_norm(tr_inputs)
        tr_inputs = self.position_embedder(tr_inputs)
        tr_outputs = self.transformer(
            tr_inputs,
            mask=self.causal_mask,
            src_key_padding_mask=x["pad_mask"][:, :-1],
            is_causal=True,
        )
        predictions = self.generator_head(tr_outputs, x)
        return predictions

    def training_step(self, x):
        preds = self(x)
        loss_dict = {
            f"train_{col}_loss": criterion(preds[col], x[col][:, 1:]).mean()
            for col, criterion in self.criteria.items()
        }
        self.log_dict(loss_dict)
        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, x):
        preds = self(x)
        loss_dict = {
            f"validation_{col}_loss": criterion(preds[col], x[col][:, 1:]).mean()
            for col, criterion in self.criteria.items()
        }

        self.log_dict(loss_dict)
        total_loss = sum(loss_dict.values())
        self.log("validation_loss", total_loss)
        return total_loss

    def generate(self, x, **kwargs):
        """Generate pitches.

        x should contain values for all the context features."""

        raise NotImplementedError("TKTK")
