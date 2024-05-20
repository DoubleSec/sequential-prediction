import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from seqpred.data import prep_data, BaseDataset
from morphers.polars.continuous import PolarsQuantiler
from morphers.polars.categorical import PolarsIntegerizer
from seqpred.nn import SequentialMargeNet

torch.set_float32_matmul_precision("medium")

with open("cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

input_files = [config["train_data_path"]]

morpher_dispatch = {
    "numeric": PolarsQuantiler,
    "categorical": PolarsIntegerizer,
}

inputs = {
    col: (morpher_dispatch[tp], kwargs) for [col, tp, kwargs] in config["features"]
}
# Loss weighting, according to inverse of feature rank
n_features = len(config["features"])
print(f"n_features: {n_features}")
# So they still sum to one.
weight_multiplier = n_features / (n_features - (n_features - 1) * 0.5)
loss_weights = {
    ls[0]: ((n_features - i) / n_features) * weight_multiplier
    for i, ls in enumerate(config["features"])
}

# Set up data
base_data, morphers = prep_data(
    data_files=input_files,
    rename=config["rename"],
    cols=inputs,
)
max_length = base_data["n_pitches"].max()
print(f"Max length: {max_length}")

morpher_states = {col: morpher.save_state_dict() for col, morpher in morphers.items()}
with open("model/morphers.yaml", "w") as f:
    yaml.dump(morpher_states, f)

ds = BaseDataset(
    base_data,
    morphers,
    max_length,
)

train_ds, valid_ds = torch.utils.data.random_split(ds, [0.75, 0.25])

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["n_workers"],
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=config["batch_size"],
    num_workers=config["n_workers"],
)

# Initialize network
net = SequentialMargeNet(
    morphers=morphers,
    hidden_size=config["hidden_size"],
    max_length=max_length,
    optim_lr=config["lr"],
    tr_args=config["tr_args"],
    # loss_weights=loss_weights,
)

trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    log_every_n_steps=config["log_every_n"],
    precision=config["precision"],
    logger=pl.loggers.TensorBoardLogger("."),
    accumulate_grad_batches=config["accumulate_batches"],
    callbacks=[
        ModelCheckpoint(
            dirpath="./model",
            save_top_k=1,
            monitor="validation_loss",
            filename="{epoch}-{validation_loss:.3f}",
        )
    ],
)

trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)
