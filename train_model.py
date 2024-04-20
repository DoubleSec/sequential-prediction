import yaml
import torch
import lightning.pytorch as pl

from seqpred.data import prep_data, BaseDataset
from seqpred.quantile_morpher import Quantiler, Integerizer
from seqpred.nn import MargeNet

with open("cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

input_files = [config["train_data_path"]]

morpher_dispatch = {
    "numeric": Quantiler,
    "categorical": Integerizer,
}

inputs = {
    col: morpher_dispatch[tp]
    for [col, tp] in config["features"]
}

# Set up data
base_data, morphers = prep_data(
    data_files=input_files,
    key_cols=config["keys"],
    cols=inputs,
)
morpher_states = {
    col: morpher.save_state_dict()
    for col, morpher in morphers.items()
}
with open("model/morphers.yaml", "w") as f:
    yaml.dump(morpher_states, f)

ds = BaseDataset(
    base_data,
    morphers,
    key_cols=config["keys"],
    return_keys=False,
)

train_ds, valid_ds = torch.utils.data.random_split(
    ds, [0.75, 0.25]
)

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=2048,
    shuffle=True,
    num_workers=4,
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=2048,
    num_workers=4,
)

# Initialize network
net = MargeNet(morphers, config["hidden_size"], "pitcher", config["lr"])

trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    log_every_n_steps=config["log_every_n"],
    precision=config["precision"],
    logger=pl.loggers.TensorBoardLogger("."),
)

trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)
trainer.save_checkpoint("model/latest.ckpt")