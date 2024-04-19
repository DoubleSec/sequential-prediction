# You may not need this, but if you want to preprocess data
# you can use this file.

import morphers
import yaml

from seqpred.data import prep_data
from seqpred.quantile_morpher import Quantiler

with open("cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

input_files = [config["train_data_path"]]

morpher_dispatch = {
    "numeric": Quantiler,
    "categorical": morphers.Integerizer,
}

# Flatten features
inputs = {
    col: morpher_dispatch[tp]
    for feature_list in config["features"]
    for col, tp in feature_list
}

prep_data(
    data_files=input_files,
    key_cols=config["keys"],
    cols=inputs,
    data_output=config["processed_data_path"],
    morpher_output=config["morpher_path"],
)