import morphers
import yaml

from seqpred.data import prep_data
from seqpred.quantile_morpher import Quantiler

with open("cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

input_files = ["data/2023h1.parquet", "data/2023h2.parquet"]

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
    data_output="data/prepped_data.parquet",
    morpher_output="data/morphers.yaml",
)