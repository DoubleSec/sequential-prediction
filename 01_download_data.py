# This only needs to happen once, really.

from os import mkdir
from os.path import dirname
import warnings

import pybaseball
import yaml

pybaseball.cache.enable()

with open("./cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

print(config["train_data_path"])
try:
    dir_to_make = dirname(config["train_data_path"])
    mkdir(dir_to_make)
    print(f"Created {dir_to_make}")
except FileExistsError:
    print(f"Directory {dir_to_make} already exists, skipping creation")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    some_data = pybaseball.statcast("2019-01-01", "2023-12-31")
    some_data.to_parquet(config["train_data_path"])
