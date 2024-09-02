# This only needs to happen once, really.

import os
from os import mkdir
from os.path import dirname
import warnings

import pandas as pd
import pybaseball
import yaml

"""Please note that at the moment this file doesn't really make sense, and was mostly
structured like this because it was convenient at the time. You may want to look at this
carefully before using it."""

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

    if not os.path.exists(config["train_data_path"]):
        # Download statcast data.
        warnings.simplefilter("ignore")
        some_data = pybaseball.statcast("2019-01-01", "2023-12-31")
        some_data.to_parquet(config["train_data_path"])

    else:
        # Add the pitcher and batter names.
        some_data = pd.read_parquet(config["train_data_path"])
        print(some_data.shape)

        pitchers = pybaseball.playerid_reverse_lookup(
            some_data.pitcher.unique().tolist()
        )
        pitchers["pitcher_name"] = pitchers.name_last.str.title().str.cat(
            pitchers.name_first.str.title(), sep=", "
        )
        pitchers = pitchers[["key_mlbam", "pitcher_name"]]
        batters = pybaseball.playerid_reverse_lookup(some_data.batter.unique().tolist())
        batters["batter_name"] = batters.name_last.str.title().str.cat(
            batters.name_first.str.title(), sep=", "
        )
        batters = batters[["key_mlbam", "batter_name"]]
        some_data = some_data.merge(pitchers, left_on="pitcher", right_on="key_mlbam")
        some_data = some_data.merge(batters, left_on="batter", right_on="key_mlbam")
        print(some_data.shape)
        some_data.to_parquet(config["train_data_path"])
