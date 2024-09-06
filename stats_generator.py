from math import ceil

import yaml
import torch
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
import polars as pl
from morphers import Integerizer
import streamlit as st

from seqpred.data import prep_data, BaseDataset
from seqpred.nn import SequentialMargeNet
from seqpred.diag import rollout

checkpoint_path = "./model/epoch=22-validation_loss=9.386.ckpt"
data_files = ["./data/2023_data.parquet"]

st.set_page_config(page_title="Synthetic Statistics", layout="wide")


def unmorph(pitches: dict, morphers):
    unmorphed_pitches = {}
    for pk, pv in pitches.items():
        if isinstance(morphers[pk], Integerizer):
            reverse_vocab = {v: k for k, v in morphers[pk].vocab.items()}
            vector = pv.tolist()
            # If there's only one feature
            if not isinstance(vector, list):
                vector = [vector]
            unmorphed_pitches[pk] = [reverse_vocab.get(item, "-") for item in vector]
        else:
            vector = pv.tolist()
            if not isinstance(vector, list):
                vector = [vector]
            qs = morphers[pk].quantiles
            unmorphed_pitches[pk] = [qs[ceil(item * len(qs))] for item in vector]
    return unmorphed_pitches


@st.cache_resource
def load_model_and_data(config_path, checkpoint_path, data_path):

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    model = SequentialMargeNet.load_from_checkpoint(checkpoint_path)

    morpher_dict = model.hparams["morphers"]

    # Set up data
    data, morphers = prep_data(
        data_files=data_path,
        group_by=["game_pk", "at_bat_number"],
        rename=config["rename"],
        cols=None,
        morphers=morpher_dict,
        extra_labels={"pitcher_name": "pitcher_reference"},
    )

    return config, model, morpher_dict, data


config, model, morpher_dict, data = load_model_and_data(
    "cfg/config.yaml",
    checkpoint_path,
    data_files,
)

pitchers = sorted(data["pitcher_reference"].unique().to_list())

with st.sidebar:
    pitcher = st.selectbox("Pitcher", pitchers)
    temperature = st.slider("Generation Temperature", 0.0, 10.0, value=1.0, step=0.1)
    if st.button("Re-run"):
        st.rerun()

pitcher_data = data.filter(pl.col("pitcher_reference") == pitcher)

ds = BaseDataset(
    pitcher_data,
    config["keys"],
    morpher_dict,
    model.hparams["max_length"],
)

dl = torch.utils.data.DataLoader(ds, batch_size=2048)

with torch.inference_mode():

    # dl will always have one batch.
    batch = next(iter(dl))
    x = {
        k: v[:, 0].to(model.device).unsqueeze(1)
        for k, v in batch.items()
        if isinstance(v, torch.Tensor) and k not in ["game_pk", "at_bat_number"]
    }
    for i in range(19):
        generated_pitch = model.generate_one(
            x, keep_attention=False, temperature=temperature
        )
        x = {
            k: torch.cat([v, generated_pitch[k].unsqueeze(-1)], dim=1)
            for k, v in x.items()
            if k != "pad_mask"
        }

    x |= {
        "game_pk": batch["game_pk"],
        "at_bat_number": batch["at_bat_number"],
        "end_position": x["end_of_at_bat"].argmax(dim=1),
    }


generated_df = pl.DataFrame({k: v.detach().cpu().numpy() for k, v in x.items()})
truncated_df = generated_df.with_columns(
    *[
        pl.col(k).list.slice(0, pl.col("end_position") + 1)
        for k, v in generated_df.schema.items()
        if v == pl.List
    ]
)

exploded_df = truncated_df.explode(
    col
    for col in truncated_df.columns
    if col not in ["game_pk", "at_bat_number", "end_position"]
)

desc_dict = {v: k for k, v in morpher_dict["description"].vocab.items()}
intermediate_df = exploded_df.with_columns(pl.col("description").replace(desc_dict))

# Some unmorph stuff
pn_dict = {v: k for k, v in morpher_dict["pitch_name"].vocab.items()}
desc_dict = {v: k for k, v in morpher_dict["description"].vocab.items()}


def unmorph_numeric(x, morpher):
    qs = morpher.quantiles
    return (x * len(qs)).ceil().replace({i: v for i, v in enumerate(qs)})


summary_df = (
    exploded_df.with_columns(
        pitch_name=pl.col("pitch_name").replace(pn_dict, default=None),
        description=pl.col("description").replace(desc_dict, default=None),
        plate_x=unmorph_numeric(pl.col("plate_x"), morpher_dict["plate_x"]),
        plate_z=unmorph_numeric(pl.col("plate_z"), morpher_dict["plate_z"]),
        release_speed=unmorph_numeric(
            pl.col("release_speed"), morpher_dict["release_speed"]
        ),
    )
    .with_columns(
        in_zone=pl.when(
            pl.col("plate_x").is_between(-0.71, 0.71),
            pl.col("plate_z").is_between(1.5, 3.5),
        )
        .then(1)
        .otherwise(0),
        is_whiff=pl.col("description").is_in(
            ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]
        ),
        is_swing=pl.col("description").is_in(
            [
                "swinging_strike",
                "swinging_strike_blocked",
                "missed_bunt",
                "foul",
                "foul_tip",
                "foul_bunt",
                "foul_pitchout",
                "hit_into_play",
            ]
        ),
    )
    .group_by("pitch_name")
    .agg(
        count=pl.len(),
        average_speed=pl.col("release_speed").mean(),
        n_zone=pl.col("in_zone").sum(),
        whiff_percent=pl.col("is_whiff").sum() / pl.col("is_swing").sum(),
    )
    .with_columns(
        zone_pct=pl.col("n_zone") / pl.col("count"),
        percent=pl.col("count") / pl.col("count").sum(),
    )
    .sort("percent", descending=True)
)

colors = {
    "ball": "green",
    "blocked_ball": "green",
    "bunt_foul_tip": "red",
    "called_strike": "red",
    "foul": "red",
    "foul_bunt": "red",
    "foul_pitchout": "red",
    "foul_tip": "red",
    "hit_by_pitch": "blue",
    "hit_into_play": "blue",
    "missed_bunt": "red",
    "pitchout": "green",
    "swinging_strike": "red",
    "swining_strike_blocked": "red",
}

st.dataframe(summary_df)
