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

st.set_page_config(page_title="Generation Tester", layout="wide")

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

markers = {
    "4-Seam Fastball": "o",
    "Changeup": "D",
    "Curveball": "v",
    "Cutter": ">",
    "Eephus": "P",
    "Forkball": "D",
    "Knuckle Curve": "v",
    "Knuckleball": "P",
    "Other": "P",
    "Pitch Out": "P",
    "Screwball": "<",
    "Sinker": "o",
    "Slider": ">",
    "Slow Curve": "v",
    "Slurve": ">",
    "Split-Finger": "D",
    "Sweeper": ">",
}


def unmorph(pitches, morphers):
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
            # raise NotImplementedError("Later")
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
    )
    ds = BaseDataset(
        data,
        config["keys"],
        morpher_dict,
        model.hparams["max_length"],
    )

    return config, model, morpher_dict, ds


config, model, morpher_dict, ds = load_model_and_data(
    "cfg/config.yaml",
    checkpoint_path,
    data_files,
)

with st.sidebar:
    pitch_index = st.number_input("Plate Appearance Index", 0, len(ds) - 1)
    example = ds[pitch_index]
    after_n_pitches = 1
    temperature = st.slider("Generation Temperature", 0.0, 10.0, value=1.0, step=0.1)
    if st.button("Re-run"):
        st.rerun()

with torch.inference_mode():
    # noooooooo
    inning_mask = (
        torch.arange(example["description"].shape[0]) < after_n_pitches
    ) & ~torch.isinf(example["pad_mask"])

    x = {
        k: v[inning_mask].to(model.device).unsqueeze(0)
        for k, v in example.items()
        if isinstance(v, torch.Tensor)
    }
    n_generated = None
    attention_per_step = []
    for i in range(30):
        generated_pitch = model.generate_one(
            x, keep_attention=True, temperature=temperature
        )
        x = {
            k: torch.cat([v, generated_pitch[k].unsqueeze(0)], dim=1)
            for k, v in x.items()
            if k != "pad_mask"
        }
        print(x)
        # Get attention activations.
        attention = [
            torch.nn.functional.softmax(layer.gq_attn.attention_activation, dim=-1)
            for layer in model.transformer.transformer_layers
        ]
        print(attention[0].shape)
        attention_per_step.append(rollout(attention, head_fusion="mean"))
        if (
            generated_pitch["end_of_at_bat"].item()
            == morpher_dict["end_of_at_bat"].vocab[True]
        ):
            n_generated = i + 1
            print(f"Reached end of at-bat: generated {i+1} pitches")
            break

    context = {k: v[:, :after_n_pitches] for k, v in x.items()}
    generated = {k: v[:, after_n_pitches:] for k, v in x.items()}
    # yuck
    attention_at_each_step = torch.zeros([n_generated, n_generated]).to(
        attention_per_step[0]
    )
    for i in range(n_generated):
        attention_at_each_step[i, : i + 1] = attention_per_step[i]

context_df = pl.DataFrame(
    {"source": "context"}
    | unmorph(
        {k: v.squeeze().cpu().numpy() for k, v in context.items()},
        morpher_dict,
    )
)
generated_df = pl.DataFrame(
    {"source": "generated"}
    | unmorph(
        {k: v.squeeze().cpu().numpy() for k, v in generated.items()},
        morpher_dict,
    )
)
pitch_df = pl.concat([context_df, generated_df]).with_row_index(offset=1)

col1, col2 = st.columns([0.7, 0.3])

with col1:

    st.markdown("#### Pitches")
    st.dataframe(pitch_df)

    fig, ax = plt.subplots(1)
    fig.set_figwidth(12)
    fig.set_figheight(2)
    sns.heatmap(attention_at_each_step.cpu().numpy(), ax=ax, annot=True, linewidth=0.2)
    st.markdown("#### Attention")
    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(1)
    fig.set_figwidth(6.8)
    fig.set_figheight(10)
    (
        so.Plot()
        .add(
            so.Dot(pointsize=25),
            data=pitch_df,
            x="plate_x",
            y="plate_z",
            color="description",
            marker="pitch_name",
            legend=False,
        )
        .add(
            so.Paths(),
            x=[-0.71, 0.71, 0.71, -0.71, -0.71],
            y=[3.5, 3.5, 1.5, 1.5, 3.5],
            legend=False,
        )
        .add(
            so.Text({"fontweight": "bold"}, color="white"),
            data=pitch_df,
            x="plate_x",
            y="plate_z",
            text="index",
        )
        .scale(
            color=colors,
            marker=markers,
        )
        .limit(x=(-1.7, 1.7), y=(-0.5, 4.5))
        .on(ax)
        .show()
    )
    st.markdown("#### Pitch Locations")
    st.pyplot(fig)

with st.sidebar:
    if n_generated is not None:
        st.markdown(f"Generated {n_generated} pitches")
    else:
        st.markdown("Reached generation limit")
