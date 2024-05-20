from seqpred.data import prep_data, BaseDataset
from seqpred.nn import SequentialMargeNet
import yaml
import torch
import polars as pl
from morphers.base.categorical import Integerizer
import streamlit as st

checkpoint_path = "./model/epoch=39-validation_loss=1.731.ckpt"
data_files = ["./data/2023_data.parquet"]

st.set_page_config(page_title="Generation Tester", layout="wide")


def unmorph(pitches, morphers):
    unmorphed_pitches = {}
    for pk, pv in pitches.items():
        if isinstance(morphers[pk], Integerizer):
            reverse_vocab = {v: k for k, v in morphers[pk].vocab.items()}
            vector = pv.tolist()
            unmorphed_pitches[pk] = [reverse_vocab.get(item, "-") for item in vector]
        else:
            raise NotImplementedError("Later")
    return unmorphed_pitches


@st.cache_resource
def load_model_and_data(config_path, checkpoint_path, data_path):

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    model = SequentialMargeNet.load_from_checkpoint(checkpoint_path)

    morpher_dict = model.hparams["morphers"]

    data, morphers = prep_data(
        data_files=data_path,
        rename=config["rename"],
        morphers=morpher_dict,
    )
    ds = BaseDataset(
        data,
        morpher_dict,
        model.hparams["max_length"],
    )

    return config, model, morpher_dict, ds


config, model, morpher_dict, ds = load_model_and_data(
    "cfg/config.yaml",
    "./model/epoch=39-validation_loss=1.731.ckpt",
    ["./data/2023_data.parquet"],
)

with st.sidebar:
    game_index = st.number_input("Game Index", 0, len(ds) - 1)
    example = ds[game_index]
    after_n_pitches = st.slider("After N Pitches", 1, 200, value=100)
    max_to_generate = st.slider("Max to Generate", 0, 300, value=300)
    temperature = st.slider("Generation Temperature", 0.0, 10.0, value=1.0, step=0.1)
    if st.button("Re-run"):
        st.rerun()

with torch.inference_mode():
    # noooooooo
    inning_mask = (
        torch.arange(example["end_of_inning"].shape[0]) < after_n_pitches
    ) & ~torch.isinf(example["pad_mask"])

    x = {
        k: v[inning_mask].to(model.device).unsqueeze(0)
        for k, v in example.items()
        if isinstance(v, torch.Tensor)
    }

    n_generated = None
    for i in range(max_to_generate):
        generated_pitch = model.generate_one(x, temperature=temperature)
        x = {
            k: torch.cat([v, generated_pitch[k].unsqueeze(0)], dim=1)
            for k, v in x.items()
            if k != "pad_mask"
        }
        if generated_pitch["end_of_game"].item() == 1:
            n_generated = i + 1
            print(f"Reached end of game: generated {i+1} pitches")
            break

    context = {k: v[:, :after_n_pitches] for k, v in x.items()}
    generated = {k: v[:, after_n_pitches:] for k, v in x.items()}

context_df = pl.DataFrame(
    unmorph(
        {k: v.squeeze().cpu().numpy() for k, v in context.items()},
        morpher_dict,
    )
)
generated_df = pl.DataFrame(
    unmorph(
        {k: v.squeeze().cpu().numpy() for k, v in generated.items()},
        morpher_dict,
    )
)

with st.sidebar:
    if n_generated is not None:
        st.markdown(f"Generated {n_generated} pitches")
    else:
        st.markdown("Reached generation limit")


col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Context Pitches")
    st.dataframe(context_df, height=600)
with col2:
    st.markdown("#### Generated Pitches")
    st.dataframe(generated_df, height=600)
