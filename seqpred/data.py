import polars as pl
import torch
import yaml
from morphers.polars.categorical import PolarsIntegerizer


def load_morphers(
    state_path: str,
    cols: dict,
    type_map: dict,
):
    with open(state_path, "r") as f:
        morpher_states = yaml.load(f, Loader=yaml.CLoader)

    morphers = {
        col: type_map[ctype].from_state_dict(morpher_states[col])
        for col, ctype in cols.items()
    }
    return morphers


def pad_tensor_dict(tensor_dict, max_length, return_mask: bool = True):
    """
    Pad a tensor dict up to the max length.
    Padded Location = 0
    """

    init_length = next(iter(tensor_dict.values())).shape[0]
    padded_tensor_dict = {
        k: torch.nn.functional.pad(v, [0, max_length - init_length], value=0)
        for k, v in tensor_dict.items()
    }
    if return_mask:
        pad_mask = torch.zeros([max_length], dtype=torch.float)
        pad_mask[:init_length] = 1
        return padded_tensor_dict, pad_mask
    else:
        return padded_tensor_dict


def prep_data(
    data_files: str,
    rename: dict = None,
    cols: dict = None,
    morphers: dict = None,
    data_output: str = None,
    morpher_output: str = None,
    write: bool = False,
):
    """Prepare data according to a morpher dict.
    The end_of_game feature is special and is constructed on the fly."""

    input_dataframes = [pl.read_parquet(file) for file in data_files]
    input_data = pl.concat(input_dataframes)

    if rename is not None:
        input_data = input_data.rename({"type": "result_type"})

    # Use existing morpher states
    if morphers is None:
        morphers = {
            feature: morpher_class.from_data(input_data[feature], **kwargs)
            for feature, (morpher_class, kwargs) in cols.items()
            if feature != "end_of_game"
        } | {"end_of_game": PolarsIntegerizer({0: 0, 1: 1})}
    else:
        morphers = morphers

    input_data = (
        input_data.select(
            "game_pk",
            "at_bat_number",
            "pitch_number",
            # morphed inputs
            *[
                morpher(morpher.fill_missing(pl.col(feature))).alias(feature)
                for feature, morpher in morphers.items()
                if feature != "end_of_game"
            ],
        )
        .sort(["at_bat_number", "pitch_number"])
        .group_by("game_pk", maintain_order=True)
        .agg(
            *[
                pl.col(feature)
                for feature in morphers.keys()
                if feature != "end_of_game"
            ],
            n_pitches=pl.col("pitch_number").count(),
        )
    )

    if write:
        input_data.write_parquet(data_output)
        morpher_dict = {
            column: morpher.save_state_dict() for column, morpher in morphers.items()
        }
        with open(morpher_output, "w") as f:
            yaml.dump(morpher_dict, f)

    return input_data, morphers


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        ds: pl.DataFrame,
        morphers: dict,
        max_length: int,
    ):

        super().__init__()
        self.ds = ds
        self.morphers = morphers
        self.max_length = max_length

    def __len__(self):
        return self.ds.height

    def __getitem__(self, idx):
        row = self.ds.row(idx, named=True)
        inputs, pad_mask = pad_tensor_dict(
            {
                k: torch.tensor(row[k], dtype=morpher.required_dtype)
                for k, morpher in self.morphers.items()
                if k != "end_of_game"
            },
            max_length=self.max_length,
        )
        # Make the end of game indicator
        inputs["end_of_game"] = torch.zeros_like(inputs["pitcher"])
        inputs["end_of_game"][row["n_pitches"] - 1] = 1

        inputs |= {"game_pk": row["game_pk"], "pad_mask": pad_mask}
        return inputs
