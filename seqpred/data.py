import polars as pl
import torch
import yaml
import morphers

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


def prep_data(
    data_files: str,
    rename: dict = None,
    key_cols: list = [],
    cols: dict = None,
    morphers: dict = None,
    data_output: str = None,
    morpher_output: str = None,
    write: bool = False,
):
    """Prepare data according to a morpher dict."""

    input_dataframes = [pl.read_parquet(file) for file in data_files]
    input_data = pl.concat(input_dataframes)

    if rename is not None:
        input_data = input_data.rename({"type": "result_type"})

    # TKTK use existing morpher states
    if morphers is None:
        morphers = {
            feature: morpher_class.from_data(input_data[feature])
            for feature, morpher_class in cols.items()
        }
    else:
        morphers = morphers

    input_data = (
        input_data.select(
            # keys
            *[pl.col(key) for key in key_cols],
            # morphed inputs
            *[
                morpher(pl.col(feature))
                for feature, morpher in morphers.items()
            ],
        )
        # Only drop nulls based on inputs
        .drop_nulls([feature for feature in morphers])
    )

    if write:
        input_data.write_parquet(data_output)
        morpher_dict = {column: morpher.save_state_dict() for column, morpher in morphers.items()}
        with open(morpher_output, "w") as f:
            yaml.dump(morpher_dict, f)

    return input_data, morphers

class BaseDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        ds: pl.DataFrame,
        morphers: dict,
        key_cols: list = [],
        return_keys: bool = False
        ):

        super().__init__()
        self.ds = ds
        self.morphers = morphers
        self.key_cols = key_cols
        self.return_keys = return_keys

    def __len__(self):
        return self.ds.height

    def __getitem__(self, idx):
        row = self.ds.row(idx, named=True)
        inputs = {
            k: torch.tensor(row[k], dtype=morpher.required_dtype)
            for k, morpher in self.morphers.items()
        }

        return_dict = inputs
        if self.return_keys:
            return_dict = return_dict | {key: row[key] for key in self.key_cols}

        return return_dict
