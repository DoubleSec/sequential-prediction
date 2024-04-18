import polars as pl
import torch
import yaml
import morphers

# So we can re-use easily enough.
DEFAULT_MORPHER_DISPATCH = {
    "numeric": morphers.RankScaler,
    "categorical": morphers.Integerizer,
}



def prep_data(
    data_files: str,
    key_cols: list,
    cols: dict,
    data_output: str,
    morpher_output: str,
):
    """Prepare data according to a morpher dict."""

    input_dataframes = [pl.read_parquet(file) for file in data_files]
    input_data = pl.concat(input_dataframes)

    # TKTK use existing morpher states
    morphers = {
        feature: morpher_class.from_data(input_data[feature])
        for feature, morpher_class in cols.items()
    }

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

    input_data.to_parquet(data_output)
    morpher_dict = {column: morpher.save_state_dict() for column, morpher in morphers.items()}
    with open(morpher_output, "w") as f:
        yaml.dump(morpher_dict, f)
