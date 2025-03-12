from pathlib import Path
import pandas as pd


## LOADING DATA
def load_trials_df(stages="all", root_dir=None):

    if root_dir is None:
        root_dir = Path.cwd()
    print(f"Loading days data from directory: {root_dir}")
    tdf = pd.read_parquet(Path(root_dir) / "days_df.parquet")

    if stages == "spoke":
        print("Filtering for spoke stages")
        tdf = tdf.query("stage < 5").copy()
    elif stages == "fix_growth":
        print("Filtering for fix growth stages")
        tdf = tdf.query("stage > 4 and stage < 9").copy()
    elif stages == "probe":
        print("Filtering for probe stages")
        tdf = tdf.query("stage == 9 or stage == 10").copy()

    return tdf


def load_days_df(stages="all", root_dir=None):
    if root_dir is None:
        root_dir = Path.cwd()
    print(f"Loading days data from directory: {root_dir}")
    ddf = pd.read_parquet(Path(root_dir) / "days_df.parquet")

    if stages == "spoke":
        print("Filtering for spoke stages")
        ddf = ddf.query("stage < 5").copy()
    elif stages == "fix_growth":
        print("Filtering for fix growth stages")
        ddf = ddf.query("stage > 4 and stage < 9").copy()
    elif stages == "probe":
        print("Filtering for probe stages")
        ddf = ddf.query("stage == 9 or stage == 10").copy()

    return ddf


def load_poke_df(stages="all", root_dir=None):

    if root_dir is None:
        root_dir = Path.cwd()
    print(f"Loading poke data from directory: {root_dir}")
    pdf = pd.read_parquet(Path(root_dir) / "poke_df.parquet")

    if stages == "fix_growth":
        print("Filtering for fix growth stages")
        pdf = pdf.query("stage > 4 and stage < 9")
    elif stages == "probe":
        print("Filtering for probe stages")
        pdf = pdf.query("stage == 9 or stage == 10")

    return pdf


## MANIPULATING DATA
def compute_days_relative_to_stage(
    df: pd.DataFrame,
    stage: int,
    date_col_name: str = "date",
    dense_rank: str = True,
) -> pd.DataFrame:
    """
    Compute the number of days relative to a specific stage in the dataframe.

    params:
    -------
    df : pd.DataFrame
        DataFrame containing the data
    stage : int
        The specific stage to compute the relative days for
    date_col_name : str, optional
        The name of the column containing the dates, by default "date"
    dense_rank : bool, (optional, default True)
        If True, days are computed as a dense rank, meaning that if there are
        multiple days of no data in-between the same stage, they will be incremented
        as 1 as opposed to the number of calendar days in-between.


    returns:
    --------
    pd.DataFrame
    DataFrame with an additional column indicating the number of days relative to the stage
    """
    # convert to date time
    df["datetime_col"] = pd.to_datetime(df[date_col_name])

    if dense_rank:
        return (
            df.groupby("animal_id")[df.columns]
            .apply(compute_relative_dense_dates, stage=stage)
            .reset_index(drop=True)
        )

    else:
        # find first day in stage for each animal
        min_date_stage = (
            df.query("stage == @stage")
            .groupby("animal_id")["datetime_col"]
            .min()
            .reset_index()
        )
        min_date_stage.rename(
            columns={"datetime_col": f"min_date_stage_{stage}"}, inplace=True
        )

        # merge on animal_id & subtract min column wise
        df = df.merge(min_date_stage, on="animal_id", how="left")
        df[f"days_relative_to_stage_{stage}"] = (
            df["datetime_col"] - df[f"min_date_stage_{stage}"]
        ).dt.days

        df.drop(columns=["datetime_col", f"min_date_stage_{stage}"], inplace=True)

    return df.copy().reset_index()


def compute_relative_dense_dates(df: pd.DataFrame, stage: int) -> pd.DataFrame:
    """
    Compute the relative dense dates for each stage in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    stage : int
        The specific stage to compute the relative dense dates for

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional column indicating the relative dense dates
    """
    # Compute the dense rank for each date
    df["dense_rank"] = df["datetime_col"].rank(method="dense").astype(int)

    # Find the minimum dense rank for the specified stage
    min_stage_rank = df.query("stage == @stage")["dense_rank"].min()

    # Compute the relative dense dates by subtracting the minimum dense rank
    df[f"days_relative_to_stage_{stage}"] = df["dense_rank"] - min_stage_rank

    # Drop the dense rank column
    df.drop(columns=["dense_rank"], inplace=True)

    return df
