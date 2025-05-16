from pathlib import Path
import pandas as pd
import config as c
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import statsmodels.stats.multitest as smm
import statsmodels.formula.api as smf

## LOADING DATA
def load_trials_df(stages="all", root_dir=None):

    if root_dir is None:
        root_dir = Path.cwd()
    print(f"Loading days data from directory: {root_dir}")
    tdf = pd.read_parquet(Path(root_dir) / "trials_df.parquet")

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


def determine_fixation_dur(row):
    """
    Function for data coming from DMS2 protocol where
    fixation_dur wasn't computed in a direct way
    """
    if row["stage"] in (5, 6, 7, 8):
        if (
            row["session"] == 1 and row["cumulative_trial"] == 1
        ):  # pre go is sometimes set to 0.45
            return row.settling_in_dur
        else:
            return row.settling_in_dur + row.pre_go_dur
    elif row["stage"] >= 9:
        return row.pre_go_dur  # settling in dur now accounted for in pre_go_dur
    else:
        KeyError("Stage not found")


def make_fixation_growth_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a dataframe with the max fixation duration for each animal_id, date
    and the change in fixation duration from the previous day
    """

    max_fixation_df = (
        df.query("stage >=5")  # only look at cpoking stages
        .groupby(
            [
                "date",
                "animal_id",
                f"days_relative_to_stage_5",
                "fix_experiment",
            ],
            observed=True,
        )
        .agg(
            max_fixation_dur=("fixation_dur", "max"),
            trials=("trial", "nunique"),
            n_violations=("violations", "sum"),
            n_settling_ins=("n_settling_ins", "sum"),
        )
        .reset_index()
    )
    max_fixation_df["valid_trials"] = (
        max_fixation_df["trials"] - max_fixation_df["n_violations"]
    )
    max_fixation_df.drop(columns=["n_violations"], inplace=True)

    # Compute the difference in fixation duration from the previous day
    max_fixation_df["fixation_growth"] = max_fixation_df.groupby(
        "animal_id"
    ).max_fixation_dur.diff()

    return max_fixation_df

## PLOTs

def box_strip_v1_vs_v2(
    data,
    x,
    order,
    y,
    ax,
    hue_type="experiment",
    alpha=0.75,
    ylabel=None,
    xlabel=None,
    dodge="auto",
    whis=0,
    s=5,
    width=0.5,
    **kwargs,

):
    sns.despine()

    sns.boxplot(
        x=x,
        y=y,
        data=data,
        order=order,
        ax=ax,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=c.EXP_PALETTE,
        fill=False,
        showfliers=False,
        dodge=dodge,
        whis=whis,
        width=width,
        **kwargs,
    )

    if hue_type == "animal":
        hue = "animal_id"
        hue_order = c.HUE_ORDER_ANIMALS
        palette = c.ANIMAL_PALETTE
    else:
        hue = "fix_experiment"
        hue_order = ["V1", "V2"]
        palette = c.EXP_PALETTE

    sns.stripplot(
        x=x,
        y=y,
        data=data,
        order=order,
        ax=ax,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        dodge=dodge,
        legend=False,
        alpha=alpha,
        s=s,
        **kwargs,
    )

    ax.legend(title=None, frameon=False)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


## STATS

"""Statistical Functions"""
def check_normality(data, alpha=0.05):
    """
    Returns True if data passes the Shapiro–Wilk test for normality
    at the specified alpha level, False otherwise.
    """
    stat, p = shapiro(data)
    return p >= alpha

def compare_two_groups(v1_values, v2_values, alpha=0.05):
    """
    Given two arrays of values (V1 and V2), checks normality in each group.
    If both pass, runs Welch's t-test. Otherwise, runs Mann–Whitney U.
    Returns a dict with test results.
    """
    normal_v1 = check_normality(v1_values, alpha=alpha)
    normal_v2 = check_normality(v2_values, alpha=alpha)

    results = {
        "normality_V1": normal_v1,
        "normality_V2": normal_v2,
        "n_V1": len(v1_values),
        "n_V2": len(v2_values),
    }

    if normal_v1 and normal_v2:
        # Use Welch’s t-test (two-sample, unequal variance)
        stat, p_val = ttest_ind(v1_values, v2_values, equal_var=False)
        results["test_type"] = "welch_t"
        results["test_statistic"] = stat
        results["p_val_raw"] = p_val
    else:
        # Use Mann–Whitney U for non-normal data
        stat, p_val = mannwhitneyu(v1_values, v2_values, alternative="two-sided")
        results["test_type"] = "mannwhitney"
        results["test_statistic"] = stat
        results["p_val_raw"] = p_val

    return results

def compare_v1_v2(df, metric_col, alpha=0.05):
    """
    Compare experimental groups V1 and V2 for data in a single stage.
    Assumes that the DataFrame `df` contains data for only one stage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            'fix_experiment' (with values 'V1' or 'V2')
            and the specified metric_col (the numeric variable to compare).
    metric_col : str
        The numeric variable to compare.
    alpha : float, optional
        Significance level for normality tests and final threshold (default 0.05).

    Returns
    -------
    pd.DataFrame
        A tidy DataFrame with a single row summarizing the test results.
    """
    # Extract data for V1 and V2
    v1_data = df.loc[df["fix_experiment"] == "V1", metric_col].to_numpy()
    v2_data = df.loc[df["fix_experiment"] == "V2", metric_col].to_numpy()

    # Compare the two groups
    result = compare_two_groups(v1_data, v2_data, alpha=alpha)
    
    # Return the result as a DataFrame with one row.
    result_df = pd.DataFrame([result])
    return result_df

def compare_v1_v2_multi_sample(df, metric_col, alpha=0.05):
    """
    Fit a mixed effects model to compare V1 and V2 for data in a single stage.
    Assumes that the DataFrame `df` contains data for only one stage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            'animal_id', 'fix_experiment' (with values 'V1' or 'V2'),
            and the specified metric_col (the numeric variable to compare).
    metric_col : str
        The numeric variable to compare.
    alpha : float, optional
        Significance level for the normality check of residuals (default 0.05).

    Returns
    -------
    pd.DataFrame
        A tidy DataFrame with a single row summarizing the mixed model test results.
    """
    required_cols = ["animal_id", "fix_experiment", metric_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")

    # Ensure both groups exist in the data
    v1_df = df[df["fix_experiment"] == "V1"]
    v2_df = df[df["fix_experiment"] == "V2"]
    if v1_df.empty or v2_df.empty:
        raise ValueError("Both experimental groups (V1 and V2) must be present in the data.")

    # Fit the mixed effects model: metric_col ~ fix_experiment with a random intercept for animal_id
    model = smf.mixedlm(formula=f"{metric_col} ~ fix_experiment",
                        data=df,
                        groups=df["animal_id"])
    try:
        model_fit = model.fit(method="lbfgs", disp=False)
    except Exception as e:
        raise RuntimeError(f"Model fitting failed with error: {e}")

    coef_key = "fix_experiment[T.V2]"
    if coef_key not in model_fit.params.index:
        raise ValueError("Coefficient for fix_experiment[T.V2] not found in model fit.")

    t_val = model_fit.tvalues[coef_key]
    p_val = model_fit.pvalues[coef_key]

    # Check normality of the residuals from the model
    residuals = model_fit.resid
    residuals_are_normal = check_normality(residuals, alpha=alpha)

    result = {
        "test_type": "mixedlm_random_intercept",
        "test_statistic": t_val,
        "p_val_raw": p_val,
        "residuals_normal": residuals_are_normal,
    }
    result_df = pd.DataFrame([result])
    return result_df