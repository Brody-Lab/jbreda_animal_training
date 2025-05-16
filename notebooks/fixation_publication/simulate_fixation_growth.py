import numpy as np
import pandas as pd

import config as c
import utils as u

from numpy.random import default_rng

class SimulateV1FixationGrowth:
    """
    A class to post-hoc simulate fixation growth data for a given animal,
    using empirical session-level statistics plus user-defined or default
    warm-up (wu_vr) and post-warm-up (pwu_vr) violation rates.
    """

    def __init__(
        self,
        tdf: pd.DataFrame,
        animal_id: str,
        wu_vr: float = None,
        pwu_vr: float = None,
        max_n_sessions: int = 90,
        experiment_name: str = "experiment",
        seed: int = None
    ):
        """
        Parameters
        ----------
        tdf : pd.DataFrame
            DataFrame of all trials for many animals. Must contain columns:
            [animal_id, stage, date, days_relative_to_stage_5, trial, violations].
            If 'days_relative_to_stage_5' is missing, it will be computed.
        animal_id : str
            The animal to filter on.
        wu_vr : float, optional
            Warm-up violation rate mean (if None, use empirical).
            If == 0, no violations are simulated in warm-up.
        pwu_vr : float, optional
            Post-warm-up violation rate mean (if None, use empirical).
            If == 0, no violations are simulated post warm-up.
        max_n_sessions : int, optional
            Maximum number of sessions to simulate before stopping.
        experiment_name : str, optional
            A label for the experiment (useful in summary).
        seed : int, optional
            If provided, used to initialize the RNG for reproducible simulations.
        """

        # If the required column is missing, compute days_relative_to_stage_5
        if "days_relative_to_stage_5" not in tdf.columns:
            tdf = u.compute_days_relative_to_stage(tdf.copy(), stage=5)

        self.tdf = tdf
        self.animal_id = animal_id
        self.max_n_sessions = max_n_sessions
        self.experiment_name = experiment_name

        # Renamed to reflect days to target in the simulation
        self.simulated_days_to_target = 0

        # Start with 0.01s (10 ms) as the "previous session’s final fixation"
        self.prev_session_dur = 0.010

        # Create a random number generator with an optional seed.
        self.rng = default_rng(seed)  # If seed is None, it uses a random seed.

        # Get the animal-level parameters (empirical means/vars, etc.)
        (
            self.emperical_wu_vr_mean,
            self.emperical_wu_vr_var,
            self.emperical_pwu_vr_mean,
            self.emperical_pwu_vr_var,
            self.emperical_n_trial_mean,
            self.emperical_n_trial_var,
            self.emperical_n_days_to_target
        ) = self.get_animal_parameters()

        # Decide on warm-up and post-warm-up violation rates
        # (If user didn’t specify, use empirical mean)
        self.wu_vr = self.emperical_wu_vr_mean if wu_vr is None else wu_vr
        self.pwu_vr = self.emperical_pwu_vr_mean if pwu_vr is None else pwu_vr

    def get_animal_parameters(self):
        """
        Filters `tdf` for this animal in the relevant GROWING_STAGES,
        then computes session-level stats for n_trials, violation_rate.

        Returns
        -------
        (wu_vr_mean, wu_vr_var, pwu_vr_mean, pwu_vr_var, n_trial_mean, n_trial_var, n_days_to_target)
        """

        # Filter the DataFrame for this animal and relevant stages
        animal_df = self.tdf.query(
            "animal_id == @self.animal_id and stage in @c.GROWING_STAGES"
        ).copy()

        # compute trial summary
        trial_summary = (
            animal_df.groupby(["animal_id", "date", "days_relative_to_stage_5"])
            .agg(
                n_trials=("trial", "nunique"),
            )
            .reset_index()
        )

        warm_up_summary = (
            animal_df.query("warm_up_imp == True")
            .groupby(["animal_id", "date", "days_relative_to_stage_5"])
            .agg(warm_up_violation_rate=("violations", "mean"))
            .reset_index()
        )

        non_warm_up_summary = (
            animal_df.query("warm_up_imp == False")
            .groupby(["animal_id", "date", "days_relative_to_stage_5"])
            .agg(non_warm_up_violation_rate=("violations", "mean"))
            .reset_index()
        )

        # merge the summaries
        session_summary = pd.merge(
            trial_summary, warm_up_summary,
            on=["animal_id", "date", "days_relative_to_stage_5"],
            how="left"
        )
        session_summary = pd.merge(
            session_summary, non_warm_up_summary,
            on=["animal_id", "date", "days_relative_to_stage_5"],
            how="left"
        )

        # Compute global means/vars
        pwu_vr_mean = session_summary["non_warm_up_violation_rate"].mean()
        pwu_vr_var = session_summary["non_warm_up_violation_rate"].var(ddof=1)
        if pd.isna(pwu_vr_var):
            pwu_vr_var = 0.0

        wu_vr_mean = session_summary["warm_up_violation_rate"].mean()
        wu_vr_var = session_summary["warm_up_violation_rate"].var(ddof=1)
        if pd.isna(wu_vr_var):
            wu_vr_var = 0.0

        n_trial_mean = session_summary["n_trials"].mean()
        n_trial_var = session_summary["n_trials"].var(ddof=1)
        if pd.isna(n_trial_var):
            n_trial_var = 0.0

        # For demonstration, define "n_days_to_target" as total number of sessions in growing stages
        n_days_to_target = len(session_summary)

        return (
            wu_vr_mean,
            wu_vr_var,
            pwu_vr_mean,
            pwu_vr_var,
            n_trial_mean,
            n_trial_var,
            n_days_to_target
        )

    def mean_var_to_alpha_beta(self, mean, var):
        """
        Given a mean and variance for a Beta distribution, compute alpha and beta.
        Clamps or returns (0,0) if mean == 0 (interpreted as no violations).
        """
        # If mean == 0 => no violation scenario
        if mean == 0:
            return 0.0, 0.0

        # Clamp mean just in case (particularly if user asked for >0.5, but they said they'd never do so)
        if mean >= 1.0:
            mean = 0.9999
        if mean <= 0.0:
            mean = 0.0001

        if var < 1e-6:
            var = 1e-6

        # Standard Beta parameterization
        alpha = (mean**2) * ((1.0 - mean) / var - 1.0 / mean)
        beta = alpha * (1.0 / mean - 1.0)

        # If we get negative alpha/beta, fallback to a small-variance distribution near mean
        if alpha <= 0 or beta <= 0:
            alpha = max(mean * 100, 1e-3)
            beta = max((1 - mean) * 100, 1e-3)

        return alpha, beta

    def sample_session_violation_rate(self, in_warm_up: bool) -> float:
        """
        Sample a session-level violation rate from a Beta distribution
        based on either warm-up (wu_vr) or post warm-up (pwu_vr).

        Returns a single float violation rate for the session.
        """
        if in_warm_up:
            mean_vr = self.wu_vr
            var_vr = self.emperical_wu_vr_var
        else:
            mean_vr = self.pwu_vr
            var_vr = self.emperical_pwu_vr_var

        if mean_vr == 0:
            # If mean is 0 => no violations at all
            return 0.0

        alpha, beta = self.mean_var_to_alpha_beta(mean_vr, var_vr)
        if (alpha == 0) and (beta == 0):
            # Interpreted as a "no violation" scenario
            return 0.0

        return self.rng.beta(alpha, beta)

    def sample_n_trials(self):
        """
        Samples the number of trials from a Normal distribution using the empirical
        mean and variance. Returns at least 1 trial.
        """
        sd = np.sqrt(self.emperical_n_trial_var)
        raw = self.rng.normal(self.emperical_n_trial_mean, sd)
        n_trials = int(round(raw))
        return max(n_trials, 1)

    def compute_warm_up_step(self) -> float:
        """
        Computes the warm-up step size.
        Warm-up step = (prev_session_dur - 0.01) / 20.
        """
        warm_up_target_duration = self.prev_session_dur
        starting_duration = 0.01
        n_warm_up_trials = 20

        return (warm_up_target_duration - starting_duration) / float(n_warm_up_trials)

    def simulate_session(self) -> pd.DataFrame:
        """
        Simulates a single session, returning a DataFrame with:
            [session, trial, fixation_dur, violation (bool), warm_up_on (bool)]

        Workflow:
          - Determine if warm-up is on/off (based on prev_session_dur > 0.01).
          - Sample session-level violation rate from Beta distribution.
          - For each trial:
              -> draw a Bernoulli violation from that session-level p
              -> if no violation, update fixation_dur accordingly.
        """

        n_trials = self.sample_n_trials()

        # Warm-up is ON if the previous session's final fixation was > 0.01
        warm_up_on = (self.prev_session_dur > 0.01)
        if warm_up_on:
            warm_up_step = self.compute_warm_up_step()

        # Sample a single session-level violation rate for warm-up or post-warm-up
        wuvr = self.sample_session_violation_rate(in_warm_up=True)
        pwuvr = self.sample_session_violation_rate(in_warm_up=False)

        # Start session at 0.01
        fixation_dur = 0.01

        trial_data = {
            "session": [],
            "trial": [],
            "fixation_dur": [],
            "violation": [],
            "warm_up_on": []
        }

        for t_idx in range(1, n_trials + 1):
            if warm_up_on and (t_idx <= 20):
                # Use the warm-up violation rate
                p_violate = wuvr
                did_violate = (self.rng.random() < p_violate)
                if not did_violate:
                    fixation_dur += warm_up_step
            else:
                # Post-warm-up
                p_violate = pwuvr
                did_violate = (self.rng.random() < p_violate)
                if not did_violate:
                    # Grow by 0.1% of current fixation or 0.001s, whichever is larger
                    grow_amount = max(0.001, 0.001 * fixation_dur)
                    fixation_dur += grow_amount

            trial_data["session"].append(self.simulated_days_to_target + 1)
            trial_data["trial"].append(t_idx)
            trial_data["fixation_dur"].append(fixation_dur)
            trial_data["violation"].append(did_violate)
            trial_data["warm_up_on"].append(warm_up_on and (t_idx <= 20))

        # End of session: increment day count, store final fixation for next session
        self.simulated_days_to_target += 1
        self.prev_session_dur = fixation_dur

        return pd.DataFrame(trial_data)

    def run_simulation(self):
        """
        Runs up to `max_n_sessions` sessions or until fixation_dur >= 2.0s.
        Returns:
          - A DataFrame of all trial-level data.
          - A single-row summary (dict) with:
             {
                 'animal_id': ...,
                 'wu_vr': ...,
                 'pwu_vr': ...,
                 'emperical_pwu_vr_var': ...,
                 'emperical_pwu_vr_mean': ...,
                 'emperical_wu_vr_var': ...,
                 'emperical_wu_vr_mean': ...,
                 'simulated_days_to_target': ...,
                 'experiment_name': ...,
                 'emperical_days_to_target': ...
             }
        """

        all_sessions = []
        for _ in range(self.max_n_sessions):
            session_df = self.simulate_session()
            all_sessions.append(session_df)

            # Stop if we reached or exceeded 2.0s fixation
            if self.prev_session_dur >= 2.0:
                break

        sim_data = pd.concat(all_sessions, ignore_index=True)

        summary = {
            "animal_id": self.animal_id,
            "wu_vr": self.wu_vr,
            "pwu_vr": self.pwu_vr,
            "emperical_pwu_vr_var": self.emperical_pwu_vr_var,
            "emperical_pwu_vr_mean": self.emperical_pwu_vr_mean,
            "emperical_wu_vr_var": self.emperical_wu_vr_var,
            "emperical_wu_vr_mean": self.emperical_wu_vr_mean,
            "emperical_days_to_target": self.emperical_n_days_to_target,
            "simulated_days_to_target": self.simulated_days_to_target,
            "experiment_name": self.experiment_name,
        }

        return sim_data, summary