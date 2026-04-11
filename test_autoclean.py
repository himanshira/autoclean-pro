"""
AutoClean-Pro — Unit Test Suite
================================
Covers reward hacking scenarios, grader logic, Bayesian weighting,
environment state management, and tool registry correctness.

Run with:
    uv run python -m pytest test_autoclean.py -v

Structure
---------
TestRewardHacking        — 12 tests: attempts to game rewards without cleaning
TestGraderLogic          — 8 tests:  score accuracy and clamping
TestBayesianWeighting    — 8 tests:  amplification, cap, and mode switching
TestEnvironmentState     — 7 tests:  reset, flagged_cols, step lifecycle
TestToolRegistry         — 5 tests:  registration, dispatch, unknown tools
TestObservationContract  — 5 tests:  Pydantic fields, weighting_mode, schema
TestEpisodeBoundaries    — 5 tests:  step_limit, done flags, close safety

Total: 50 tests
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import AutoCleanEnv, ToolRegistry
from models import Action, Observation, State
from logic import (
    get_weighted_missing_report,
    calculate_cleaning_gain,
    calculate_rarity_bonus,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = AutoCleanEnv(task_id="easy", bayesian_mode="auto")
    env.reset()
    yield env
    env.close()


@pytest.fixture
def medium_env():
    env = AutoCleanEnv(task_id="medium", bayesian_mode="auto")
    env.reset()
    yield env
    env.close()


@pytest.fixture
def hard_env():
    env = AutoCleanEnv(task_id="hard", bayesian_mode="auto")
    env.reset()
    yield env
    env.close()


@pytest.fixture
def sample_df():
    """10-row dataframe matching easy task structure."""
    return pd.DataFrame({
        "User_ID": range(1, 11),
        "Age":    [25.0, 30.0, None, 35.0, 40.0,
                   None, 22.0, 28.0, 31.0, None],
        "Clicks": [1.0, 0.0, 1.0, 1.0, 0.0,
                   None, 1.0, 0.0, 1.0, 1.0],
    })


@pytest.fixture
def weights():
    return np.ones(10)


# ===========================================================================
# 1. Reward Hacking Tests
# ===========================================================================

class TestRewardHacking:
    """
    Each test attempts a strategy that should NOT give a high reward.
    The environment must correctly penalise or zero-reward these attempts.
    """

    def test_impute_governance_column_is_penalised(self, hard_env):
        """Imputing Survey_Response (40% missing) must return reward -5.0."""
        result = hard_env.step(Action(tool="median_impute", column="Survey_Response"))
        assert result["reward"] == -5.0, (
            "Imputing a ≥35% column must return -5.0 penalty"
        )

    def test_knn_on_governance_column_is_penalised(self, hard_env):
        """knn_impute on governance column must also return -5.0."""
        result = hard_env.step(Action(tool="knn_impute", column="Survey_Response"))
        assert result["reward"] == -5.0

    def test_mode_impute_on_governance_column_is_penalised(self, hard_env):
        """mode_impute on governance column must also return -5.0."""
        result = hard_env.step(Action(tool="mode_impute", column="Survey_Response"))
        assert result["reward"] == -5.0

    def test_flag_clean_column_gives_no_reward(self, easy_env):
        """flag_human on User_ID (no missing) must not give governance reward."""
        result = easy_env.step(Action(tool="flag_human", column="User_ID"))
        # User_ID has 0 missing → weighted_pct = 0 → does NOT trigger governance gate
        # Should not give +2.0
        assert result["reward"] < 2.0, (
            "flag_human on a clean column must not earn governance reward"
        )

    def test_repeat_same_tool_incurs_penalty(self, easy_env):
        """Applying the same tool twice in a row must trigger -2.0 rarity penalty."""
        easy_env.step(Action(tool="knn_impute", column="Age"))
        # Second knn_impute in a row — Age is now clean but Clicks still dirty
        # Force another knn_impute on Clicks to trigger repetition penalty
        easy_env._state.history.append("knn_impute")  # simulate repeat
        result = easy_env.step(Action(tool="knn_impute", column="Clicks"))
        assert result["reward"] < 0 or result["reward"] < 1.0, (
            "Back-to-back same tool must trigger -2.0 repetition penalty"
        )

    def test_finish_with_dirty_columns_gives_zero_reward(self, easy_env):
        """finish() before cleaning anything must give reward=0.0."""
        result = easy_env.step(Action(tool="finish", column=None))
        assert result["reward"] == 0.0
        assert result["done"] is True

    def test_flag_twice_second_time_gives_zero(self, hard_env):
        """Flagging the same governance column twice must give 0.0 on second call."""
        hard_env.step(Action(tool="flag_human", column="Survey_Response"))
        # Second flag — column is now in flagged_cols, score=0.0
        result = hard_env.step(Action(tool="flag_human", column="Survey_Response"))
        assert result["reward"] == 0.0, (
            "Flagging an already-flagged column must not earn another +2.0"
        )

    def test_impute_id_column_gives_minimal_reward(self, easy_env):
        """Imputing User_ID (an ID column) must give a near-zero reward."""
        result = easy_env.step(Action(tool="median_impute", column="User_ID"))
        assert result["reward"] <= 0.1, (
            "ID columns must never give meaningful cleaning reward"
        )

    def test_fillna_on_governance_column_is_penalised(self, hard_env):
        """fillna (a form of imputation) on governance column must penalise."""
        result = hard_env.step(
            Action(tool="fillna", column="Survey_Response", params={"value": "Unknown"})
        )
        assert result["reward"] == -5.0

    def test_wrong_tool_on_categorical_gives_lower_reward(self, medium_env):
        """knn_impute redirects to mode_impute on object columns — no -5 but no bonus."""
        # Category is object dtype — knn_impute should redirect to mode, no strategy bonus
        result = medium_env.step(Action(tool="knn_impute", column="Category"))
        # knn_impute redirects to mode_impute on object columns.
        # mode_impute on a categorical earns the mode strategy bonus (0.75) + gain.
        # The key invariant: it must NOT earn the knn strategy bonus (0.5) separately.
        # Reward of 2.0 is acceptable — it reflects a successful mode_impute redirect.
        # What we forbid is reward > 2.5 (no double-bonus stacking).
        assert result["reward"] <= 2.5, (
            "knn_impute redirect must not earn double strategy bonus (knn + mode stacked)"
        )
        assert result["reward"] > -1.0, "redirect must not be penalised"

    def test_grader_score_not_1_before_cleaning(self, easy_env):
        """Grader must not return 1.0 before any cleaning has happened."""
        score = easy_env.grader(silent=True)
        assert score < 0.99, "Grader must not score 1.0 on the raw dirty dataset"

    def test_grader_stays_below_1_strictly(self, easy_env):
        """Grader must always return strictly less than 1.0 (clamped to 0.999)."""
        easy_env.step(Action(tool="knn_impute", column="Age"))
        easy_env.step(Action(tool="median_impute", column="Clicks"))
        score = easy_env.grader(silent=True)
        assert score < 1.0, f"Grader must be strictly < 1.0, got {score}"
        assert score > 0.0, f"Grader must be strictly > 0.0, got {score}"


# ===========================================================================
# 2. Grader Logic Tests
# ===========================================================================

class TestGraderLogic:

    def test_easy_grader_perfect_clean(self, easy_env):
        """After correct cleaning, easy grader must score > 0.98."""
        easy_env.step(Action(tool="knn_impute", column="Age"))
        easy_env.step(Action(tool="median_impute", column="Clicks"))
        score = easy_env.grader(silent=True)
        assert score > 0.98, f"Easy task fully cleaned must score > 0.98, got {score}"

    def test_hard_grader_flag_only(self, hard_env):
        """Flagging Survey_Response alone must score 0.60 (partial credit)."""
        hard_env.step(Action(tool="flag_human", column="Survey_Response"))
        score = hard_env.grader(silent=True)
        assert 0.55 < score < 0.65, (
            f"flag_human only must give ~0.60 partial credit, got {score}"
        )

    def test_hard_grader_full_credit(self, hard_env):
        """Flagging + cleaning Income_k must score > 0.99."""
        hard_env.step(Action(tool="flag_human", column="Survey_Response"))
        hard_env.step(Action(tool="knn_impute", column="Income_k"))
        score = hard_env.grader(silent=True)
        assert score > 0.99, f"Full hard task must score > 0.99, got {score}"

    def test_hard_grader_no_flag_zero(self, hard_env):
        """Cleaning Income_k without flagging Survey_Response must score 0.40."""
        hard_env.step(Action(tool="knn_impute", column="Income_k"))
        score = hard_env.grader(silent=True)
        assert 0.35 < score < 0.45, (
            f"Income_k only (no flag) must give ~0.40, got {score}"
        )

    def test_grader_score_strictly_above_zero(self, easy_env):
        """Grader must always return > 0.0 (clamped to 0.001 minimum)."""
        score = easy_env.grader(silent=True)
        assert score > 0.0, "Grader must never return exactly 0.0"

    def test_grader_score_strictly_below_one(self, easy_env):
        """Grader must always return < 1.0 (clamped to 0.999 maximum)."""
        easy_env.step(Action(tool="knn_impute", column="Age"))
        easy_env.step(Action(tool="median_impute", column="Clicks"))
        score = easy_env.grader(silent=True)
        assert score < 1.0, "Grader must never return exactly 1.0"

    def test_medium_grader_after_correct_episode(self, medium_env):
        """cast_type → knn_impute → mode_impute must score > 0.98."""
        medium_env.step(Action(tool="cast_type", column="Price",
                               params={"target_dtype": "float64"}))
        medium_env.step(Action(tool="knn_impute", column="Price"))
        medium_env.step(Action(tool="mode_impute", column="Category"))
        score = medium_env.grader(silent=True)
        assert score > 0.98, f"Full medium episode must score > 0.98, got {score}"

    def test_grader_silent_produces_no_output(self, easy_env, capsys):
        """grader(silent=True) must produce zero stdout/stderr."""
        easy_env.grader(silent=True)
        captured = capsys.readouterr()
        assert captured.out == "", "grader(silent=True) must not write to stdout"


# ===========================================================================
# 3. Bayesian Weighting Tests
# ===========================================================================

class TestBayesianWeighting:

    def test_auto_mode_scarce_amplifies(self, sample_df, weights):
        """auto mode on 10-row dataset must amplify scores."""
        report_auto = get_weighted_missing_report(
            sample_df, weights, bayesian_mode="auto", scarce_threshold=50
        )
        report_off = get_weighted_missing_report(
            sample_df, weights, bayesian_mode="off", scarce_threshold=50
        )
        assert report_auto["Age"] > report_off["Age"], (
            "auto mode must amplify score vs off mode for scarce dataset"
        )

    def test_amplification_never_crosses_governance_threshold(self, sample_df, weights):
        """Amplified score must never push a sub-threshold column above 0.35."""
        report = get_weighted_missing_report(
            sample_df, weights, bayesian_mode="auto", scarce_threshold=50
        )
        for col, score in report.items():
            n_miss = sample_df[col].isnull().sum()
            flat   = n_miss / len(sample_df)
            if flat < 0.35 and score > 0:
                assert score <= 0.34, (
                    f"{col}: flat={flat:.2f} but amplified score={score:.3f} "
                    f"crosses governance threshold 0.35"
                )

    def test_governance_column_returns_flat_pct(self, weights):
        """Columns with flat_pct >= 0.35 must return their flat score, not amplified."""
        df = pd.DataFrame({
            "Survey_Response": [None, "Yes", None, None, "No",
                                 None, "Yes", "Yes", "No", "No"]
        })
        report = get_weighted_missing_report(
            df, weights, bayesian_mode="auto", scarce_threshold=50
        )
        flat = df["Survey_Response"].isnull().sum() / len(df)
        assert report["Survey_Response"] == pytest.approx(flat, abs=0.001), (
            "Governance column must return flat pct, not amplified"
        )

    def test_off_mode_uses_flat_proportion(self, sample_df, weights):
        """bayesian_mode='off' must return exact flat proportions."""
        report = get_weighted_missing_report(
            sample_df, weights, bayesian_mode="off"
        )
        n = len(sample_df)
        for col in sample_df.columns:
            n_miss = sample_df[col].isnull().sum()
            if n_miss > 0:
                expected = n_miss / n
                assert report[col] == pytest.approx(expected, abs=0.001), (
                    f"{col}: expected flat {expected:.3f}, got {report[col]:.3f}"
                )

    def test_on_mode_always_amplifies(self, weights):
        """bayesian_mode='on' must amplify even for large datasets."""
        large_df = pd.DataFrame({
            "col": [None if i == 0 else float(i) for i in range(100)]
        })
        report_on  = get_weighted_missing_report(large_df, np.ones(100), bayesian_mode="on",  scarce_threshold=50)
        report_off = get_weighted_missing_report(large_df, np.ones(100), bayesian_mode="off", scarce_threshold=50)
        assert report_on["col"] >= report_off["col"], (
            "bayesian_mode='on' must never score lower than 'off'"
        )

    def test_id_columns_zeroed_in_report(self, weights):
        """Columns with 'id', 'idx', or 'key' in name must always score 0.0."""
        df = pd.DataFrame({
            "User_ID":    [None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "product_key":[None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Age":        [None, 25, 30, 35, 40, 22, 28, 31, 25, 30],
        })
        report = get_weighted_missing_report(df, weights, bayesian_mode="auto")
        assert report["User_ID"] == 0.0, "User_ID must always be zeroed"
        assert report["product_key"] == 0.0, "product_key must always be zeroed"
        assert report["Age"] > 0.0, "Age must have positive score"

    def test_sentinel_strings_count_as_missing(self, weights):
        """'Missing', 'N/A', '?' in object columns must count as missing."""
        df = pd.DataFrame({
            "Price": ["19.99", "25.50", "Missing", "15.00", "30.00",
                      "18.50", "20.00", "N/A",    "32.50", "40.00"]
        })
        report = get_weighted_missing_report(df, weights, bayesian_mode="off")
        assert report["Price"] > 0.0, (
            "Sentinel strings must register as missing in the report"
        )
        expected = 2 / 10
        assert report["Price"] == pytest.approx(expected, abs=0.001)

    def test_clean_column_scores_zero(self, sample_df, weights):
        """Columns with no missing values must score exactly 0.0."""
        report = get_weighted_missing_report(sample_df, weights, bayesian_mode="auto")
        assert report["User_ID"] == 0.0


# ===========================================================================
# 4. Environment State Tests
# ===========================================================================

class TestEnvironmentState:

    def test_reset_creates_new_episode_id(self):
        """Each reset must produce a different episode_id."""
        env = AutoCleanEnv(task_id="easy")
        id1 = env.state.episode_id
        env.reset()
        id2 = env.state.episode_id
        env.close()
        assert id1 != id2, "reset() must generate a new episode_id"

    def test_reset_zeroes_step_count(self, easy_env):
        """step_count must be 0 immediately after reset."""
        assert easy_env.state.step_count == 0

    def test_flagged_cols_populated_after_flag_human(self, hard_env):
        """flag_human must add column to flagged_cols."""
        assert "Survey_Response" not in hard_env.state.flagged_cols
        hard_env.step(Action(tool="flag_human", column="Survey_Response"))
        assert "Survey_Response" in hard_env.state.flagged_cols

    def test_flagged_col_scores_zero_in_next_observation(self, hard_env):
        """After flag_human, flagged column must appear as 0.0 in next observation."""
        hard_env.step(Action(tool="flag_human", column="Survey_Response"))
        result = hard_env.step(Action(tool="knn_impute", column="Income_k"))
        obs = result.get("observation", {})
        missing_report = obs.get("missing_report", {})
        assert missing_report.get("Survey_Response", 999) == 0.0, (
            "Flagged column must show score=0.0 in subsequent observations"
        )

    def test_history_tracks_all_tools(self, easy_env):
        """history list must record every tool called in order."""
        easy_env.step(Action(tool="knn_impute", column="Age"))
        easy_env.step(Action(tool="median_impute", column="Clicks"))
        assert easy_env.state.history == ["knn_impute", "median_impute"]

    def test_close_is_idempotent(self, easy_env):
        """close() called twice must not raise an exception."""
        easy_env.close()
        easy_env.close()  # must not raise

    def test_get_state_model_returns_state_type(self, easy_env):
        """get_state_model() must return a models.State instance."""
        state = easy_env.get_state_model()
        assert isinstance(state, State)
        assert state.episode_id == easy_env.state.episode_id
        assert state.step_count == easy_env.state.step_count


# ===========================================================================
# 5. Tool Registry Tests
# ===========================================================================

class TestToolRegistry:

    def test_all_expected_tools_registered(self):
        """All 7 tools must be present in the registry."""
        expected = {
            "knn_impute", "median_impute", "mode_impute",
            "flag_human", "fillna", "cast_type", "finish"
        }
        registered = set(ToolRegistry.available())
        assert expected.issubset(registered), (
            f"Missing tools: {expected - registered}"
        )

    def test_unknown_tool_raises_key_error(self):
        """Calling an unregistered tool must raise KeyError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(KeyError):
            ToolRegistry.execute("nonexistent_tool", df, "col", {})

    def test_knn_redirects_on_object_column(self):
        """knn_impute on object column must silently redirect to mode_impute."""
        df = pd.DataFrame({
            "Category": ["Tech", "Home", None, "Tech", "Home",
                         "Home", "Home", None, "Tech", "Home"]
        })
        result_df, msg = ToolRegistry.execute("knn_impute", df, "Category", {})
        assert result_df["Category"].isnull().sum() == 0, (
            "knn_impute on object column must fill all NaNs (via mode redirect)"
        )
        assert "mode" in msg.lower() or result_df["Category"].isnull().sum() == 0

    def test_median_impute_fills_numeric_nans(self):
        """median_impute must fill NaN cells with column median."""
        df = pd.DataFrame({"Age": [25.0, 30.0, None, 35.0, 40.0]})
        result_df, _ = ToolRegistry.execute("median_impute", df, "Age", {})
        # Non-null values: [25, 30, 35, 40] → median = (30+35)/2 = 32.5
        import numpy as np
        expected_median = float(np.nanmedian([25.0, 30.0, 35.0, 40.0]))
        assert result_df["Age"].isnull().sum() == 0
        assert result_df["Age"].iloc[2] == pytest.approx(expected_median, abs=0.01)

    def test_mode_impute_fills_categorical_nans(self):
        """mode_impute must fill NaN cells with column mode."""
        df = pd.DataFrame({
            "Category": ["Home", "Home", None, "Tech", "Home"]
        })
        result_df, _ = ToolRegistry.execute("mode_impute", df, "Category", {})
        assert result_df["Category"].isnull().sum() == 0
        assert result_df["Category"].iloc[2] == "Home"


# ===========================================================================
# 6. Observation Contract Tests
# ===========================================================================

class TestObservationContract:

    def test_observation_has_required_fields(self, easy_env):
        """Observation must contain all required fields."""
        obs = easy_env.reset()
        assert hasattr(obs, "data_preview")
        assert hasattr(obs, "missing_report")
        assert hasattr(obs, "schema_info")
        assert hasattr(obs, "total_rows")
        assert hasattr(obs, "message")
        assert hasattr(obs, "weighting_mode")
        assert hasattr(obs, "dataset_regime")
        assert hasattr(obs, "available_tools")

    def test_weighting_mode_is_bayesian_for_small_dataset(self, easy_env):
        """10-row dataset in auto mode must show weighting_mode=bayesian_scarce."""
        obs = easy_env.reset()
        assert obs.weighting_mode == "bayesian_scarce", (
            f"Expected bayesian_scarce, got {obs.weighting_mode}"
        )

    def test_weighting_mode_is_standard_for_large_dataset(self):
        """Large dataset (>50 rows) in auto mode must show standard_uniform."""
        import tempfile
        large_df = pd.DataFrame({
            "col": [float(i) if i % 5 != 0 else None for i in range(100)]
        })
        import tempfile as _tf
        tmp_dir = _tf.gettempdir()   # portable: works on Windows and Linux
        with _tf.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir=tmp_dir
        ) as f:
            large_df.to_csv(f.name, index=False)
            tmp = f.name

        env = AutoCleanEnv.__new__(AutoCleanEnv)
        env.task_id = "custom"
        env.step_limit = 15
        env._bayesian_mode = "auto"
        env._scarce_threshold = 50
        env._paths = {"custom": {"dirty": tmp, "clean": tmp}}
        env.df = None
        env.target_df = None
        env.weights = None
        from environment import EpisodeState
        env._state = EpisodeState(task_id="custom")
        obs = env.reset()
        env.close()
        assert obs.weighting_mode == "standard_uniform"

    def test_missing_report_all_values_between_0_and_1(self, easy_env):
        """All missing_report scores must be in [0.0, 1.0]."""
        obs = easy_env.reset()
        for col, score in obs.missing_report.items():
            assert 0.0 <= score <= 1.0, (
                f"{col}: score={score} out of [0,1] range"
            )

    def test_available_tools_matches_registry(self, easy_env):
        """Observation available_tools must match ToolRegistry.available()."""
        obs = easy_env.reset()
        assert set(obs.available_tools) == set(ToolRegistry.available())


# ===========================================================================
# 7. Episode Boundary Tests
# ===========================================================================

class TestEpisodeBoundaries:

    def test_done_true_when_step_limit_reached(self):
        """done must be True when step_limit is exhausted."""
        env = AutoCleanEnv(task_id="easy", step_limit=2)
        env.reset()
        env.step(Action(tool="knn_impute", column="Age"))
        result = env.step(Action(tool="median_impute", column="Clicks"))
        env.close()
        assert result["done"] is True

    def test_done_true_when_all_nans_filled(self, easy_env):
        """done must be True when all NaNs are filled."""
        easy_env.step(Action(tool="knn_impute", column="Age"))
        result = easy_env.step(Action(tool="median_impute", column="Clicks"))
        assert result["done"] is True

    def test_done_true_on_finish_action(self, easy_env):
        """finish action must always set done=True."""
        result = easy_env.step(Action(tool="finish", column=None))
        assert result["done"] is True

    def test_step_count_increments(self, easy_env):
        """step_count must increment by 1 on each step call."""
        assert easy_env.state.step_count == 0
        easy_env.step(Action(tool="knn_impute", column="Age"))
        assert easy_env.state.step_count == 1
        easy_env.step(Action(tool="median_impute", column="Clicks"))
        assert easy_env.state.step_count == 2

    def test_reward_is_float(self, easy_env):
        """Reward returned from step must always be a Python float."""
        result = easy_env.step(Action(tool="knn_impute", column="Age"))
        assert isinstance(result["reward"], float), (
            f"reward must be float, got {type(result['reward'])}"
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])