# ============================================================
# PYTEST UNIT TESTS — Credit Card Fraud Detection ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import Clipper, build_preprocessors
from src.leakage_check import detect_leakage
from src.metrics       import tune_threshold, psi, recall_at_k, lift_at_k, ks_statistic
from src.rule_engine   import rule_engine
from src.config        import (
    HIGH_AMOUNT_RULE, REVIEW_PROB_RATIO,
    PSI_MODERATE, PSI_HIGH,
    MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP
)


# ============================================================
# CLIPPER TESTS
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        """Output shape must match input shape."""
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        """Extreme values must be clipped."""
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        """Values within IQR range should not be changed."""
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input_handled(self):
        """1-D array must be handled without error."""
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        """get_feature_names_out must return array of correct length."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        """
        Clipper fitted on train must use training bounds on test —
        not re-fit on test data (leakage prevention).
        """
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test  = np.array([[100.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0, "Test outlier should be clipped to training bounds"


# ============================================================
# PREPROCESSOR TESTS
# ============================================================

class TestBuildPreprocessors:

    def _sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "V1":         np.random.normal(0, 1, 50),
            "V2":         np.random.normal(0, 1, 50),
            "log_amount": np.random.exponential(2, 50),
            "hour_sin":   np.sin(np.linspace(0, 2 * np.pi, 50)),
            "hour_cos":   np.cos(np.linspace(0, 2 * np.pi, 50)),
        })

    def test_returns_three_outputs(self):
        """build_preprocessors must return (pre_scaled, pre_unscaled, cont_cols)."""
        df = self._sample_df()
        result = build_preprocessors(df)
        assert len(result) == 3

    def test_cont_cols_covers_all_input(self):
        """cont_cols must contain all input columns."""
        df = self._sample_df()
        _, _, cont_cols = build_preprocessors(df)
        for col in df.columns:
            assert col in cont_cols

    def test_scaled_preprocessor_transforms(self):
        """Scaled preprocessor must produce output without NaN."""
        df = self._sample_df()
        pre_scaled, _, _ = build_preprocessors(df)
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_unscaled_preprocessor_transforms(self):
        """Unscaled preprocessor must produce output without NaN."""
        df = self._sample_df()
        _, pre_unscaled, _ = build_preprocessors(df)
        out = pre_unscaled.fit_transform(df)
        assert not np.isnan(out).any()


# ============================================================
# LEAKAGE CHECK TESTS
# ============================================================

class TestDetectLeakage:

    def test_catches_identical_feature(self):
        """Feature identical to target must be flagged."""
        X = pd.DataFrame({"a": [1, 0, 1, 0, 1]})
        y = pd.Series(        [1, 0, 1, 0, 1])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0
        assert any("a" in w for w in warnings)

    def test_no_false_positives_on_clean_data(self):
        """Normal random features must not be flagged."""
        np.random.seed(42)
        X = pd.DataFrame({"income": np.random.uniform(20, 200, 100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_high_correlation(self):
        """Near-perfect correlation must be flagged."""
        vals = np.arange(50, dtype=float)
        X    = pd.DataFrame({"leaky": vals})
        y    = pd.Series((vals > 25).astype(int))
        warnings = detect_leakage(X, y, threshold_corr=0.85)
        assert len(warnings) > 0

    def test_empty_dataframe_no_crash(self):
        """Empty feature set must not raise an exception."""
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)


# ============================================================
# METRICS TESTS
# ============================================================

class TestTuneThreshold:

    def test_returns_float_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)
        thr = tune_threshold(y_true, y_prob)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_perfect_separation_low_threshold(self):
        """Perfect predictions → threshold ≤ 0.9."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        thr = tune_threshold(y_true, y_prob)
        assert thr <= 0.9


class TestPSI:

    def test_identical_distributions_near_zero(self):
        """PSI of identical arrays must be near 0."""
        x = np.random.normal(0, 1, 500)
        assert psi(x, x) < 0.05

    def test_shifted_distribution_higher_psi(self):
        """Shifted distribution must produce higher PSI than identical."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)
        assert psi(ref, new) > psi(ref, ref)

    def test_uses_reference_edges_not_actual(self):
        """
        PSI must bin BOTH arrays using edges from reference only.
        Old bug: ranking both independently → PSI always ~0.
        """
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 500)
        new = rng.normal(5, 1, 500)   # completely disjoint
        score = psi(ref, new)
        assert score > 0.20, (
            f"Completely shifted distribution should have PSI >> 0.20, got {score:.4f}"
        )


class TestRecallAtK:

    def test_top_scores_captured(self):
        """Top-k by score must contain most fraud cases."""
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        r = recall_at_k(y, p, k=0.3)
        assert r >= 0.66  # top 3 of 10 should capture at least 2 of 3 fraud

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(recall_at_k(y, p), float)


class TestLiftAtK:

    def test_lift_above_one_for_good_model(self):
        """Good model must have lift > 1."""
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        assert lift_at_k(y, p, k=0.3) > 1.0

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(lift_at_k(y, p), float)


class TestKSStatistic:

    def test_returns_float_in_range(self):
        np.random.seed(1)
        y = np.random.randint(0, 2, 100)
        p = np.random.uniform(0, 1, 100)
        ks = ks_statistic(y, p)
        assert isinstance(ks, float)
        assert 0.0 <= ks <= 1.0

    def test_perfect_model_high_ks(self):
        """Perfect separation → KS close to 1."""
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        assert ks_statistic(y, p) > 0.7


# ============================================================
# RULE ENGINE TESTS
# ============================================================

class TestRuleEngine:

    def _make_df(self, amount_original):
        return pd.DataFrame({
            "Amount_original": [amount_original],
            "Amount":          [amount_original],
        })

    def test_high_amount_triggers_block_rule(self):
        """Amount above HIGH_AMOUNT_RULE threshold must trigger BLOCK_RULE."""
        df     = self._make_df(HIGH_AMOUNT_RULE + 1)
        result = rule_engine(df, probs=[0.05], threshold=0.5)
        assert result[0] == "BLOCK_RULE"

    def test_high_prob_triggers_block_model(self):
        """High ML probability must trigger BLOCK_MODEL."""
        df     = self._make_df(100.0)
        result = rule_engine(df, probs=[0.95], threshold=0.5)
        assert result[0] == "BLOCK_MODEL"

    def test_borderline_prob_triggers_review(self):
        """Borderline probability must trigger REVIEW."""
        df     = self._make_df(100.0)
        thr    = 0.5
        result = rule_engine(df, probs=[thr * REVIEW_PROB_RATIO + 0.01], threshold=thr)
        assert result[0] == "REVIEW"

    def test_low_prob_low_amount_approves(self):
        """Low probability + normal amount must APPROVE."""
        df     = self._make_df(50.0)
        result = rule_engine(df, probs=[0.02], threshold=0.5)
        assert result[0] == "APPROVE"

    def test_output_length_matches_input(self):
        """Output list length must equal number of input rows."""
        df     = pd.DataFrame({
            "Amount_original": [100, 200, 6000],
            "Amount":          [100, 200, 6000],
        })
        result = rule_engine(df, probs=[0.05, 0.9, 0.05], threshold=0.5)
        assert len(result) == 3

    def test_all_valid_decision_labels(self):
        """All decisions must be one of the 4 valid labels."""
        valid  = {"APPROVE", "REVIEW", "BLOCK_MODEL", "BLOCK_RULE"}
        df     = pd.DataFrame({
            "Amount_original": [50, 200, 6000, 10],
            "Amount":          [50, 200, 6000, 10],
        })
        result = rule_engine(df, probs=[0.02, 0.9, 0.05, 0.35], threshold=0.5)
        for r in result:
            assert r in valid


# ============================================================
# CONFIG TESTS
# ============================================================

class TestConfig:

    def test_psi_thresholds_ordered(self):
        """Moderate PSI threshold must be less than High."""
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_reasonable(self):
        """Promotion thresholds must be in valid ranges."""
        assert 0.0 < MIN_F1_IMPROVEMENT < 0.1
        assert 0.5 < MIN_ROCAUC_THRESHOLD < 1.0
        assert 0.0 < MAX_GENERALIZATION_GAP < 1.0

    def test_review_prob_ratio_valid(self):
        """Review threshold ratio must be between 0 and 1."""
        assert 0.0 < REVIEW_PROB_RATIO < 1.0

    def test_high_amount_rule_positive(self):
        """HIGH_AMOUNT_RULE must be a positive number."""
        assert HIGH_AMOUNT_RULE > 0