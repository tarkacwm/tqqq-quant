#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    date_col: str = "Date"
    tqqq_col: str = "TQQQ Close"
    qqq_col: Optional[str] = "QQQ Close"
    soxx_col: Optional[str] = "SOXX Close"
    spy_col: Optional[str] = "SPY Close"
    vix_col: Optional[str] = "VIX"
    min_state_count: int = 15
    mc_sims: int = 3000
    mc_block_size: int = 20

    # Optimized binary-gate parameters
    vol_ratio_threshold: float = 1.45
    drawdown_flag_threshold: float = -0.12
    extension_threshold: float = 2.25
    compression_rank_threshold: float = 0.30
    range_rank_threshold: float = 0.55
    entry_toxic_max: int = 1
    exit_toxic_min: int = 4
    hard_crash_drawdown_threshold: float = -0.15
    hard_crash_ret3_threshold: float = -0.10
    min_hold_days: int = 3
    constructive_score_min: int = 3


class TQQQPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = [self.cfg.date_col, self.cfg.tqqq_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        keep_cols = [self.cfg.date_col, self.cfg.tqqq_col]
        for col in [self.cfg.qqq_col, self.cfg.soxx_col, self.cfg.spy_col, self.cfg.vix_col]:
            if col and col in df.columns:
                keep_cols.append(col)
        df = df[keep_cols].copy()

        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
        df = df.dropna(subset=[self.cfg.date_col]).copy()

        rename = {self.cfg.date_col: "date", self.cfg.tqqq_col: "close"}
        if self.cfg.qqq_col and self.cfg.qqq_col in df.columns:
            rename[self.cfg.qqq_col] = "qqq_close"
        if self.cfg.soxx_col and self.cfg.soxx_col in df.columns:
            rename[self.cfg.soxx_col] = "soxx_close"
        if self.cfg.spy_col and self.cfg.spy_col in df.columns:
            rename[self.cfg.spy_col] = "spy_close"
        if self.cfg.vix_col and self.cfg.vix_col in df.columns:
            rename[self.cfg.vix_col] = "vix"
        df = df.rename(columns=rename)

        numeric_cols = [c for c in ["close", "qqq_close", "soxx_close", "spy_close", "vix"] if c in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        min_periods = min_periods or max(20, window // 3)
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _pct_rank(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        min_periods = min_periods or max(20, window // 3)

        def rank_last(x: np.ndarray) -> float:
            s = pd.Series(x)
            return float(s.rank(pct=True).iloc[-1])

        return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        px = out["close"]

        # Returns and labels
        out["ret_1d"] = np.log(px / px.shift(1))
        out["ret_3d"] = np.log(px / px.shift(3))
        out["ret_5d"] = np.log(px / px.shift(5))
        out["ret_10d"] = np.log(px / px.shift(10))
        out["fwd_1d"] = np.log(px.shift(-1) / px)
        out["fwd_3d"] = np.log(px.shift(-3) / px)
        out["fwd_5d"] = np.log(px.shift(-5) / px)
        out["target_alpha"] = 0.20 * out["fwd_1d"] + 0.50 * out["fwd_3d"] + 0.30 * out["fwd_5d"]

        # EMA structure
        out["ema_8"] = px.ewm(span=8, adjust=False, min_periods=8).mean()
        out["ema_21"] = px.ewm(span=21, adjust=False, min_periods=21).mean()
        out["ema_63"] = px.ewm(span=63, adjust=False, min_periods=63).mean()
        out["dist_ema8"] = px / out["ema_8"] - 1.0
        out["dist_ema21"] = px / out["ema_21"] - 1.0
        out["dist_ema63"] = px / out["ema_63"] - 1.0
        out["ema8_slope"] = out["ema_8"].pct_change()
        out["ema21_slope"] = out["ema_21"].pct_change()
        out["ema63_slope"] = out["ema_63"].pct_change()

        # Volatility and drawdown
        out["vol_5d"] = out["ret_1d"].rolling(5, min_periods=5).std(ddof=0)
        out["vol_20d"] = out["ret_1d"].rolling(20, min_periods=20).std(ddof=0)
        out["vol_ratio_5_20"] = out["vol_5d"] / out["vol_20d"]
        out["rolling_peak_20"] = px.rolling(20, min_periods=20).max()
        out["drawdown_20"] = px / out["rolling_peak_20"] - 1.0

        # Cross-asset context
        if "qqq_close" in out.columns:
            out["qqq_ret_3d"] = np.log(out["qqq_close"] / out["qqq_close"].shift(3))
        else:
            out["qqq_ret_3d"] = np.nan
        if "soxx_close" in out.columns:
            out["soxx_ret_3d"] = np.log(out["soxx_close"] / out["soxx_close"].shift(3))
        else:
            out["soxx_ret_3d"] = np.nan
        if "spy_close" in out.columns:
            out["spy_ret_3d"] = np.log(out["spy_close"] / out["spy_close"].shift(3))
        else:
            out["spy_ret_3d"] = np.nan
        out["soxx_vs_qqq_3d"] = out["soxx_ret_3d"] - out["qqq_ret_3d"]
        out["qqq_vs_spy_3d"] = out["qqq_ret_3d"] - out["spy_ret_3d"]

        # Overextension and compression-break diagnostics
        out["z_dist_ema21_126"] = self._rolling_zscore(out["dist_ema21"], 126, min_periods=40)
        out["z_ret_5d_126"] = self._rolling_zscore(out["ret_5d"], 126, min_periods=40)
        out["z_ret_10d_126"] = self._rolling_zscore(out["ret_10d"], 126, min_periods=40)
        out["extension_score"] = out["z_dist_ema21_126"] + 0.75 * out["z_ret_5d_126"] + 0.50 * out["z_ret_10d_126"]
        out["compression_rank_126"] = self._pct_rank(out["vol_ratio_5_20"], 126, min_periods=40)
        out["range_proxy"] = out["ret_1d"].abs()
        out["range_rank_126"] = self._pct_rank(out["range_proxy"], 126, min_periods=40)
        out["compression_break_signal"] = (
            (out["extension_score"] > self.cfg.extension_threshold)
            & (out["compression_rank_126"] < self.cfg.compression_rank_threshold)
            & (out["range_rank_126"] < self.cfg.range_rank_threshold)
            & (out["close"] < out["ema_8"])
        ).astype(int)

        # Research buckets
        out["trend_bucket"] = np.select(
            [out["dist_ema21"] > 0.0075, out["dist_ema21"] < -0.015],
            ["UP", "DOWN"],
            default="MID",
        )
        out["vol_bucket"] = np.select(
            [out["vol_ratio_5_20"] > self.cfg.vol_ratio_threshold, out["vol_ratio_5_20"] < 0.90],
            ["VOL_EXPAND", "VOL_COMP"],
            default="VOL_MID",
        )
        out["leadership_bucket"] = np.select(
            [out["soxx_vs_qqq_3d"] > -0.01, out["soxx_vs_qqq_3d"] < -0.02],
            ["SEMI_OK", "SEMI_LAG"],
            default="SEMI_MID",
        )
        out["shock_bucket"] = np.select(
            [out["ret_3d"] < -0.08, out["ret_3d"] > 0.05],
            ["SHOCK_DN", "SHOCK_UP"],
            default="NEUTRAL",
        )
        out["state_key"] = (
            out["trend_bucket"]
            + "|" + out["shock_bucket"]
            + "|" + out["vol_bucket"]
            + "|" + out["leadership_bucket"]
        )

        # Diagnostics
        out["regime_class"] = out.apply(self._classify_regime, axis=1)
        out["continuation_score"] = out.apply(self._continuation_score, axis=1)
        out["pullback_score"] = out.apply(self._pullback_score, axis=1)
        out["compression_score"] = out.apply(self._compression_score, axis=1)
        out["alpha_score"] = 0.55 * out["continuation_score"] + 0.20 * out["pullback_score"] + 0.25 * out["compression_score"]
        return out

    def _classify_regime(self, row: pd.Series) -> str:
        toxic_count = self.compute_toxic_flags(row)
        if toxic_count >= 2:
            return "TOXIC"

        weak_count = 0
        if pd.notna(row.get("ema_21")) and row["close"] < row["ema_21"]:
            weak_count += 1
        if pd.notna(row.get("ema8_slope")) and row["ema8_slope"] < 0:
            weak_count += 1
        if pd.notna(row.get("qqq_vs_spy_3d")) and row["qqq_vs_spy_3d"] < 0:
            weak_count += 1
        if weak_count >= 1:
            return "WEAK"

        strong_count = 0
        if pd.notna(row.get("ema_21")) and row["close"] > row["ema_21"]:
            strong_count += 1
        if pd.notna(row.get("ema21_slope")) and row["ema21_slope"] > 0:
            strong_count += 1
        if pd.notna(row.get("vol_ratio_5_20")) and row["vol_ratio_5_20"] < 1.20:
            strong_count += 1
        if pd.notna(row.get("soxx_vs_qqq_3d")) and row["soxx_vs_qqq_3d"] > -0.01:
            strong_count += 1
        if strong_count >= 4:
            return "STRONG"
        return "GOOD"

    @staticmethod
    def _continuation_score(row: pd.Series) -> float:
        score = 0
        if pd.notna(row.get("ema_21")) and row["close"] > row["ema_21"]:
            score += 1
        if pd.notna(row.get("ema8_slope")) and row["ema8_slope"] > 0:
            score += 1
        if pd.notna(row.get("ret_3d")) and row["ret_3d"] > 0:
            score += 1
        if pd.notna(row.get("soxx_vs_qqq_3d")) and row["soxx_vs_qqq_3d"] > -0.01:
            score += 1
        return score / 4.0

    @staticmethod
    def _pullback_score(row: pd.Series) -> float:
        score = 0
        if pd.notna(row.get("ema_21")) and row["close"] > row["ema_21"]:
            score += 1
        if pd.notna(row.get("dist_ema8")) and (-0.03 < row["dist_ema8"] < 0):
            score += 1
        if pd.notna(row.get("ema21_slope")) and row["ema21_slope"] > 0:
            score += 1
        if pd.notna(row.get("vol_ratio_5_20")) and row["vol_ratio_5_20"] < 1.35:
            score += 1
        return score / 4.0

    @staticmethod
    def _compression_score(row: pd.Series) -> float:
        score = 0
        if pd.notna(row.get("dist_ema8")) and abs(row["dist_ema8"]) < 0.01:
            score += 1
        if pd.notna(row.get("dist_ema21")) and abs(row["dist_ema21"]) < 0.015:
            score += 1
        if pd.notna(row.get("vol_ratio_5_20")) and row["vol_ratio_5_20"] < 0.90:
            score += 1
        if pd.notna(row.get("ema21_slope")) and row["ema21_slope"] > 0:
            score += 1
        return score / 4.0

    def compute_toxic_flags(self, row: pd.Series) -> int:
        flags = 0
        if pd.notna(row.get("ema_63")) and row["close"] < row["ema_63"]:
            flags += 1
        if pd.notna(row.get("ema21_slope")) and row["ema21_slope"] < 0:
            flags += 1
        if pd.notna(row.get("vol_ratio_5_20")) and row["vol_ratio_5_20"] > self.cfg.vol_ratio_threshold:
            flags += 1
        if pd.notna(row.get("soxx_vs_qqq_3d")) and row["soxx_vs_qqq_3d"] < -0.025:
            flags += 1
        if pd.notna(row.get("drawdown_20")) and row["drawdown_20"] < self.cfg.drawdown_flag_threshold:
            flags += 1
        return flags

    def compute_constructive_score(self, row: pd.Series) -> int:
        score = 0
        if pd.notna(row.get("ema_21")) and row["close"] > row["ema_21"]:
            score += 1
        if pd.notna(row.get("ema21_slope")) and row["ema21_slope"] > 0:
            score += 1
        if pd.notna(row.get("ret_5d")) and row["ret_5d"] > 0:
            score += 1
        if pd.notna(row.get("soxx_vs_qqq_3d")) and row["soxx_vs_qqq_3d"] > -0.01:
            score += 1
        return score

    def compute_hard_crash_trigger(self, row: pd.Series) -> bool:
        if pd.isna(row.get("drawdown_20")) or pd.isna(row.get("ret_3d")):
            return False
        return bool(
            (row["drawdown_20"] < self.cfg.hard_crash_drawdown_threshold)
            and (row["ret_3d"] < self.cfg.hard_crash_ret3_threshold)
        )

    def compute_meltup_failure(self, row: pd.Series) -> bool:
        needed = ["extension_score", "compression_break_signal", "close", "ema_8"]
        if any(pd.isna(row.get(col)) for col in needed):
            return False
        return bool(
            (row["extension_score"] > self.cfg.extension_threshold)
            and (row["compression_break_signal"] == 1)
            and (row["close"] < row["ema_8"])
        )

    def build_research_table(self, features: pd.DataFrame) -> pd.DataFrame:
        grp = features.dropna(subset=["state_key"]).groupby("state_key", dropna=False)
        research = grp.agg(
            count=("state_key", "size"),
            avg_fwd_1d=("fwd_1d", "mean"),
            avg_fwd_3d=("fwd_3d", "mean"),
            avg_fwd_5d=("fwd_5d", "mean"),
            hit_1d=("fwd_1d", lambda x: float((x > 0).mean())),
            hit_3d=("fwd_3d", lambda x: float((x > 0).mean())),
            hit_5d=("fwd_5d", lambda x: float((x > 0).mean())),
            avg_target_alpha=("target_alpha", "mean"),
        ).reset_index()
        research["alpha_score_hist"] = 0.15 * research["avg_fwd_1d"] + 0.45 * research["avg_fwd_3d"] + 0.40 * research["avg_fwd_5d"]
        research["invest_flag"] = np.where(
            (research["count"] >= self.cfg.min_state_count) &
            (
                ((research["avg_fwd_3d"] > 0.02) & (research["avg_fwd_5d"] > 0.03) & (research["hit_3d"] > 0.58)) |
                ((research["avg_fwd_3d"] > 0.005) & (research["hit_3d"] >= 0.50))
            ),
            1,
            0,
        )
        research = research.sort_values(["alpha_score_hist", "avg_fwd_3d"], ascending=False).reset_index(drop=True)
        return research

    def _state_history_lookup(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = ["state_key", "fwd_1d", "fwd_3d", "fwd_5d"]
        tmp = features[cols].copy().reset_index(drop=True)
        stats: dict[str, dict[str, float]] = {}
        rows: list[dict[str, float | int]] = []

        for _, row in tmp.iterrows():
            key = row["state_key"]
            s = stats.get(key)
            if s is None:
                rows.append({
                    "hist_count": 0,
                    "hist_avg_fwd_1d": np.nan,
                    "hist_avg_fwd_3d": np.nan,
                    "hist_avg_fwd_5d": np.nan,
                    "hist_hit_3d": np.nan,
                    "hist_invest_flag": 0,
                })
            else:
                avg1 = s["sum1"] / s["count"]
                avg3 = s["sum3"] / s["count"]
                avg5 = s["sum5"] / s["count"]
                hit3 = s["hit3"] / s["count"]
                invest_flag = int(
                    (s["count"] >= self.cfg.min_state_count) and
                    (((avg3 > 0.02) and (avg5 > 0.03) and (hit3 > 0.58)) or ((avg3 > 0.005) and (hit3 >= 0.50)))
                )
                rows.append({
                    "hist_count": s["count"],
                    "hist_avg_fwd_1d": avg1,
                    "hist_avg_fwd_3d": avg3,
                    "hist_avg_fwd_5d": avg5,
                    "hist_hit_3d": hit3,
                    "hist_invest_flag": invest_flag,
                })

            if pd.notna(key) and pd.notna(row["fwd_3d"]):
                cur = stats.setdefault(key, {"count": 0, "sum1": 0.0, "sum3": 0.0, "sum5": 0.0, "hit3": 0.0})
                cur["count"] += 1
                cur["sum1"] += float(row["fwd_1d"])
                cur["sum3"] += float(row["fwd_3d"])
                cur["sum5"] += float(row["fwd_5d"])
                cur["hit3"] += float(row["fwd_3d"] > 0)

        hist = pd.DataFrame(rows)
        return pd.concat([features.reset_index(drop=True), hist], axis=1)

    def run_backtest(self, features: pd.DataFrame) -> pd.DataFrame:
        df = self._state_history_lookup(features).copy()
        df["toxic_flags"] = df.apply(self.compute_toxic_flags, axis=1)
        df["constructive_score"] = df.apply(self.compute_constructive_score, axis=1)
        df["hard_crash_trigger"] = df.apply(self.compute_hard_crash_trigger, axis=1)
        df["meltup_failure"] = df.apply(self.compute_meltup_failure, axis=1)

        target_position = []
        in_position = 0
        days_in_trade = 0

        for _, row in df.iterrows():
            if pd.isna(row.get("ema_21")) or pd.isna(row.get("ema_63")):
                target_position.append(0.0)
                in_position = 0
                days_in_trade = 0
                continue

            investable_state = int(row.get("hist_invest_flag", 0)) == 1
            can_enter = (
                (not bool(row["hard_crash_trigger"]))
                and (row["toxic_flags"] <= self.cfg.entry_toxic_max)
                and (row["constructive_score"] >= self.cfg.constructive_score_min)
                and investable_state
            )

            must_exit = (
                bool(row["hard_crash_trigger"])
                or (row["toxic_flags"] >= self.cfg.exit_toxic_min)
                or bool(row["meltup_failure"])
            )

            if in_position == 0:
                if can_enter:
                    in_position = 1
                    days_in_trade = 1
                else:
                    in_position = 0
                    days_in_trade = 0
            else:
                days_in_trade += 1
                if must_exit and days_in_trade >= self.cfg.min_hold_days:
                    in_position = 0
                    days_in_trade = 0

            target_position.append(float(in_position))

        df["target_position"] = target_position
        df["position"] = df["target_position"].shift(1).fillna(0.0)
        df["strategy_ret_1d"] = df["position"] * df["ret_1d"]
        df["buyhold_equity"] = np.exp(df["ret_1d"].fillna(0.0).cumsum())
        df["strategy_equity"] = np.exp(df["strategy_ret_1d"].fillna(0.0).cumsum())
        df["live_exposure_class"] = np.where(df["target_position"] > 0, "INVESTED", "CASH")
        return df

    @staticmethod
    def max_drawdown_from_returns(returns: np.ndarray) -> float:
        equity = np.exp(np.cumsum(returns))
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        return float(np.nanmin(dd))

    @staticmethod
    def block_bootstrap_returns(returns: np.ndarray, n_sims: int = 5000, block_size: int = 20, seed: int = 42) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        clean = np.asarray(returns, dtype=float)
        clean = clean[np.isfinite(clean)]
        n = len(clean)
        sims: list[np.ndarray] = []
        if n == 0:
            return sims
        max_start = max(1, n - block_size + 1)
        for _ in range(n_sims):
            resampled: list[float] = []
            while len(resampled) < n:
                start = int(rng.integers(0, max_start))
                block = clean[start:start + block_size]
                resampled.extend(block.tolist())
            sims.append(np.array(resampled[:n], dtype=float))
        return sims

    def mc_summary(self, returns: np.ndarray, n_sims: int = 5000, block_size: int = 20, seed: int = 42) -> dict[str, float]:
        sims = self.block_bootstrap_returns(returns, n_sims=n_sims, block_size=block_size, seed=seed)
        if not sims:
            return {}
        stats = []
        for sim in sims:
            equity = np.exp(np.cumsum(sim))
            terminal = float(equity[-1])
            years = len(sim) / 252.0
            cagr = terminal ** (1.0 / years) - 1.0 if years > 0 else np.nan
            mdd = self.max_drawdown_from_returns(sim)
            stats.append({"terminal_return": terminal - 1.0, "cagr": cagr, "max_drawdown": mdd})
        out = pd.DataFrame(stats)
        return {
            "median_cagr": float(out["cagr"].median()),
            "p05_cagr": float(out["cagr"].quantile(0.05)),
            "p95_max_drawdown": float(out["max_drawdown"].quantile(0.95)),
            "median_max_drawdown": float(out["max_drawdown"].median()),
            "median_terminal_return": float(out["terminal_return"].median()),
        }

    def performance_summary(self, bt: pd.DataFrame) -> dict[str, float | dict[str, float]]:
        valid = bt.dropna(subset=["ret_1d", "strategy_ret_1d"]).copy()
        if valid.empty:
            return {}
        n = len(valid)
        years = n / 252.0
        strat_eq = valid["strategy_equity"].iloc[-1]
        bh_eq = valid["buyhold_equity"].iloc[-1]
        strat_cagr = strat_eq ** (1 / years) - 1 if years > 0 else np.nan
        bh_cagr = bh_eq ** (1 / years) - 1 if years > 0 else np.nan
        strat_dd = valid["strategy_equity"] / valid["strategy_equity"].cummax() - 1
        bh_dd = valid["buyhold_equity"] / valid["buyhold_equity"].cummax() - 1
        strat_sharpe = np.sqrt(252) * valid["strategy_ret_1d"].mean() / valid["strategy_ret_1d"].std(ddof=0)
        bh_sharpe = np.sqrt(252) * valid["ret_1d"].mean() / valid["ret_1d"].std(ddof=0)
        strat_vol = np.sqrt(252) * valid["strategy_ret_1d"].std(ddof=0)
        bh_vol = np.sqrt(252) * valid["ret_1d"].std(ddof=0)
        upside_days = valid["ret_1d"] > 0
        downside_days = valid["ret_1d"] < 0
        upside_capture = valid.loc[upside_days, "strategy_ret_1d"].sum() / valid.loc[upside_days, "ret_1d"].sum()
        downside_capture = valid.loc[downside_days, "strategy_ret_1d"].sum() / valid.loc[downside_days, "ret_1d"].sum()
        strategy_mc = self.mc_summary(valid["strategy_ret_1d"].values, n_sims=self.cfg.mc_sims, block_size=self.cfg.mc_block_size)
        buyhold_mc = self.mc_summary(valid["ret_1d"].values, n_sims=self.cfg.mc_sims, block_size=self.cfg.mc_block_size)
        return {
            "rows": int(n),
            "strategy_total_return": float(strat_eq - 1),
            "buyhold_total_return": float(bh_eq - 1),
            "strategy_cagr": float(strat_cagr),
            "buyhold_cagr": float(bh_cagr),
            "strategy_max_drawdown": float(strat_dd.min()),
            "buyhold_max_drawdown": float(bh_dd.min()),
            "strategy_sharpe": float(strat_sharpe),
            "buyhold_sharpe": float(bh_sharpe),
            "strategy_vol_annualized": float(strat_vol),
            "buyhold_vol_annualized": float(bh_vol),
            "avg_exposure": float(valid["position"].mean()),
            "upside_capture": float(upside_capture),
            "downside_capture": float(downside_capture),
            "strategy_mc": strategy_mc,
            "buyhold_mc": buyhold_mc,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TQQQ long-only invested-or-cash pipeline using YCharts CSV input")
    p.add_argument("input_csv", help="Path to YCharts-style CSV")
    p.add_argument("--output-dir", default="output", help="Directory for output CSV/JSON files")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--tqqq-col", default="TQQQ Close")
    p.add_argument("--qqq-col", default="QQQ Close")
    p.add_argument("--soxx-col", default="SOXX Close")
    p.add_argument("--spy-col", default="SPY Close")
    p.add_argument("--vix-col", default="VIX")
    p.add_argument("--min-state-count", type=int, default=15)
    p.add_argument("--mc-sims", type=int, default=3000)
    p.add_argument("--mc-block-size", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        date_col=args.date_col,
        tqqq_col=args.tqqq_col,
        qqq_col=args.qqq_col,
        soxx_col=args.soxx_col,
        spy_col=args.spy_col,
        vix_col=args.vix_col,
        min_state_count=args.min_state_count,
        mc_sims=args.mc_sims,
        mc_block_size=args.mc_block_size,
    )
    pipe = TQQQPipeline(cfg)

    df = pipe.load_csv(args.input_csv)
    features = pipe.build_features(df)
    research = pipe.build_research_table(features)
    bt = pipe.run_backtest(features)
    summary = pipe.performance_summary(bt)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    features.to_csv(outdir / "features.csv", index=False)
    research.to_csv(outdir / "research_table.csv", index=False)
    bt.to_csv(outdir / "backtest.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    latest = bt.dropna(subset=["date"]).iloc[-1]
    latest_signal = {
        "date": latest["date"].strftime("%Y-%m-%d") if hasattr(latest["date"], "strftime") else str(latest["date"]),
        "target_position": float(latest["target_position"]),
        "live_exposure_class": str(latest["live_exposure_class"]),
        "regime_class": str(latest.get("regime_class", "")),
        "toxic_flags": int(latest.get("toxic_flags", 0)),
        "constructive_score": int(latest.get("constructive_score", 0)),
        "hard_crash_trigger": bool(latest.get("hard_crash_trigger", False)),
        "meltup_failure": bool(latest.get("meltup_failure", False)),
        "hist_invest_flag": int(latest.get("hist_invest_flag", 0)),
        "extension_score": None if pd.isna(latest.get("extension_score")) else float(latest.get("extension_score")),
        "compression_break_signal": int(latest.get("compression_break_signal", 0)),
        "drawdown_20": None if pd.isna(latest.get("drawdown_20")) else float(latest.get("drawdown_20")),
        "vol_ratio_5_20": None if pd.isna(latest.get("vol_ratio_5_20")) else float(latest.get("vol_ratio_5_20")),
    }
    with open(outdir / "latest_signal.json", "w", encoding="utf-8") as f:
        json.dump(latest_signal, f, indent=2)


if __name__ == "__main__":
    main()
