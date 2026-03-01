"""Tests for backtesting engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pm_sim.backtest import (
    BacktestResult,
    PriceSnapshot,
    _build_synthetic_book,
    load_snapshots_csv,
    load_snapshots_json,
    run_backtest,
)
from pm_sim.engine import Engine


# ---------------------------------------------------------------------------
# Synthetic book
# ---------------------------------------------------------------------------


class TestBuildSyntheticBook:
    def test_basic_structure(self):
        book = _build_synthetic_book(0.50)
        assert len(book.asks) == 3
        assert len(book.bids) == 3
        # Asks should be above midpoint
        assert all(a.price >= 0.50 for a in book.asks)
        # Bids should be below midpoint
        assert all(b.price <= 0.50 for b in book.bids)

    def test_extreme_high_price(self):
        book = _build_synthetic_book(0.98)
        assert all(a.price <= 0.99 for a in book.asks)

    def test_extreme_low_price(self):
        book = _build_synthetic_book(0.02)
        assert all(b.price >= 0.01 for b in book.bids)

    def test_custom_depth(self):
        book = _build_synthetic_book(0.50, depth=1000.0)
        assert book.asks[0].size == 1000.0


# ---------------------------------------------------------------------------
# Loading snapshots
# ---------------------------------------------------------------------------


class TestLoadSnapshotsCsv:
    def test_loads_csv(self, tmp_path: Path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "timestamp,market_slug,outcome,midpoint\n"
            "2026-01-01T00:00:00Z,test-market,yes,0.65\n"
            "2026-01-01T01:00:00Z,test-market,yes,0.70\n"
        )
        snapshots = load_snapshots_csv(csv_file)
        assert len(snapshots) == 2
        assert snapshots[0].midpoint == 0.65
        assert snapshots[1].midpoint == 0.70
        assert snapshots[0].market_slug == "test-market"
        assert snapshots[0].outcome == "yes"


class TestLoadSnapshotsJson:
    def test_loads_json(self, tmp_path: Path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"timestamp": "2026-01-01T00:00:00Z", "market_slug": "m1", "outcome": "Yes", "midpoint": 0.55},
            {"timestamp": "2026-01-01T01:00:00Z", "market_slug": "m1", "outcome": "yes", "midpoint": 0.60},
        ]))
        snapshots = load_snapshots_json(json_file)
        assert len(snapshots) == 2
        assert snapshots[0].outcome == "yes"  # normalized to lower


# ---------------------------------------------------------------------------
# Running backtests
# ---------------------------------------------------------------------------


def noop_strategy(engine: Engine, snapshot: PriceSnapshot, prices: dict) -> None:
    """Does nothing."""
    pass


def buy_on_dip_strategy(engine: Engine, snapshot: PriceSnapshot, prices: dict) -> None:
    """Buys when price dips below 0.50."""
    if snapshot.midpoint < 0.50:
        engine.buy(snapshot.market_slug, snapshot.outcome, 100.0)


class TestRunBacktest:
    def test_noop_strategy(self):
        snapshots = [
            PriceSnapshot("2026-01-01T00:00:00Z", "m1", "yes", 0.50),
            PriceSnapshot("2026-01-01T01:00:00Z", "m1", "yes", 0.55),
        ]
        result = run_backtest(snapshots, noop_strategy, "noop")
        assert result.strategy == "noop"
        assert result.starting_balance == 10_000.0
        assert result.ending_cash == 10_000.0
        assert result.total_trades == 0
        assert result.pnl == 0.0
        assert result.snapshots_processed == 2

    def test_custom_balance(self):
        snapshots = [PriceSnapshot("2026-01-01T00:00:00Z", "m1", "yes", 0.50)]
        result = run_backtest(snapshots, noop_strategy, balance=5_000.0)
        assert result.starting_balance == 5_000.0

    def test_strategy_error_doesnt_crash(self):
        def bad_strategy(engine, snapshot, prices):
            raise ValueError("boom")

        snapshots = [PriceSnapshot("2026-01-01T00:00:00Z", "m1", "yes", 0.50)]
        result = run_backtest(snapshots, bad_strategy, "bad")
        assert result.snapshots_processed == 1
        assert result.total_trades == 0

    def test_result_fields(self):
        snapshots = [PriceSnapshot("2026-01-01T00:00:00Z", "m1", "yes", 0.50)]
        result = run_backtest(snapshots, noop_strategy)
        assert isinstance(result, BacktestResult)
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "max_drawdown")
