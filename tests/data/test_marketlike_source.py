"""检查生成器的端点和状态，不评价模拟数据的金融拟合程度。"""

import json

import numpy as np
import pytest

from factor_engine.marketlike import (
    MarketLikeSource,
    counter_uniform,
    derive_marketlike,
    input_coverage,
)
from factor_engine.session_source import SessionSource, create_session_source


EXPECTED_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "range_pct",
    "body_pct",
    "close_location",
    "relative_volume_60",
    "realized_vol_60",
)


def prices(time=100, lanes=1):
    close = (
        np.broadcast_to((2 * np.exp(0.001 * np.arange(time)))[None, None], (1, lanes, time))
        .astype("f4")
        .copy()
    )
    return {
        "open": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.ones_like(close),
    }, np.ones(close.shape, dtype=bool)


def test_ohlc_zero_trade_and_missing_keep_distinct_observations():
    raw, available = prices()
    for field in ("open", "high", "low", "close"):
        raw[field][0, 0, 10] = np.nan
    raw["volume"][0, 0, 10] = 0
    available[0, 0, 15] = False
    for array in raw.values():
        array[0, 0, 15] = np.nan
    panel, target = derive_marketlike(raw, available)
    assert panel.valid is not None
    assert tuple(panel.fields) == EXPECTED_FIELDS
    assert panel.valid[0, 0, 10] and panel.fields["volume"][0, 0, 10] == 0
    assert np.isnan(panel.fields["close"][0, 0, 10])
    assert not panel.valid[0, 0, 15]
    assert all(np.isnan(a[0, 0, 15]) for a in panel.fields.values())
    assert all(a.dtype == np.float32 for a in panel.fields.values()) and target.dtype == np.float32
    assert "target_h30" not in panel.fields
    assert np.isnan(target[0, 0, 10]) and np.isnan(target[0, 0, 15])


def test_h30_endpoint_alignment_and_no_cross_session_future_label():
    raw, available = prices(time=80, lanes=2)
    panel, target = derive_marketlike(raw, available)
    expected = np.float32(np.log(float(raw["close"][0, 0, 30]) / float(raw["close"][0, 0, 0])))
    assert target[0, 0, 0] == expected
    assert np.isfinite(target[0, 0, 49]) and np.isnan(target[0, 0, 50:]).all()
    for name in ("open", "high", "low", "close"):
        raw[name][0, 0, 30] = np.nan
    raw["volume"][0, 0, 30] = 0
    _, changed = derive_marketlike(raw, available)
    assert np.isnan(changed[0, 0, 0]) and np.isfinite(changed[0, 1, 0])
    assert np.array_equal(panel.fields["close"][0, 1], raw["close"][0, 1])


def test_neutral_zero_range_no_epsilon_and_full_finite_window_reset():
    shape = (2, 1, 65)
    one = np.ones(shape, dtype="f4")
    raw = {n: one.copy() for n in ("open", "high", "low", "close", "volume")}
    panel, target = derive_marketlike(raw, np.ones(shape, dtype=bool))
    assert (panel.fields["close_location"] == np.float32(0.5)).all()
    assert (panel.fields["range_pct"] == 0).all() and (panel.fields["body_pct"] == 0).all()
    assert np.isnan(panel.fields["ret_1"][..., 0]).all()
    assert (panel.fields["ret_1"][..., 1:] == 0).all()
    assert np.isnan(panel.fields["ret_5"][..., :5]).all()
    assert np.isnan(panel.fields["relative_volume_60"][..., :59]).all()
    assert (panel.fields["relative_volume_60"][..., 59:] == 1).all()
    assert np.isnan(panel.fields["realized_vol_60"][..., :60]).all()
    assert (panel.fields["realized_vol_60"][..., 60:] == 0).all()
    assert np.isnan(target[..., -30:]).all()


def test_zero_volume_window_has_no_arbitrary_denominator_floor():
    raw, available = prices(time=65)
    for name in ("open", "high", "low", "close"):
        raw[name][..., :60] = np.nan
    raw["volume"][..., :60] = 0
    panel, _ = derive_marketlike(raw, available)
    assert np.isnan(panel.fields["relative_volume_60"][..., 59]).all()
    assert panel.fields["relative_volume_60"][0, 0, 60] == np.float32(60)


def test_terminals_are_causal_even_though_target_is_future_outcome():
    raw, available = prices(time=100)
    first, first_target = derive_marketlike(raw, available)
    for name in ("open", "high", "low", "close"):
        raw[name][..., 80:] *= 2
    raw["volume"][..., 80:] *= 3
    changed, changed_target = derive_marketlike(raw, available)
    for name in EXPECTED_FIELDS:
        np.testing.assert_array_equal(first.fields[name][..., :80], changed.fields[name][..., :80])
    assert first_target[0, 0, 60] != changed_target[0, 0, 60]


@pytest.mark.parametrize(
    "bad", ["negative_volume", "illegal_high", "zero_price", "fake_zero_trade_ohlc"]
)
def test_illegal_market_input_is_rejected_not_silently_repaired(bad):
    raw, available = prices()
    if bad == "negative_volume":
        raw["volume"][0, 0, 0] = -1
    if bad == "illegal_high":
        raw["high"][0, 0, 0] = 0.1
    if bad == "zero_price":
        raw["low"][0, 0, 0] = 0
    if bad == "fake_zero_trade_ohlc":
        raw["volume"][0, 0, 0] = 0
    with pytest.raises(ValueError):
        derive_marketlike(raw, available)


def test_counter_domains_and_positions_do_not_alias_at_uint32_boundary():
    positions = np.array([0, 1, 2**32, 2**32 + 1, 2**63 + 3], dtype=np.uint64)
    values = counter_uniform(positions, seed=7, stream=3)
    assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all()
    assert len(np.unique(values)) == len(positions)
    np.testing.assert_array_equal(values[::-1], counter_uniform(positions[::-1], seed=7, stream=3))
    assert not np.array_equal(values, counter_uniform(positions, seed=8, stream=3))
    assert not np.array_equal(values, counter_uniform(positions, seed=7, stream=4))


def test_generated_lanes_match_batch_order_and_long_session_indices():
    source = MarketLikeSource(seed=71, sessions=500, securities=400, time=23400)
    indices = (0, 1, 458 * 400, 459 * 400 + 3)
    panel, target = source.read_lanes(indices)
    for i, lane in enumerate(indices):
        single, label = source.read_lanes((lane,))
        for name in EXPECTED_FIELDS:
            np.testing.assert_array_equal(panel.fields[name][:, i : i + 1], single.fields[name])
        np.testing.assert_array_equal(target[:, i : i + 1], label)
    reverse, labels = source.read_lanes(indices[::-1])
    for name in EXPECTED_FIELDS:
        np.testing.assert_array_equal(panel.fields[name][:, ::-1], reverse.fields[name])
    np.testing.assert_array_equal(target[:, ::-1], labels)
    assert panel.valid is not None
    traded = panel.valid & (panel.fields["volume"] > 0)
    assert np.all(
        panel.fields["high"][traded]
        >= np.maximum(panel.fields["open"], panel.fields["close"])[traded]
    )
    assert np.all(
        panel.fields["low"][traded]
        <= np.minimum(panel.fields["open"], panel.fields["close"])[traded]
    )
    assert np.all(panel.fields["low"][traded] > 0)


def test_uint64_source_range_and_invalid_seed_reject_before_allocation():
    with pytest.raises(ValueError):
        MarketLikeSource(seed=1, sessions=2**63, securities=400, time=23400)
    with pytest.raises(ValueError):
        MarketLikeSource(seed=2**64, sessions=1, securities=1, time=32)
    source = MarketLikeSource(seed=1, sessions=1, securities=2, time=32)
    with pytest.raises(ValueError):
        source.read_lanes((-1,))
    with pytest.raises(ValueError):
        source.read_lanes((2,))


def test_source_persistence_reopen_and_writer_batch_width_identity(tmp_path):
    src = MarketLikeSource(seed=6, sessions=2, securities=3, time=128)
    for width, name in ((1, "one"), (4, "four")):
        create_session_source(
            tmp_path / name,
            src.iter_batches(0, width),
            sessions=2,
            securities=3,
            time=128,
            provenance=src.provenance,
            byte_limit=2 * 3 * 128 * 53,
        )
    a = json.loads((tmp_path / "one/manifest.json").read_text())
    b = json.loads((tmp_path / "four/manifest.json").read_text())
    assert a["source_identity"] == b["source_identity"]
    with SessionSource(tmp_path / "four") as disk:
        chunks = list(disk.view((1, 0)).iter_batches(0, 2))
        expected, target = src.read_lanes((3, 4, 5, 0, 1, 2))
        for name in EXPECTED_FIELDS:
            np.testing.assert_array_equal(
                np.concatenate([x.panel.fields[name] for x in chunks], axis=1),
                expected.fields[name],
            )
        np.testing.assert_array_equal(np.concatenate([x.target for x in chunks], axis=1), target)
    with SessionSource(tmp_path / "one") as reopened:
        assert reopened.source_identity == a["source_identity"]


def test_coverage_reports_strict_windows_and_actual_state_counts():
    raw, available = prices(time=100)
    panel, target = derive_marketlike(raw, available)
    report = input_coverage(panel, target)
    assert report["ohlc_legal_fraction"] == 1 and report["volume_nonnegative_fraction"] == 1
    assert report["zero_trade_fraction"] == 0 and report["missing_source_fraction"] == 0
    assert report["terminal_finite_fraction"]["volume"] == 1
    assert report["window_valid_fraction"]["volume"]["60"] == pytest.approx(0.41)
    assert report["window_valid_fraction"]["volume"]["300"] == 0
    assert report["window_valid_fraction"]["volume"]["900"] == 0
    assert report["rank_ic_valid_lane_fraction"]["close"] == 1
    assert report["rank_ic_valid_lane_fraction"]["volume"] == 0  # 成交量保持不变
    assert (
        MarketLikeSource(seed=1, sessions=5, securities=400, time=23400).required_bytes
        == 2480400000
    )
