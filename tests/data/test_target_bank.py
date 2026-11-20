from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from factor_engine.marketlike import TERMINALS, MarketLikeSource, derive_marketlike
from factor_engine.session_source import SessionSource, create_session_source


def api():
    from factor_engine.target_bank import MultiTargetSource, TargetBatch, targets_from_close

    return TargetBatch, targets_from_close, MultiTargetSource


def test_target_batch_owns_contiguous_immutable_snapshots():
    TargetBatch, _, _ = api()
    values = np.arange(48, dtype=np.float64).reshape(2, 3, 8)[..., ::2]
    eligible = np.ones(values.shape, dtype=bool)
    mature = eligible.copy()
    batch = TargetBatch((5, 30), values, eligible, mature)
    expected = values.astype(np.float32).copy()
    values[:] = -100
    eligible[:] = False
    mature[:] = False
    assert batch.horizons == (5, 30)
    np.testing.assert_array_equal(batch.values, expected)
    assert batch.mature is not None
    assert batch.eligible.all() and batch.mature.all()
    for array, dtype in (
        (batch.values, np.float32),
        (batch.eligible, np.bool_),
        (batch.mature, np.bool_),
    ):
        assert array.dtype == dtype and array.flags.c_contiguous
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(batch, "values", expected)
    assert TargetBatch((1,), expected[:1], batch.eligible[:1]).mature is None


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (True,), (1.5,), (2, 2)])
def test_bad_horizons_are_rejected(horizons):
    TargetBatch, _, _ = api()
    with pytest.raises(ValueError, match="horizon"):
        TargetBatch(
            horizons, np.ones((len(horizons), 1, 3)), np.ones((len(horizons), 1, 3), dtype=bool)
        )


def test_target_shape_masks_and_maturity_are_checked():
    TargetBatch, _, _ = api()
    values = np.ones((1, 2, 3), dtype=np.float32)
    eligible = np.ones(values.shape, dtype=bool)
    for invalid in (values[0], values[:, :0], values[..., :0]):
        with pytest.raises(ValueError, match="shape"):
            TargetBatch((1,), invalid, np.ones(invalid.shape, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        TargetBatch((1, 2), values, eligible)
    with pytest.raises(ValueError, match="shape"):
        TargetBatch((1,), values, eligible[:, :1])
    with pytest.raises(ValueError, match="boolean"):
        TargetBatch((1,), values, eligible.astype(np.int32))
    with pytest.raises(ValueError, match="mature"):
        TargetBatch((1,), values, eligible, np.zeros_like(eligible))


@pytest.mark.parametrize("time", [1, 2, 3, 29, 30, 31, 33, 67])
def test_close_targets_respect_lane_boundaries_and_h30_exact_math(time):
    _, targets_from_close, _ = api()
    rng = np.random.default_rng(time)
    close = np.exp(rng.normal(4, 0.1, (2, 3, time))).astype(np.float32)
    available = rng.random(close.shape) > 0.1
    close[~available] = np.nan
    volume = np.where(available, 1, np.nan).astype(np.float32)
    fields = {name: close.copy() for name in TERMINALS[:4]}
    fields["volume"] = volume
    _, old = derive_marketlike(fields, available)
    batch = targets_from_close(close, (1, 30, 300), available)
    assert batch.mature is not None
    assert batch.values.shape == (3, 6, time)
    assert batch.values[1].tobytes() == old.reshape(6, time).tobytes()
    for index, horizon in enumerate(batch.horizons):
        width = max(time - horizon, 0)
        assert batch.mature[index, :, :width].all()
        assert not batch.mature[index, :, width:].any()
        assert not batch.eligible[index, :, width:].any()
        assert np.isnan(batch.values[index, :, width:]).all()
        expected = np.zeros((6, time), dtype=bool)
        if width:
            mask = available.reshape(6, time)
            expected[:, :width] = mask[:, :width] & mask[:, horizon:]
        np.testing.assert_array_equal(batch.eligible[index], expected)


def test_maturity_does_not_turn_missing_prices_into_observations():
    _, targets_from_close, _ = api()
    close = np.array([[1, 2, np.nan, 4, 0, -1, np.inf, 8]], dtype=np.float64)
    batch = targets_from_close(close, (1,))
    assert batch.mature is not None
    np.testing.assert_array_equal(batch.mature[0, 0], [True] * 7 + [False])
    np.testing.assert_array_equal(batch.eligible[0, 0], [True] + [False] * 7)
    assert batch.values[0, 0, 0] == np.float32(np.log(2))
    assert np.isnan(batch.values[~batch.eligible]).all()


def test_h30_generated_source_regression_with_missing_and_zero_trade():
    _, targets_from_close, _ = api()
    source = MarketLikeSource(seed=42, sessions=3, securities=2, time=1025)
    panel, old = source.read_lanes((0, 1, 4, 5))
    actual = targets_from_close(panel.fields["close"], (30,), panel.valid)
    assert actual.values[0].tobytes() == old.reshape(4, 1025).tobytes()


@pytest.mark.parametrize("close", [np.ones(4), np.ones((1, 1, 1, 4)), np.ones((0, 4))])
def test_invalid_close_shape_rejected(close):
    _, targets_from_close, _ = api()
    with pytest.raises(ValueError, match="close"):
        targets_from_close(close, (1,))


def test_close_availability_must_be_aligned_boolean():
    _, targets_from_close, _ = api()
    for valid in (np.ones((2, 5), dtype=bool), np.ones((1, 5), dtype=int)):
        with pytest.raises(ValueError, match="valid"):
            targets_from_close(np.ones((1, 5)), (1,), valid)


def test_multitarget_source_preserves_original_batches_and_tail(tmp_path):
    _, targets_from_close, MultiTargetSource = api()
    generator = MarketLikeSource(seed=11, sessions=3, securities=2, time=67)
    create_session_source(
        tmp_path / "source",
        generator.iter_batches(0, 2),
        sessions=3,
        securities=2,
        time=67,
        provenance=generator.provenance,
        byte_limit=generator.required_bytes,
    )
    panel, _ = generator.read_lanes(range(6))
    targets = targets_from_close(panel.fields["close"], (1, 5, 30), panel.valid)
    with SessionSource(tmp_path / "source") as source:
        wrapped = MultiTargetSource(source, targets)
        assert wrapped.field_names == source.field_names
        assert wrapped.lane_count == 6 and wrapped.time == 67
        batches = list(wrapped.iter_batches(1, 4))
        assert [batch.first_lane for batch, _ in batches] == [1, 5]
        for original, selected in batches:
            count = original.panel.shape[0] * original.panel.shape[1]
            begin = original.first_lane
            assert original.target.shape == original.panel.shape
            assert selected.horizons == targets.horizons
            np.testing.assert_array_equal(selected.values, targets.values[:, begin : begin + count])
            np.testing.assert_array_equal(
                selected.eligible, targets.eligible[:, begin : begin + count]
            )
            assert not selected.values.flags.writeable
        with pytest.raises(ValueError, match="shape"):
            MultiTargetSource(source, targets_from_close(np.ones((5, 67)), (1,)))
        with pytest.raises(ValueError, match="shape"):
            MultiTargetSource(source, targets_from_close(np.ones((6, 68)), (1,)))
    with pytest.raises(RuntimeError):
        next(wrapped.iter_batches(0, 1))
