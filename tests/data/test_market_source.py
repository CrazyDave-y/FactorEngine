import hashlib

import numpy as np
import pytest
from factor_engine.marketlike import MarketLikeSource
from factor_engine.session_source import create_session_source


def api():
    from factor_engine.market_source import CompositeSource, prepare_tail

    return CompositeSource, prepare_tail


def head_source(tmp_path):
    generator = MarketLikeSource(seed=42, sessions=2, securities=2, time=72)
    head = tmp_path / "head"
    create_session_source(
        head,
        generator.iter_batches(0, 2),
        sessions=2,
        securities=2,
        time=72,
        provenance=generator.provenance,
        byte_limit=generator.required_bytes,
    )
    return head


def test_complete_prefix_tail_and_boundary(tmp_path):
    CompositeSource, prepare_tail = api()
    head = head_source(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in head.iterdir()}
    tail = tmp_path / "tail"
    report = prepare_tail(head, tail, sessions=4, securities=2, time=72, batch_lanes=2)
    assert report["prefix_checked_lanes"] == 3
    assert report["tail_global_first_lane"] == 4
    assert report["tail_bytes"] == 4 * 72 * 53
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in head.iterdir()}
    golden = MarketLikeSource(seed=42, sessions=4, securities=2, time=72)
    with CompositeSource(head, tail, max_batch_bytes=3 * 72 * 53) as source:
        assert source.shape == (4, 2, 72)
        assert source.provenance["schema"] == "factorengine-marketlike-12-v2.0"
        assert source.provenance["seed"] == 42
        assert source.global_lanes(3, 3) == (3, 4, 5)
        batches = list(source.iter_batches(0, 3))
        assert [b.first_lane for b in batches] == [0, 3, 4, 7]
        for batch in batches:
            count = batch.panel.shape[1]
            panel, target = golden.read_lanes(range(batch.first_lane, batch.first_lane + count))
            for name in panel.fields:
                assert batch.panel.fields[name].tobytes() == panel.fields[name].tobytes()
                assert not batch.panel.fields[name].flags.writeable
            assert batch.target.tobytes() == target.tobytes()
            np.testing.assert_array_equal(batch.panel.valid, panel.valid)
        assert source.stats["read_lanes"] == 8
        assert source.stats["read_bytes"] == 8 * 72 * 53
        assert source.stats["max_batch_bytes"] <= 3 * 72 * 53
        with pytest.raises(MemoryError):
            source.read_batch(0, 4)
        with pytest.raises(ValueError):
            source.read_batch(8, 1)


def test_prepare_rejects_wrong_seed(tmp_path):
    _, prepare_tail = api()
    head = head_source(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        prepare_tail(head, tmp_path / "wrong", sessions=4, securities=2, time=72, seed=1)


def test_unrelated_tail_is_rejected(tmp_path):
    CompositeSource, _ = api()
    head = head_source(tmp_path)
    with pytest.raises(ValueError, match="extension"):
        CompositeSource(head, head)


def test_composite_view_reads_arbitrary_session_order_across_stores(tmp_path):
    from factor_engine.market_source import CompositeSource, prepare_tail

    head = head_source(tmp_path)
    tail = tmp_path / "tail"
    prepare_tail(head, tail, sessions=4, securities=2, time=72)
    expected = MarketLikeSource(seed=42, sessions=4, securities=2, time=72)
    with CompositeSource(head, tail) as source:
        batches = list(source.view((3, 0)).iter_batches(0, 3))
        for batch in batches:
            lanes = (6, 7, 0, 1)[batch.first_lane : batch.first_lane + batch.panel.shape[1]]
            panel, target = expected.read_lanes(lanes)
            for name in source.field_names:
                np.testing.assert_array_equal(batch.panel.fields[name], panel.fields[name])
            np.testing.assert_array_equal(batch.target, target)


def test_source_opener_keeps_composite_prefix_and_rejects_identity_changes(tmp_path):
    import json

    from factor_engine.market_source import open_market_source, prepare_marketlike, prepare_tail

    gen = MarketLikeSource(seed=42, sessions=2, securities=2, time=72)
    head = tmp_path / "head"
    prepare_marketlike(head, gen, batch_lanes=2)
    tail = tmp_path / "tail"
    prepare_tail(head, tail, sessions=4, securities=2, time=72)
    with open_market_source(tail) as source:
        assert source.sessions == 4 and source.lane_count == 8
        assert source.stats["read_lanes"] == 0
    path = tail / "composite.json"
    record = json.loads(path.read_text())
    record["head_source_identity"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        open_market_source(tail)


def test_missing_composite_descriptor_cannot_silently_open_only_the_tail(tmp_path):
    from factor_engine.market_source import open_market_source, prepare_tail

    head = head_source(tmp_path)
    tail = tmp_path / "tail"
    prepare_tail(head, tail, sessions=4, securities=2, time=72)
    (tail / "composite.json").unlink()
    with pytest.raises(ValueError, match="descriptor"):
        open_market_source(tail)


@pytest.mark.parametrize("missing", ["head_source_identity", "tail_source_identity"])
def test_incomplete_descriptor_does_not_leak_file_descriptors(tmp_path, missing):
    import json
    from pathlib import Path

    from factor_engine.market_source import open_market_source, prepare_tail

    head = head_source(tmp_path)
    tail = tmp_path / "tail"
    prepare_tail(head, tail, sessions=4, securities=2, time=72)
    descriptor = tail / "composite.json"
    body = json.loads(descriptor.read_text())
    del body[missing]
    descriptor.write_text(json.dumps(body))
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises((KeyError, ValueError)) as error:
        open_market_source(tail)
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    assert isinstance(error.value, ValueError)
