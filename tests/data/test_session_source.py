import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from factor_engine.data import IndexedBatch, Panel
from factor_engine.session_source import SessionSource, create_session_source


def fixture_batches():
    # 两个 session、三只标的，每条 lane 有四个时点。
    for first, count in ((0, 2), (2, 3), (5, 1)):
        x = np.arange(first * 4, (first + count) * 4, dtype=np.float32).reshape(1, count, 4)
        yield IndexedBatch(first, Panel({"x": x, "y": -x}, x % 3 != 0), x + 1)


def make_source(path):
    create_session_source(
        path,
        fixture_batches(),
        sessions=2,
        securities=3,
        time=4,
        provenance={"kind": "hand_checked_test"},
        byte_limit=4096,
    )
    return SessionSource(path, max_batch_bytes=1024)


@pytest.mark.parametrize("explicit_valid", [False, True])
def test_panel_normalizes_valid_before_session_selection(explicit_valid):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    valid = x % 3 != 0 if explicit_valid else None
    panel = Panel({"x": x}, valid)
    expected = x % 3 != 0 if explicit_valid else np.ones(x.shape, dtype=bool)
    # 磁盘读取与 session 选择均依赖构造后的完整布尔资格数组。
    assert isinstance(panel.valid, np.ndarray) and panel.valid.dtype == np.bool_
    np.testing.assert_array_equal(panel.valid, expected)
    selected = panel.subset_sessions(slice(1, 2))
    np.testing.assert_array_equal(selected.valid, expected[1:2])
    np.testing.assert_array_equal(selected.fields["x"], x[1:2])


def test_complete_lane_reads_fold_order_and_owned_buffers(tmp_path):
    with make_source(tmp_path / "source") as source:
        view = source.view([1, 0])
        batches = list(view.iter_batches(0, 2))
        assert [b.first_lane for b in batches] == [0, 2, 4]
        assert view.global_lanes(0, 4) == (3, 4, 5, 0)
        assert batches[0].panel.fields["x"].tolist() == [
            [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0]]
        ]
        assert batches[1].panel.fields["x"].tolist() == [
            [[20.0, 21.0, 22.0, 23.0], [0.0, 1.0, 2.0, 3.0]]
        ]
        assert batches[0].target.tolist() == [[[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0]]]
        assert not batches[0].panel.fields["x"].flags.writeable
        assert not batches[0].target.flags.writeable
        assert (
            source.stats["max_batch_bytes"] == 104
        )  # 2 条 lane × 4 个时点 ×（两个字段 8 字节 + valid 1 字节 + target 4 字节）
        assert source.stats["read_lanes"] == 6
        assert len(source.source_identity) == 64
        assert view.source_identity != source.source_identity
        assert next(view.iter_batches(4, 2)).first_lane == 4
    with pytest.raises(RuntimeError, match="closed"):
        next(view.iter_batches(0, 1))


def test_source_budget_is_checked_before_reads(tmp_path):
    with make_source(tmp_path / "source") as source:
        with pytest.raises(MemoryError):
            next(source.iter_batches(0, 100))
        assert source.stats["read_lanes"] == 0


def test_random_complete_batch_keeps_fold_order_and_checks_limit_before_io(tmp_path):
    with make_source(tmp_path / "source") as source:
        view = source.view([1, 0])
        batch = view.read_batch(2, 2)
        assert batch.first_lane == 2
        assert batch.panel.fields["x"].tolist() == [
            [[20.0, 21.0, 22.0, 23.0], [0.0, 1.0, 2.0, 3.0]]
        ]
        assert batch.target.tolist() == [[[21.0, 22.0, 23.0, 24.0], [1.0, 2.0, 3.0, 4.0]]]
        assert not batch.target.flags.writeable
        before = dict(source.stats)
        with pytest.raises(MemoryError):
            view.read_batch(0, 100)
        with pytest.raises(ValueError):
            view.read_batch(6, 1)
        assert source.stats == before
        assert source.read_batch(5, 3).panel.fields["x"].tolist() == [[[20.0, 21.0, 22.0, 23.0]]]


def test_creation_rejects_missing_duplicate_or_reordered_lanes_and_cleans_only_own_stage(tmp_path):
    path = tmp_path / "source"
    keep = tmp_path / "keep"
    keep.write_text("untouched")
    bad = list(fixture_batches())
    bad[1] = IndexedBatch(1, bad[1].panel, bad[1].target)
    with pytest.raises(ValueError, match="ordinal"):
        create_session_source(
            path, bad, sessions=2, securities=3, time=4, provenance={}, byte_limit=4096
        )
    assert not path.exists()
    assert keep.read_text() == "untouched"
    with pytest.raises(MemoryError):
        create_session_source(
            path, fixture_batches(), sessions=2, securities=3, time=4, provenance={}, byte_limit=1
        )
    with make_source(path):
        pass
    with pytest.raises(FileExistsError):
        create_session_source(
            path,
            fixture_batches(),
            sessions=2,
            securities=3,
            time=4,
            provenance={},
            byte_limit=4096,
        )


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("batch_type", TypeError),
        ("fields", ValueError),
        ("budget", MemoryError),
        ("provenance", TypeError),
        ("parent_directory", FileExistsError),
    ],
)
def test_creation_closes_consumed_generator_on_preparation_failure(tmp_path, failure, error):
    lease_path = tmp_path / "input.bin"
    lease_path.write_bytes(b"input resource")
    opened = []
    first = next(fixture_batches())
    provenance = {}
    byte_limit = 4096
    path = tmp_path / "source"
    if failure == "batch_type":
        first = object()
    elif failure == "fields":
        first = IndexedBatch(0, Panel({"": first.target}), first.target)
    elif failure == "budget":
        byte_limit = 1
    elif failure == "provenance":
        provenance = {"unsupported": object()}
    elif failure == "parent_directory":
        parent = tmp_path / "parent"
        parent.write_text("keep existing file")
        path = parent / "source"

    def batches():
        with lease_path.open("rb") as resource:
            opened.append(resource)
            yield first
            yield from fixture_batches()

    iterator = batches()
    try:
        with pytest.raises(error):
            create_session_source(
                path,
                iterator,
                sessions=2,
                securities=3,
                time=4,
                provenance=provenance,
                byte_limit=byte_limit,
            )
        assert opened and opened[0].closed
        assert not path.exists()
        assert not list(tmp_path.glob(".source-*"))
        assert lease_path.read_bytes() == b"input resource"
    finally:
        iterator.close()


def test_creation_closes_partial_output_files_and_removes_stage(tmp_path, monkeypatch):
    original_open = Path.open
    opened = []

    def open_until_second_field(path, mode="r", *args, **kwargs):
        if mode == "xb" and path.name == "field-1.bin":
            raise OSError("injected output open failure")
        stream = original_open(path, mode, *args, **kwargs)
        if mode == "xb":
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_until_second_field)
    try:
        with pytest.raises(OSError, match="output open failure"):
            create_session_source(
                tmp_path / "source",
                fixture_batches(),
                sessions=2,
                securities=3,
                time=4,
                provenance={},
                byte_limit=4096,
            )
        assert opened and all(stream.closed for stream in opened)
        assert not list(tmp_path.iterdir())
    finally:
        for stream in opened:
            stream.close()


def test_manifest_tampering_invalid_dimensions_and_duplicate_view_fail(tmp_path):
    path = tmp_path / "source"
    with make_source(path) as source:
        with pytest.raises(ValueError):
            source.view([0, 0])
        with pytest.raises(ValueError):
            source.view([2])
        with pytest.raises(ValueError):
            next(source.iter_batches(-1, 1))
    manifest = path / "manifest.json"
    body = json.loads(manifest.read_text())
    body["securities"] = 4
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        SessionSource(path)


@pytest.mark.skipif(
    importlib.util.find_spec("factor_engine.session_source") is None,
    reason="complete upstream source tree required",
)
def test_full_source_read_does_not_recreate_session_view(tmp_path, monkeypatch):
    from factor_engine import session_source as module
    from factor_engine.data import IndexedBatch, Panel

    x = np.arange(28, dtype=np.float32).reshape(1, 7, 4)
    path = tmp_path / "source"
    module.create_session_source(
        path,
        [IndexedBatch(0, Panel({"x": x}), x)],
        sessions=1,
        securities=7,
        time=4,
        provenance={"fixture": "fast-read"},
        byte_limit=1024,
    )
    with module.SessionSource(path, max_batch_bytes=1024) as source:

        def forbidden(*args, **kwargs):
            raise AssertionError("full-source path created SessionView")

        monkeypatch.setattr(module, "SessionView", forbidden)
        batches = list(source.iter_batches(0, 4))
        assert [b.first_lane for b in batches] == [0, 4]
        np.testing.assert_array_equal(np.concatenate([b.target for b in batches], axis=1), x)
        assert all(not b.target.flags.writeable for b in batches)
