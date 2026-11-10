# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for scrubbing the API key out of DeepSWE results."""

from __future__ import annotations

import json
import pathlib

import deepswe_redact
import pytest

SECRET = "sk-test-0123456789abcdefghijklmnopqrstuvwxyz"


def _deny_b(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes ``b.txt`` unreadable, like a root-owned file to the runner."""
    read_bytes = pathlib.Path.read_bytes

    def deny_b(self: pathlib.Path) -> bytes:
        if self.name == "b.txt":
            raise PermissionError(13, "Permission denied", str(self))
        return read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", deny_b)


def test_plain_and_json_copies_are_removed(tmp_path: pathlib.Path) -> None:
    trial = tmp_path / "trial__1" / "agent"
    trial.mkdir(parents=True)
    (trial / "log.txt").write_text(f"OPENAI_API_KEY={SECRET}\nPATH=/bin\n")
    (trial / "trajectory.json").write_text(
        json.dumps({"observation": f"HOME=/root\nOPENAI_API_KEY={SECRET}\n"})
    )
    (trial / "clean.txt").write_text("nothing to see\n")

    changed, failed = deepswe_redact.redact_tree(tmp_path, SECRET)

    assert sorted(p.name for p in changed) == ["log.txt", "trajectory.json"]
    assert failed == []
    for path in trial.iterdir():
        assert SECRET not in path.read_text()
    assert "PATH=/bin" in (trial / "log.txt").read_text()
    json.loads((trial / "trajectory.json").read_text())


def test_copies_split_by_wrapping_are_removed(tmp_path: pathlib.Path) -> None:
    # A terminal log wraps the env line mid-key; JSON turns the break into \n.
    wrapped = f"OPENAI_API_KEY={SECRET[:20]}\n{SECRET[20:]}\nPAGER=cat\n"
    (tmp_path / "agent.txt").write_text(wrapped)
    (tmp_path / "agent.json").write_text(json.dumps({"out": wrapped}))

    deepswe_redact.redact_tree(tmp_path, SECRET)

    for name in ("agent.txt", "agent.json"):
        text = (tmp_path / name).read_text()
        assert SECRET[:20] not in text
        assert SECRET[20:] not in text
        assert "PAGER=cat" in text


def test_short_or_dummy_keys_are_left_alone(tmp_path: pathlib.Path) -> None:
    (tmp_path / "log.txt").write_text("OPENAI_API_KEY=dummy\n")

    assert deepswe_redact.redact_tree(tmp_path, "dummy") == ([], [])
    assert (tmp_path / "log.txt").read_text() == "OPENAI_API_KEY=dummy\n"


def test_unreadable_file_is_reported_and_the_rest_still_scrubbed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A root-owned file from the rootful podman service, as the runner sees it.
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(f"OPENAI_API_KEY={SECRET}\n")
    _deny_b(monkeypatch)

    changed, failed = deepswe_redact.redact_tree(tmp_path, SECRET)

    assert sorted(p.name for p in changed) == ["a.txt", "c.txt"]
    assert [p.name for p, _ in failed] == ["b.txt"]
    monkeypatch.undo()
    assert SECRET not in (tmp_path / "c.txt").read_text()


def test_main_fails_when_a_file_could_not_be_checked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "log.txt").write_text(f"OPENAI_API_KEY={SECRET}\n")
    monkeypatch.setattr(
        deepswe_redact,
        "redact_tree",
        lambda root, secret: ([], [(root / "log.txt", "Permission denied")]),
    )
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr("sys.argv", ["deepswe_redact.py", str(tmp_path)])

    with pytest.raises(SystemExit) as exit_info:
        deepswe_redact.main()

    assert exit_info.value.code == 1


def test_a_file_that_disappears_is_not_a_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A container cleaning up while the scrub walks the directory.
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(f"OPENAI_API_KEY={SECRET}\n")
    read_bytes = pathlib.Path.read_bytes

    def vanish_b(self: pathlib.Path) -> bytes:
        if self.name == "b.txt":
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", vanish_b)

    changed, failed = deepswe_redact.redact_tree(tmp_path, SECRET)

    assert [p.name for p in changed] == ["a.txt"]
    assert failed == []


def test_remove_unscrubbable_deletes_and_lists_the_bad_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = tmp_path / "trial__1"
    trial.mkdir()
    for name in ("a.txt", "b.txt"):
        (trial / name).write_text(f"OPENAI_API_KEY={SECRET}\n")
    _deny_b(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(
        "sys.argv",
        ["deepswe_redact.py", "--remove-unscrubbable", str(tmp_path)],
    )

    deepswe_redact.main()

    assert sorted(p.name for p in trial.iterdir()) == ["a.txt"]
    assert SECRET not in (trial / "a.txt").read_text()
    listed = (tmp_path / deepswe_redact.REMOVED_LIST).read_text()
    assert listed.startswith("trial__1/b.txt: ")
    assert "Permission denied" in listed


def test_remove_unscrubbable_fails_when_the_file_cannot_be_deleted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "b.txt").write_text(f"OPENAI_API_KEY={SECRET}\n")
    _deny_b(monkeypatch)

    def refuse_unlink(self: pathlib.Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "Operation not permitted", str(self))

    monkeypatch.setattr(pathlib.Path, "unlink", refuse_unlink)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(
        "sys.argv",
        ["deepswe_redact.py", "--remove-unscrubbable", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        deepswe_redact.main()

    assert exit_info.value.code == 1
    assert not (tmp_path / deepswe_redact.REMOVED_LIST).exists()


def test_remove_unscrubbable_leaves_an_unlistable_directory(
    tmp_path: pathlib.Path,
) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()

    removed, remaining = deepswe_redact.remove_unscrubbable(
        tmp_path, [(sub, "Permission denied")]
    )

    assert removed == []
    assert remaining == [(sub, "Permission denied")]
    assert sub.is_dir()
