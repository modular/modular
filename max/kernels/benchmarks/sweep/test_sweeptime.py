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
"""Tests for the sweep phase report."""

import json
from pathlib import Path

from sweeptime import collect, format_table, main, parse_lines

LOG_A = """\
Running something
SWEEPTIME,combined,device_init,1,0.5
SWEEPTIME,combined,bench_measured,4,1.0
SWEEPTIME,combined,pre_main,1,0.05
SWEEPTIME,combined,process_total,1,1.55
"""

LOG_B = """\
SWEEPTIME,h8_s6,device_init,1,0.75
SWEEPTIME,h8_s6,bench_measured,2,0.25
SWEEPTIME,h8_s6,pre_main,1,0.0
SWEEPTIME,h8_s6,process_total,1,1.5
"""


def test_parse_lines_skips_other_output() -> None:
    rows = list(parse_lines(LOG_A.splitlines() + ["SWEEPTIME,bad"]))
    assert rows[0] == ("combined", "device_init", 1, 0.5)
    assert len(rows) == 4


def test_collect_sums_across_processes(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "test.log").write_text(LOG_A)
    (tmp_path / "b.json").write_text(
        json.dumps(
            {
                "tag": "h8_s6",
                "phases": {
                    "device_init": {"count": 1, "seconds": 0.75},
                    "bench_measured": {"count": 2, "seconds": 0.25},
                    "pre_main": {"count": 1, "seconds": 0.0},
                    "process_total": {"count": 1, "seconds": 1.5},
                },
            }
        )
    )
    total = collect([tmp_path], by_tag=False)["all"]
    assert total.processes == 2
    assert total.seconds["device_init"] == 1.25
    assert total.counts["bench_measured"] == 6
    assert set(collect([tmp_path], by_tag=True)) == {"combined", "h8_s6"}


def test_format_table_reports_gaps(tmp_path: Path) -> None:
    path = tmp_path / "test.log"
    path.write_text(
        LOG_A + LOG_B.replace("process_total,1,1.5", "process_total,1,2.5")
    )
    table = format_table("all", collect([path], by_tag=False)["all"])
    assert "unaccounted" in table
    assert "process_total" in table
    exact = tmp_path / "exact.log"
    exact.write_text(LOG_A)
    assert "unaccounted" not in format_table(
        "all", collect([exact], by_tag=False)["all"]
    )


def test_main_without_lines_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n")
    assert main([str(empty)]) == 1
