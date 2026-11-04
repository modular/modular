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
"""Cell lists: the shapes a sparse MLA sweep runs, as runtime axes."""

from bench_sweep import Config
from internal_utils import arg_parse


def cells_from_arg(name: String, default: String) raises -> List[Config]:
    """Returns the cells of `--<name>`, a comma list of `batch:seq:cache`.

    Each becomes a config with axes `batch_size`, `seq_len` and `cache_len`.
    """
    var out = List[Config]()
    for cell in String(arg_parse(name, default)).split(","):
        var trimmed = cell.strip()
        if trimmed.byte_length() == 0:
            continue
        var parts = trimmed.split(":")
        if len(parts) != 3:
            raise Error("cell ", trimmed, " is not batch:seq:cache")
        out.append(
            Config(
                ["batch_size", "seq_len", "cache_len"],
                [Int(parts[0]), Int(parts[1]), Int(parts[2])],
            )
        )
    if len(out) == 0:
        raise Error("--", name, " has no cells")
    return out^


def parse_names(name: String, default: String) raises -> List[String]:
    """Returns `--<name>` split on commas, such as `prefill,decode`."""
    var out = List[String]()
    for part in String(arg_parse(name, default)).split(","):
        var trimmed = part.strip()
        if trimmed.byte_length() > 0:
            out.append(String(trimmed))
    return out^
