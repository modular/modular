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
"""Tests for _format_config_entries helper."""

from __future__ import annotations

from max.pipelines.lib.config import _format_config_entries


class TestFormatConfigEntries:
    """Test suite for the _format_config_entries helper."""

    def test_colons_are_aligned(self) -> None:
        """All colons should appear at the same column position."""
        entries = [
            ("model", "modularai/Llama-3.1-8B"),
            ("architecture", "LlamaArch"),
            ("pipeline", "TextGenerationPipeline"),
            ("devices", "gpu[0]"),
            ("max_batch_size", 32),
            ("max_seq_len", 4096),
            ("cache_memory", "8.00 GiB"),
            ("device_graph_capture", True),
        ]

        lines = _format_config_entries(entries)

        colon_positions = [line.index(" : ") for line in lines]
        assert len(set(colon_positions)) == 1, (
            f"Colons are not aligned; positions: {colon_positions}"
        )

    def test_longest_key_determines_alignment(self) -> None:
        """The longest key should have no extra padding."""
        entries = [
            ("a", 1),
            ("medium_key", 2),
            ("this_is_the_longest_key", 3),
        ]

        lines = _format_config_entries(entries)

        # The longest key line should have exactly one space before ":"
        longest_line = lines[2]
        assert "this_is_the_longest_key : 3" in longest_line

        # Shorter keys should be padded to match
        assert "a                       : 1" in lines[0]
        assert "medium_key              : 2" in lines[1]

    def test_default_indent(self) -> None:
        """Lines should be indented with 4 spaces by default."""
        entries = [("key", "value")]

        lines = _format_config_entries(entries)

        assert lines[0].startswith("    ")

    def test_custom_indent(self) -> None:
        """Custom indent prefix should be applied."""
        entries = [("key", "value")]

        lines = _format_config_entries(entries, indent="  >> ")

        assert lines[0].startswith("  >> ")

    def test_single_entry(self) -> None:
        """A single entry should format correctly."""
        entries = [("model", "test-model")]

        lines = _format_config_entries(entries)

        assert len(lines) == 1
        assert lines[0] == "    model : test-model"

    def test_values_are_preserved(self) -> None:
        """Non-string values should appear in their str() form."""
        entries = [
            ("count", 42),
            ("enabled", True),
            ("ratio", 3.14),
        ]

        lines = _format_config_entries(entries)

        assert "42" in lines[0]
        assert "True" in lines[1]
        assert "3.14" in lines[2]

    def test_matches_real_config_shape(self) -> None:
        """Verify alignment with the actual config entry keys used in
        log_basic_config."""
        # These are the real keys from PipelineConfig.log_basic_config
        entries = [
            ("model", "modularai/Llama-3.1-8B-Instruct-GGUF"),
            ("architecture", "LlamaForCausalLM"),
            ("pipeline", "TextGenerationPipeline"),
            ("devices", "gpu[0]"),
            ("max_batch_size", 1),
            ("max_seq_len", 4096),
            ("cache_memory", "7.54 GiB"),
            ("device_graph_capture", True),
        ]

        lines = _format_config_entries(entries)

        # "device_graph_capture" is the longest key (20 chars), all
        # colons must line up at the same column.
        colon_col = lines[0].index(" : ")
        for i, line in enumerate(lines):
            assert line.index(" : ") == colon_col, (
                f"Line {i} colon at {line.index(' : ')}, expected {colon_col}:\n"
                + "\n".join(lines)
            )
