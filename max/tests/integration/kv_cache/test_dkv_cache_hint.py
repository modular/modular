# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Unit tests for _parse_dkv_cache_hint in the tokenizer module."""

import pytest
from max.pipelines.lib.tokenizer import _parse_dkv_cache_hint


class TestParseDkvCacheHint:
    """Tests for _parse_dkv_cache_hint."""

    def test_none_input_returns_none(self) -> None:
        assert _parse_dkv_cache_hint(None) is None

    def test_self_source_returns_none(self) -> None:
        hint = {"source": "self", "blocks": [{"hash": 1, "offset": 0}]}
        assert _parse_dkv_cache_hint(hint) is None

    def test_empty_blocks_returns_none(self) -> None:
        hint = {"source": "dkv-peer", "blocks": []}
        assert _parse_dkv_cache_hint(hint) is None

    def test_basic_blocks_without_agent_info(self) -> None:
        hint = {
            "source": "dkv-peer",
            "blocks": [
                {"hash": 111, "offset": 0, "length": 4096},
                {"hash": 222, "offset": 4096, "length": 4096},
            ],
        }
        result = _parse_dkv_cache_hint(hint)
        assert result is not None
        assert len(result) == 2
        assert 111 in result
        assert 222 in result
        # transfer_engine should be None when no agent_info
        assert result[111].transfer_engine is None
        assert result[222].transfer_engine is None
        assert result[111].offset == 0
        assert result[222].offset == 4096

    def test_blocks_with_agent_info_and_block_size(self) -> None:
        hint = {
            "source": "dkv-peer",
            "block_size": 4096,
            "blocks": [
                {"hash": 111, "offset": 0, "length": 4096},
                {"hash": 222, "offset": 4096, "length": 4096},
                {"hash": 333, "offset": 8192, "length": 4096},
            ],
            "nixl_agent_info": {
                "agent_name": "test-agent",
                "agent_metadata": "",
                "base_addr": 0x1000,
            },
            "source_endpoint": "tcp://10.0.0.1:5556",
        }
        result = _parse_dkv_cache_hint(hint)
        assert result is not None
        assert len(result) == 3
        # transfer_engine should be built
        te = result[111].transfer_engine
        assert te is not None
        assert te.bytes_per_page == 4096
        # total_num_pages derived from max offset: 8192/4096 + 1 = 3
        assert te.total_num_pages == 3
        assert te.hostname == "10.0.0.1"

    def test_block_size_zero_skips_transfer_engine(self) -> None:
        hint = {
            "source": "dkv-peer",
            "block_size": 0,
            "blocks": [{"hash": 42, "offset": 0, "length": 128}],
            "nixl_agent_info": {
                "agent_name": "test-agent",
            },
        }
        result = _parse_dkv_cache_hint(hint)
        assert result is not None
        assert result[42].transfer_engine is None

    def test_localhost_endpoint_uses_gethostname(self) -> None:
        import socket

        hint = {
            "source": "dkv-peer",
            "block_size": 1024,
            "blocks": [{"hash": 1, "offset": 0, "length": 1024}],
            "nixl_agent_info": {
                "agent_name": "local-agent",
                "base_addr": 0,
            },
            "source_endpoint": "tcp://localhost:5556",
        }
        result = _parse_dkv_cache_hint(hint)
        assert result is not None
        te = result[1].transfer_engine
        assert te is not None
        assert te.hostname == socket.gethostname()

    def test_remote_hint_without_block_size_raises(self) -> None:
        hint = {
            "source": "remote_dkv",
            "block_size": 0,
            "blocks": [{"hash": 42, "offset": 0, "length": 128}],
            "nixl_agent_info": {"agent_name": "remote-agent"},
            "source_endpoint": "tcp://10.0.0.2:5556",
        }
        with pytest.raises(ValueError, match="block_size > 0"):
            _parse_dkv_cache_hint(hint)

    def test_malformed_hint_raises(self) -> None:
        # Missing required 'source' field
        with pytest.raises((KeyError, TypeError)):
            _parse_dkv_cache_hint({"blocks": []})

    def test_hash_masked_to_uint64(self) -> None:
        big_hash = (1 << 64) + 42
        hint = {
            "source": "dkv-peer",
            "blocks": [{"hash": big_hash, "offset": 0, "length": 128}],
        }
        result = _parse_dkv_cache_hint(hint)
        assert result is not None
        assert 42 in result  # masked to uint64
