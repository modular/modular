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
"""Tests when ``configure_session`` compiles out the vendor-BLAS fallback.

The matmul dispatcher reads ``MODULAR_DISABLE_VENDOR_FALLBACK`` as a
kernel-compile define, so the environment variable only takes effect because
``configure_session`` forwards it. On HIP with device graph capture the define
is forced on, since the fallback's hipBLASLt handle creation exits the process
inside a capture region.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from max.pipelines.lib import PipelineConfig, PipelineRuntimeConfig

_DEFINE = "MODULAR_DISABLE_VENDOR_FALLBACK"


def _defines_passed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api: str,
    device_graph_capture: bool | None,
    env: str | None,
) -> list[tuple[object, ...]]:
    """Runs ``configure_session`` and returns its ``_set_mojo_define`` args."""
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api", lambda: api
    )
    monkeypatch.delenv("ENABLE_BLASST", raising=False)
    if env is None:
        monkeypatch.delenv(_DEFINE, raising=False)
    else:
        monkeypatch.setenv(_DEFINE, env)

    config = PipelineConfig.model_construct(
        runtime=PipelineRuntimeConfig.model_construct(
            device_graph_capture=device_graph_capture
        )
    )
    session = MagicMock()
    config.configure_session(session)
    return [c.args for c in session._set_mojo_define.call_args_list]


@pytest.mark.parametrize(
    ("api", "device_graph_capture", "env", "expect_define"),
    [
        pytest.param("hip", True, None, True, id="hip-capture-forces-it"),
        pytest.param("hip", False, None, False, id="hip-eager-keeps-it"),
        pytest.param("hip", None, None, False, id="hip-unresolved-capture"),
        pytest.param("cuda", True, None, False, id="cuda-capture-keeps-it"),
        pytest.param("cuda", False, "1", True, id="env-on-other-backend"),
        pytest.param("cuda", False, "true", True, id="env-true-spelling"),
        pytest.param("cuda", False, "0", False, id="env-explicitly-off"),
        pytest.param("cuda", False, None, False, id="default-off"),
    ],
)
def test_vendor_fallback_define(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    device_graph_capture: bool | None,
    env: str | None,
    expect_define: bool,
) -> None:
    defines = _defines_passed(
        monkeypatch,
        api=api,
        device_graph_capture=device_graph_capture,
        env=env,
    )
    assert ((_DEFINE, "true") in defines) is expect_define
