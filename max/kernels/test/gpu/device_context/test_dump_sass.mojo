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

import std.subprocess
import std.tempfile
from std.os import setenv, unsetenv
from std.pathlib import Path
from std.sys._assembly import inlined_assembly
from std.testing import assert_raises, assert_true

from max.gpu.host import DeviceContext

comptime ptxas_path = Path("/usr/local/cuda/bin/ptxas")
comptime nvdisasm_path = Path("/usr/local/cuda/bin/nvdisasm")
comptime ptxas_env = "MODULAR_PTXAS_PATH"
comptime nvdisasm_env = "MODULAR_NVDISASM_PATH"


def dump_sass() raises:
    def kernel_inlined_assembly():
        inlined_assembly["nanosleep.u32 $0;", NoneType, constraints="r"](
            UInt32(100)
        )

    with DeviceContext() as ctx:
        _ = ctx.compile_function[kernel_inlined_assembly, _dump_sass=True]()


def test__dump_sass() raises:
    _ = unsetenv(ptxas_env)
    _ = unsetenv(nvdisasm_env)
    dump_sass()


def check_missing_tool(env_var: String) raises:
    _ = setenv(env_var, "/nonexistent/tool")
    try:
        with assert_raises(contains=env_var):
            dump_sass()
    finally:
        _ = unsetenv(env_var)


def test__dump_sass_ptxas_path_missing() raises:
    check_missing_tool(ptxas_env)


def test__dump_sass_nvdisasm_path_missing() raises:
    check_missing_tool(nvdisasm_env)


def check_wrapper_used(env_var: String, real_tool: Path) raises:
    # A tool that records its use and then forwards to the default one.
    with std.tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var marker = Path(tmpdir) / "used"
        var wrapper = Path(tmpdir) / "tool"
        wrapper.write_text(
            String(t"#!/bin/sh\ntouch {marker}\nexec {real_tool} \"$@\"\n")
        )
        _ = std.subprocess.run(String(t"chmod +x {wrapper}"))
        _ = setenv(env_var, String(wrapper))
        try:
            dump_sass()
        finally:
            _ = unsetenv(env_var)
        assert_true(marker.exists(), String(t"{env_var} was not used"))


def test__dump_sass_ptxas_path_override() raises:
    check_wrapper_used(ptxas_env, ptxas_path)


def test__dump_sass_nvdisasm_path_override() raises:
    check_wrapper_used(nvdisasm_env, nvdisasm_path)


def main() raises:
    if ptxas_path.exists() and nvdisasm_path.exists():
        test__dump_sass()
        test__dump_sass_ptxas_path_missing()
        test__dump_sass_nvdisasm_path_missing()
        test__dump_sass_ptxas_path_override()
        test__dump_sass_nvdisasm_path_override()
