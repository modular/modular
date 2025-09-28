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


fn reference_ptx[M: Int, N: Int, K: Int]() raises -> String:
    # When running with Bazel, the PTX files are in the runfiles directory
    # The path is relative to the test binary's location
    var pwd = "open-source/max/max/kernels/test/testdata/"
    # FIXME Handle differences in internal vs external location
    var oss_pwd = "max/kernels/test/testdata/"

    try:
        var fname = String(pwd, "matmul_sm100_", M, "x", N, "x", K, ".ptx")
        with open(fname, "r") as f:
            return f.read()
    except:
        var fname = String(oss_pwd, "matmul_sm100_", M, "x", N, "x", K, ".ptx")
        with open(fname, "r") as f:
            return f.read()
