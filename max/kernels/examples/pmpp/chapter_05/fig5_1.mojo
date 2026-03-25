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

# Figure 5.1: Matrix multiplication inner loop (from PMPP Chapter 5)
# Shows the basic computation pattern for matrix multiplication

# ========================== KERNEL CODE ==========================

# This is a code snippet showing the inner loop for matrix multiplication:
# for k in range(Width):
#     Pvalue += M[row * Width + k] * N[k * Width + col]
#
# Where:
# - M and N are input matrices
# - Pvalue accumulates the dot product
# - row and col are the output element coordinates
# - Width is the matrix dimension
