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


@extern("nvshmem_int_p")
fn nvshmem_int_p(destination: UnsafePointer[c_int], mype: c_int, peer: c_int):
    ...


@extern("nvshmem_int_g")
fn nvshmem_int_g(destination: UnsafePointer[c_int], mype: c_int, peer: c_int):
    ...


@extern("nvshmem_barrier_all")
fn nvshmem_barrier_all():
    ...


@extern("nvshmem_my_pe")
fn nvshmem_my_pe() -> c_int:
    ...


@extern("nvshmem_n_pes")
fn nvshmem_n_pes() -> c_int:
    ...
