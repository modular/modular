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
"""Provides ULP comparisons, libm references and inputs for testing math
functions on the host and on GPUs."""

from .compare import compare
from .lgamma import lgamma_bit_refs, lgamma_zeros_f32, lgamma_zeros_f64
from .libm import libm_call, libm_call_binary, libm_call_f64, libm_scalb
from .log10 import (
    f32_max_subnormal_bits,
    f32_min_subnormal_bits,
    f64_max_subnormal_bits,
    f64_min_subnormal_bits,
    log10_f32_inputs,
    log10_f64_inputs,
    num_log10_inputs,
    num_log10_specials,
)
from .pair_inputs import (
    PairMode,
    hypot_specials,
    pair_inputs,
    remainder_specials,
    scalb_specials,
)
from .rng import SplitMix64, xorshift64
from .ulp import ulp_diff, ulp_distance, within_two_ulp
