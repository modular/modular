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

"""Distributed functional API — sharding-aware per-op dispatch.

Provides distributed versions of standard tensor operations that
propagate sharding placements across a ``DeviceMesh``.  Each op wraps
the corresponding ``max.graph.ops`` primitive with placement resolution
and redistribution logic.

Usage::

    from max.experimental import distributed_functional as DF

    y = DF.matmul(a, b)
    z = DF.add(x, y)
    w = DF.all_reduce_sum(z)
"""

# Collectives + materialization
from .collectives import (
    all_gather,
    all_reduce_sum,
    auto_reduce_partial,
    materialize,
    reduce_scatter,
    resolve_partials,
    shard,
    to_numpy,
)

# Creation
from .creation import full, ones, zeros
from .elementwise import (
    abs,
    add,
    cast,
    cos,
    div,
    exp,
    gelu,
    log,
    mod,
    mul,
    negate,
    pow,
    relu,
    rsqrt,
    sigmoid,
    silu,
    sin,
    sqrt,
    sub,
    tanh,
)

# Linear algebra
from .matmul import matmul

# Random
from .random import gaussian, gaussian_like, uniform, uniform_like

# Reductions
from .reduction import (
    mean,
    softmax,
    sum,
)

# Shape
from .shape import (
    concat,
    gather,
    permute,
    reshape,
    split,
    squeeze,
    transpose,
    unsqueeze,
)
