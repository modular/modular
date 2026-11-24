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
"""Allocates a real expert-parallel exchange for Kimi K3's eager EP tests."""

from __future__ import annotations

from max import functional as F
from max.driver import CPU, Buffer
from max.engine import InferenceSession
from max.graph import BufferValue
from max.nn import ep
from max.nn.ep import init as ep_init
from max.nn.ep.buffers import EPTables
from max.nn.ep_config import EPConfig
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded


def _host_table(buffer: Buffer) -> Tensor:
    """Wraps one host pointer table, which every device reads, as a constant.

    A served graph takes the tables as graph inputs. Only eager tests bake
    them in.
    """
    with ensure_context():
        return F.constant(buffer.to_numpy(), buffer.dtype, device=CPU())


def allocate_ep_buffers(
    config: EPConfig, mesh: DeviceMesh, session: InferenceSession
) -> ep.EPBuffers:
    """Registers the exchange's SHMEM buffers and wraps them for :mod:`max.nn.ep`.

    Runs the ``ep_init`` graph once. Its two buffer groups become the
    dispatch and combine tables, so combine never writes to buffers a
    dispatch still reads. Both directions synchronize on the first group's
    atomic counters.
    """
    init = ep_init.init_graph(config)
    buffers = ep_init.start(
        config, session.load(init.trace(*init.input_types()))
    ).buffers
    n = mesh.num_devices

    with ensure_context():
        counters = Tensor.from_shard_values(
            [
                BufferValue(
                    Tensor(buf.to_numpy(), dtype=buf.dtype, device=device)
                )
                for buf, device in zip(buffers[:n], mesh.devices, strict=True)
            ],
            DeviceMapping(mesh, (Sharded(axis=0),) * mesh.ndim),
        )
    # The tables follow both groups' counters, type-major: send, receive,
    # then count, each for both groups.
    tables = [_host_table(table) for table in buffers[2 * n : 2 * n + 6]]
    dispatch, combine = (
        EPTables(send=tables[g], recv=tables[2 + g], count=tables[4 + g])
        for g in range(2)
    )
    return ep.EPBuffers(counters=counters, dispatch=dispatch, combine=combine)
