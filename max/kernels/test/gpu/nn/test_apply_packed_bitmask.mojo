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

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.gather_scatter import apply_packed_bitmask


def test_apply_packed_bitmask(ctx: DeviceContext) raises:
    # batch=2, vocab=40 -> packed_vocab = ceil(40 / 32) = 2 words/row.
    comptime batch = 2
    comptime vocab = 40
    comptime packed_vocab = 2
    comptime fill_value: Float32 = -10000.0

    # logits[b, v] = b * 100 + v, so every kept value is unique and checkable.
    var logits_stack = InlineArray[Float32, batch * vocab](uninitialized=True)
    var logits = TileTensor(logits_stack, row_major[batch, vocab]())
    for b in range(batch):
        for v in range(vocab):
            logits[b, v] = Float32(b * 100 + v)

    # Build a packed bitmask by setting an explicit set of valid tokens, then
    # derive the expected masked logits from the same source of truth.
    var valid = InlineArray[Bool, batch * vocab](fill=False)
    var packed_stack = InlineArray[Int32, batch * packed_vocab](fill=0)
    var packed = TileTensor(packed_stack, row_major[batch, packed_vocab]())

    # Row 0: a spread of tokens incl. ones that cross the 32-bit word boundary.
    for v in [0, 1, 31, 32, 33, 39]:
        valid[0 * vocab + v] = True
    # Row 1: a different set, including the very last token.
    for v in [5, 7, 30, 38, 39]:
        valid[1 * vocab + v] = True

    for b in range(batch):
        for v in range(vocab):
            if valid[b * vocab + v]:
                packed[b, v >> 5] |= Int32(1) << Int32(v & 31)

    # Copy inputs to device.
    var logits_gpu_buf = ctx.enqueue_create_buffer[DType.float32](batch * vocab)
    ctx.enqueue_copy(logits_gpu_buf, logits.ptr)
    var logits_gpu = TileTensor(logits_gpu_buf, row_major[batch, vocab]())

    var packed_gpu_buf = ctx.enqueue_create_buffer[DType.int32](
        batch * packed_vocab
    )
    ctx.enqueue_copy(packed_gpu_buf, packed.ptr)
    var packed_gpu = TileTensor(
        packed_gpu_buf, row_major[batch, packed_vocab]()
    )

    var out_gpu_buf = ctx.enqueue_create_buffer[DType.float32](batch * vocab)
    var out_gpu = TileTensor(out_gpu_buf, row_major[batch, vocab]())

    apply_packed_bitmask[target="gpu"](
        out_gpu, logits_gpu, packed_gpu, fill_value, ctx
    )

    var out_stack = InlineArray[Float32, batch * vocab](uninitialized=True)
    ctx.enqueue_copy(Span(out_stack), out_gpu_buf)
    var out = TileTensor(out_stack, row_major[batch, vocab]())

    for b in range(batch):
        for v in range(vocab):
            var expected = Float32(b * 100 + v) if valid[
                b * vocab + v
            ] else fill_value
            if out[b, v] != expected:
                raise Error(
                    "out[",
                    b,
                    ", ",
                    v,
                    "] = ",
                    out[b, v],
                    " != ",
                    expected,
                )


def main() raises:
    with DeviceContext() as ctx:
        test_apply_packed_bitmask(ctx)
