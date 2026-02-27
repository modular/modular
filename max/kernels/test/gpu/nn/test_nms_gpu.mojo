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


from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from nn.nms_gpu import non_max_suppression_gpu

from utils import IndexList
from gpu.host import DeviceContext


@register_passable("trivial")
struct BoxCoords[dtype: DType]:
    var y1: Scalar[dtype]
    var x1: Scalar[dtype]
    var y2: Scalar[dtype]
    var x2: Scalar[dtype]

    fn __init__(
        out self,
        y1: Scalar[dtype],
        x1: Scalar[dtype],
        y2: Scalar[dtype],
        x2: Scalar[dtype],
    ):
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2


alias unknown_layout_3d = Layout.row_major(
    UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
)


fn fill_boxes[
    dtype: DType
](batch_size: Int, box_list: VariadicList[BoxCoords[dtype]]) -> LayoutTensor[
    dtype, unknown_layout_3d, MutableAnyOrigin
]:
    var num_boxes = len(box_list) // batch_size
    var shape = IndexList[3](batch_size, num_boxes, 4)
    var storage = UnsafePointer[Scalar[dtype]].alloc(shape.flattened_length())
    var boxes = LayoutTensor[dtype, unknown_layout_3d](
        storage.as_any_origin(),
        RuntimeLayout[unknown_layout_3d].row_major(shape),
    )
    for i in range(len(box_list)):
        var coords = linear_offset_to_coords(
            i, IndexList[2](batch_size, num_boxes)
        )
        boxes[coords[0], coords[1], 0] = box_list[i].y1
        boxes[coords[0], coords[1], 1] = box_list[i].x1
        boxes[coords[0], coords[1], 2] = box_list[i].y2
        boxes[coords[0], coords[1], 3] = box_list[i].x2

    return boxes


fn linear_offset_to_coords[
    rank: Int
](idx: Int, shape: IndexList[rank]) -> IndexList[rank]:
    var output = IndexList[rank](0)
    var curr_idx = idx
    for i in reversed(range(rank)):
        output[i] = curr_idx % shape[i]
        curr_idx //= shape[i]

    return output


fn fill_scores[
    dtype: DType
](
    batch_size: Int, num_classes: Int, scores_list: VariadicList[Scalar[dtype]]
) -> LayoutTensor[dtype, unknown_layout_3d, MutableAnyOrigin]:
    var num_boxes = len(scores_list) // batch_size // num_classes

    var shape = IndexList[3](batch_size, num_classes, num_boxes)
    var storage = UnsafePointer[Scalar[dtype]].alloc(shape.flattened_length())
    var scores = LayoutTensor[dtype, unknown_layout_3d](
        storage.as_any_origin(),
        RuntimeLayout[unknown_layout_3d].row_major(shape),
    )
    for i in range(len(scores_list)):
        var coords = linear_offset_to_coords(i, shape)
        scores[coords[0], coords[1], coords[2]] = scores_list[i]

    return scores

fn test_case_gpu[
    dtype: DType
](
    batch_size: Int,
    num_classes: Int,
    num_boxes: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    max_output_boxes_per_class: Int,
    box_list: VariadicList[BoxCoords[dtype]],
    scores_list: VariadicList[Scalar[dtype]],
    ctx: DeviceContext,
) raises:
    var boxes = fill_boxes[dtype](batch_size, box_list)
    var scores = fill_scores[dtype](batch_size, num_classes, scores_list)

    var boxes_len = batch_size * num_boxes * 4
    var scores_len = batch_size * num_classes * num_boxes
    var out_len = batch_size * num_classes * max_output_boxes_per_class * 3

    var boxes_dev = ctx.enqueue_create_buffer[dtype](boxes_len).enqueue_fill(0)
    var scores_dev = ctx.enqueue_create_buffer[dtype](scores_len).enqueue_fill(0)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](out_len).enqueue_fill(-1)

    ctx.enqueue_copy(boxes_dev, boxes.ptr)
    ctx.enqueue_copy(scores_dev, scores.ptr)

    non_max_suppression_gpu[dtype](
        boxes_dev,
        scores_dev,
        out_dev,
        batch_size,
        num_classes,
        num_boxes,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        ctx,
    )
    
    ctx.synchronize()

    with out_dev.map_to_host() as out_host:
        for i in range(out_len // 3):
            var b = out_host[i * 3 + 0]
            var c = out_host[i * 3 + 1]
            var k = out_host[i * 3 + 2]
            if k != -1:
                print(b, ",", c, ",", k)

    boxes.ptr.free()
    scores.ptr.free()

fn main() raises:
    with DeviceContext() as ctx:
        fn test_no_score_threshold(ctx:DeviceContext) raises:
            print("== test_no_score_threshold")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
                BoxCoords[DType.float32](0.0, 0.1, 1.0, 1.1),
                BoxCoords[DType.float32](0.0, -0.1, 1.0, 0.9),
                BoxCoords[DType.float32](0.0, 10.0, 1.0, 11.0),
                BoxCoords[DType.float32](0.0, 10.1, 1.0, 11.1),
                BoxCoords[DType.float32](0.0, 100.0, 1.0, 101.0),
            )
            var scores_list = VariadicList[Float32](0.9, 0.75, 0.6, 0.95, 0.5, 0.3)

            test_case_gpu[DType.float32](
                1, 1, 6, Float32(0.5), Float32(0.0), 3, box_list, scores_list, ctx
            )

        fn test_flipped_coords() raises:
            print("== test_flipped_coords")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](1.0, 1.0, 0.0, 0.0),
                BoxCoords[DType.float32](1.0, 1.1, 0.0, 0.1),
                BoxCoords[DType.float32](1.0, 0.9, 0.0, -0.1),
                BoxCoords[DType.float32](1.0, 11.0, 0.0, 10.0),
                BoxCoords[DType.float32](1.0, 11.1, 0.0, 10.1),
                BoxCoords[DType.float32](1.0, 101.0, 0.0, 100.0),
            )
            var scores_list = VariadicList[Float32](0.9, 0.75, 0.6, 0.95, 0.5, 0.3)

            test_case_gpu[DType.float32](
                1, 1, 6, Float32(0.5), Float32(0.0), 3, box_list, scores_list, ctx
            )

        fn test_reflect_over_yx() raises:
            print("== test_reflect_over_yx")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](-1.0, -1.0, 0.0, 0.0),
                BoxCoords[DType.float32](-1.0, -1.1, 0.0, -0.1),
                BoxCoords[DType.float32](-1.0, -0.9, 0.0, 0.1),
                BoxCoords[DType.float32](-1.0, -11.0, 0.0, -10.0),
                BoxCoords[DType.float32](-1.0, -11.1, 0.0, -10.1),
                BoxCoords[DType.float32](-1.0, -101.0, 0.0, -100.0),
            )
            var scores_list = VariadicList[Float32](0.9, 0.75, 0.6, 0.95, 0.5, 0.3)

            test_case_gpu[DType.float32](
                1, 1, 6, Float32(0.5), Float32(0.0), 3, box_list, scores_list, ctx
            )

        fn test_score_threshold() raises:
            print("== test_score_threshold")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
                BoxCoords[DType.float32](0.0, 0.1, 1.0, 1.1),
                BoxCoords[DType.float32](0.0, -0.1, 1.0, 0.9),
                BoxCoords[DType.float32](0.0, 10.0, 1.0, 11.0),
                BoxCoords[DType.float32](0.0, 10.1, 1.0, 11.1),
                BoxCoords[DType.float32](0.0, 100.0, 1.0, 101.0),
            )
            var scores_list = VariadicList[Float32](0.9, 0.75, 0.6, 0.95, 0.5, 0.3)

            test_case_gpu[DType.float32](
                1, 1, 6, Float32(0.5), Float32(0.4), 3, box_list, scores_list, ctx
            )

        fn test_limit_outputs() raises:
            print("== test_limit_outputs")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
                BoxCoords[DType.float32](0.0, 0.1, 1.0, 1.1),
                BoxCoords[DType.float32](0.0, -0.1, 1.0, 0.9),
                BoxCoords[DType.float32](0.0, 10.0, 1.0, 11.0),
                BoxCoords[DType.float32](0.0, 10.1, 1.0, 11.1),
                BoxCoords[DType.float32](0.0, 100.0, 1.0, 101.0),
            )
            var scores_list = VariadicList[Float32](0.9, 0.75, 0.6, 0.95, 0.5, 0.3)

            test_case_gpu[DType.float32](
                1, 1, 6, Float32(0.5), Float32(0.0), 2, box_list, scores_list, ctx
            )

        fn test_single_box() raises:
            print("== test_single_box")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
            )
            var scores_list = VariadicList[Float32](0.9)

            test_case_gpu[DType.float32](
                1, 1, 1, Float32(0.5), Float32(0.0), 2, box_list, scores_list, ctx
            )

        fn test_two_classes() raises:
            print("== test_two_classes")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
                BoxCoords[DType.float32](0.0, 0.1, 1.0, 1.1),
                BoxCoords[DType.float32](0.0, -0.1, 1.0, 0.9),
                BoxCoords[DType.float32](0.0, 10.0, 1.0, 11.0),
                BoxCoords[DType.float32](0.0, 10.1, 1.0, 11.1),
                BoxCoords[DType.float32](0.0, 100.0, 1.0, 101.0),
            )
            var scores_list = VariadicList[Float32](
                0.9,
                0.75,
                0.6,
                0.95,
                0.5,
                0.3,
                0.9,
                0.75,
                0.6,
                0.95,
                0.5,
                0.3,
            )

            test_case_gpu[DType.float32](
                1, 2, 6, Float32(0.5), Float32(0.0), 2, box_list, scores_list, ctx
            )

        fn test_two_batches() raises:
            print("== test_two_batches")
            var box_list = VariadicList[BoxCoords[DType.float32]](
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
                BoxCoords[DType.float32](0.0, 0.1, 1.0, 1.1),
                BoxCoords[DType.float32](0.0, -0.1, 1.0, 0.9),
                BoxCoords[DType.float32](0.0, 10.0, 1.0, 11.0),
                BoxCoords[DType.float32](0.0, 10.1, 1.0, 11.1),
                BoxCoords[DType.float32](0.0, 100.0, 1.0, 101.0),
                BoxCoords[DType.float32](0.0, 0.0, 1.0, 1.0),
                BoxCoords[DType.float32](0.0, 0.1, 1.0, 1.1),
                BoxCoords[DType.float32](0.0, -0.1, 1.0, 0.9),
                BoxCoords[DType.float32](0.0, 10.0, 1.0, 11.0),
                BoxCoords[DType.float32](0.0, 10.1, 1.0, 11.1),
                BoxCoords[DType.float32](0.0, 100.0, 1.0, 101.0),
            )
            var scores_list = VariadicList[Float32](
                0.9,
                0.75,
                0.6,
                0.95,
                0.5,
                0.3,
                0.9,
                0.75,
                0.6,
                0.95,
                0.5,
                0.3,
            )

            test_case_gpu[DType.float32](
                2, 1, 6, Float32(0.5), Float32(0.0), 2, box_list, scores_list, ctx
            )

        # CHECK-LABEL: == test_no_score_threshold
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        # CHECK-NEXT: 0,0,5,
        test_no_score_threshold(ctx)

        # CHECK-LABEL: == test_flipped_coords
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        # CHECK-NEXT: 0,0,5,
        test_flipped_coords()

        # CHECK-LABEL: == test_reflect_over_yx
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        # CHECK-NEXT: 0,0,5,
        test_reflect_over_yx()

        # CHECK-LABEL: == test_score_threshold
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        test_score_threshold()

        # CHECK-LABEL: == test_limit_outputs
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        test_limit_outputs()

        # CHECK-LABEL: == test_single_box
        # CHECK: 0,0,0,
        test_single_box()

        # CHECK-LABEL: == test_two_classes
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        # CHECK-NEXT: 0,1,3,
        # CHECK-NEXT: 0,1,0,
        test_two_classes()

        # CHECK-LABEL: == test_two_batches
        # CHECK: 0,0,3,
        # CHECK-NEXT: 0,0,0,
        # CHECK-NEXT: 1,0,3,
        # CHECK-NEXT: 1,0,0,
        test_two_batches()
