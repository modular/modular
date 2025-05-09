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

import ._c
import ._c.IR
from .ir import Block, Context, Location, Operation, Region, Type, Value


@register_passable
struct Rewriter:
    """
    The Rewriter class implements the RewriterBase class from MLIR, and is both
    replacing the IRRewriter and PatternRewriter MLIR classes.
    """

    alias cType = _c.Rewrite.MlirRewriterBase
    var c: Self.cType

    @implicit
    fn __init__(out self, context: Context):
        self.c = _c.Rewrite.mlirIRRewriterCreate(context.c)

    @implicit
    fn __init__(out self, op: Operation):
        self.c = _c.Rewrite.mlirIRRewriterCreateFromOp(op.c)

    fn __del__(owned self):
        _c.Rewrite.mlirIRRewriterDestroy(self.c)

    fn context(self) -> Context:
        return Context(_c.Rewrite.mlirRewriterBaseGetContext(self.c))

    fn clear_insertion_point(mut self):
        _c.Rewrite.mlirRewriterBaseClearInsertionPoint(self.c)

    fn set_insertion_point_before(mut self, op: Operation):
        _c.Rewrite.mlirRewriterBaseSetInsertionPointBefore(self.c, op.c)

    fn set_insertion_point_after(mut self, op: Operation):
        _c.Rewrite.mlirRewriterBaseSetInsertionPointAfter(self.c, op.c)

    fn set_insertion_point_after(mut self, val: Value):
        _c.Rewrite.mlirRewriterBaseSetInsertionPointAfterValue(self.c, val.c)

    fn set_insertion_point_to_start(mut self, b: Block):
        _c.Rewrite.mlirRewriterBaseSetInsertionPointToStart(self.c, b.c)

    fn set_insertion_point_to_end(mut self, b: Block):
        _c.Rewrite.mlirRewriterBaseSetInsertionPointToEnd(self.c, b.c)

    fn get_insertion_block(self) -> Block:
        return Block(_c.Rewrite.mlirRewriterBaseGetInsertionBlock(self.c))

    fn get_block(self) -> Block:
        return Block(_c.Rewrite.mlirRewriterBaseGetBlock(self.c))

    fn create_block_before(
        mut self,
        insert_before: Block,
        args: List[Type],
        locations: List[Location],
    ) -> Block:
        debug_assert(
            len(args) == len(locations), "Each arg must have a location"
        )
        return Block(
            _c.Rewrite.mlirRewriterBaseCreateBlockBefore(
                self.c,
                insert_before.c,
                len(args),
                args.data.bitcast[Type.cType](),
                locations.data.bitcast[Location.cType](),
            )
        )

    fn insert(mut self, op: Operation) -> Operation:
        return Operation(_c.Rewrite.mlirRewriterBaseInsert(self.c, op.c))

    fn clone(mut self, op: Operation) -> Operation:
        return Operation(_c.Rewrite.mlirRewriterBaseClone(self.c, op.c))

    fn clone_without_regions(mut self, op: Operation) -> Operation:
        return Operation(
            _c.Rewrite.mlirRewriterBaseCloneWithoutRegions(self.c, op.c)
        )

    fn clone_region_before(mut self, region: Region, before: Block):
        _c.Rewrite.mlirRewriterBaseCloneRegionBefore(self.c, region.c, before.c)

    fn inline_region_before(mut self, region: Region, before: Block):
        _c.Rewrite.mlirRewriterBaseInlineRegionBefore(
            self.c, region.c, before.c
        )

    fn replace_op_with(mut self, op: Operation, values: List[Value]):
        _c.Rewrite.mlirRewriterBaseReplaceOpWithValues(
            self.c, op.c, len(values), values.data.bitcast[Value.cType]()
        )

    fn replace_op_with(mut self, op: Operation, new_op: Operation):
        _c.Rewrite.mlirRewriterBaseReplaceOpWithOperation(
            self.c, op.c, new_op.c
        )

    fn erase_op(mut self, op: Operation):
        _c.Rewrite.mlirRewriterBaseEraseOp(self.c, op.c)

    fn erase_block(mut self, block: Block):
        _c.Rewrite.mlirRewriterBaseEraseBlock(self.c, block.c)

    fn inline_block_before(
        mut self, source: Block, op: Operation, arg_values: List[Value]
    ):
        _c.Rewrite.mlirRewriterBaseInlineBlockBefore(
            self.c,
            source.c,
            op.c,
            len(arg_values),
            arg_values.data.bitcast[Value.cType](),
        )

    fn merge_blocks(
        mut self, source: Block, dest: Block, arg_values: List[Value]
    ):
        _c.Rewrite.mlirRewriterBaseMergeBlocks(
            self.c,
            source.c,
            dest.c,
            len(arg_values),
            arg_values.data.bitcast[Value.cType](),
        )

    fn move_op_before(mut self, op: Operation, existing_op: Operation):
        _c.Rewrite.mlirRewriterBaseMoveOpBefore(self.c, op.c, existing_op.c)

    fn move_op_after(mut self, op: Operation, existing_op: Operation):
        _c.Rewrite.mlirRewriterBaseMoveOpAfter(self.c, op.c, existing_op.c)

    fn move_block_before(mut self, block: Block, existing_block: Block):
        _c.Rewrite.mlirRewriterBaseMoveBlockBefore(
            self.c, block.c, existing_block.c
        )

    fn start_op_modification(mut self, op: Operation):
        _c.Rewrite.mlirRewriterBaseStartOpModification(self.c, op.c)

    fn finalize_op_modification(mut self, op: Operation):
        _c.Rewrite.mlirRewriterBaseFinalizeOpModification(self.c, op.c)

    fn cancel_op_modification(mut self, op: Operation):
        _c.Rewrite.mlirRewriterBaseCancelOpModification(self.c, op.c)

    fn replace_all_uses_with(mut self, value: Value, to: Value):
        _c.Rewrite.mlirRewriterBaseReplaceAllUsesWith(self.c, value.c, to.c)

    fn replace_all_uses_with(mut self, values: List[Value], to: List[Value]):
        debug_assert(
            len(values) == len(to),
            (
                "The number of values should be equal to the number of"
                " replacements"
            ),
        )
        _c.Rewrite.mlirRewriterBaseReplaceAllValueRangeUsesWith(
            self.c,
            len(values),
            values.data.bitcast[Value.cType](),
            to.data.bitcast[Value.cType](),
        )

    fn replace_all_uses_with(mut self, op: Operation, to: List[Value]):
        _c.Rewrite.mlirRewriterBaseReplaceAllOpUsesWithValueRange(
            self.c, op.c, len(to), to.data.bitcast[Value.cType]()
        )

    fn replace_all_uses_with(mut self, op: Operation, to: Operation):
        _c.Rewrite.mlirRewriterBaseReplaceAllOpUsesWithOperation(
            self.c, op.c, to.c
        )

    fn replace_op_uses_within_block(
        mut self, op: Operation, new_values: List[Value], block: Block
    ):
        _c.Rewrite.mlirRewriterBaseReplaceOpUsesWithinBlock(
            self.c,
            op.c,
            len(new_values),
            new_values.data.bitcast[Value.cType](),
            block.c,
        )

    fn replace_all_uses_except(
        mut self, val: Value, to: Value, excepted_user: Operation
    ):
        _c.Rewrite.mlirRewriterBaseReplaceAllUsesExcept(
            self.c, val.c, to.c, excepted_user.c
        )
