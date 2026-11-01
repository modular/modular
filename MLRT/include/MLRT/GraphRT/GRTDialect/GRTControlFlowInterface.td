//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_TD

include "mlir/IR/OpBase.td"

def GRTControlFlowInterface : OpInterface<"GRTControlFlowInterface"> {
  let description = [{
    This interface describes common properties of GRT control flow operators.
  }];

  let cppNamespace = "::M::GRT";

  let methods = [
    InterfaceMethod<
      [{
        Returns the N control flow op 'input' operands which are implicitly
        passed as the first N operands to each of the op's control flow blocks.
        The control flow op may take additional operands before these inputs
        to control its operation, such as a condition or trip count.
      }],
      "::mlir::Operation::operand_range", "getInputs", (ins)
    >,
    InterfaceMethod<
      [{
        As for getInputs, but returned as an array of OpOperands for use
        by dense, backwards dataflow analyses.
      }],
      "llvm::ArrayRef<::mlir::OpOperand>", "getInputOpOperands", (ins)
    >,
    InterfaceMethod<
      [{
        Returns any operands, beyond those covered by getInputs, which are
        consumed strictly by the control flow operation, such as a condition
        or trip count.
      }],
      "::mlir::Operation::operand_range", "getStrictExtraOperands", (ins)
    >,
    InterfaceMethod<
      [{
        As for getStrictExtraOperands, but returned as an array of OpOperands
        for use by dense, backwards dataflow analyses.
      }],
      "::llvm::ArrayRef<::mlir::OpOperand>", "getStrictExtraOpOperands", (ins)
    >,
    InterfaceMethod<
      [{
        Returns the 'flavor' of the control flow op.

        For a `kLoopLike` op (eg grt.{while, repeat.i32}):
         - The op inputs match each sub-block's arguments.
         - If one sub-block, it is assumed to be executed 0 or more times.
         - If two sub-blocks, the first sub-block is considered a condition,
           and may exit the loop. The second sub-block is considered the body,
           and is only ever run after the condition.
         - The first N results of each of the op's control flow blocks
           represent updated versions of the N op inputs. Blocks may return
           additional outputs, such as a condition value.
         - The updated values are chained from the op inputs through each
           of the sub-blocks and back to the inputs or the control flow op's
           results.

        For a 'kBranchLike` op (eg grt.if):
         - The op inputs match each sub-block's arguments.
         - Exactly one of the sub-blocks is executed.
         - All the sub-blocks outputs agree with the control flow op results.
      }],
      "M::GRT::ControlFlowFlavor", "getFlavor", (ins)
    >,
    InterfaceMethod<
      [{
        Returns the human-readable names for each of the control flow
        ops nested regions.
      }],
      "llvm::SmallVector<llvm::StringRef>", "getRegionNames", (ins)
    >,
  ];

  let verify = [{
    return M::GRT::verifyControlFlowInterfaceOp($_op);
  }];
}

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_TD
