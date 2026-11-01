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
//
// Interfaces for MAP ops.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def MAP_CapturingFusionClosure : OpInterface<"CapturingFusionClosure"> {
  let description = [{
    Common interface for the three closure kinds a `map.iter.opaque` (or a
    to-be-outlined `map.iter.elementwise`) holds: `map.iter.elementwise.body`,
    `map.iter.prologue`, and `map.iter.epilogue`. Each is a single-block,
    isolated-from-above region reaching every outer value through a
    `$captures` operand, shadowed as the block's trailing arguments.
  }];

  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the closure's single-block body region.",
      "::mlir::Region &", "getBody", (ins),
      "return $_op.getBody();"
    >,
    InterfaceMethod<
      "Returns the outer-scope values this closure captures.",
      "::mlir::ValueRange", "getCaptures", (ins),
      "return $_op.getCaptures();"
    >,
    InterfaceMethod<
      "Returns the body block arguments shadowing getCaptures(), one to "
      "one, in the same order -- always the trailing block arguments.",
      "::mlir::Block::BlockArgListType", "getCaptureBlockArgs", (ins),
      [{
        return $_op.getBody().front().getArguments().take_back(
            $_op.getCaptures().size());
      }]
    >,
    InterfaceMethod<
      "Returns the outer-scope value `arg` shadows, or null when `arg` is not "
      "one of this closure's capture block args (e.g. the index, or an "
      "epilogue's stored value). The inverse of getCaptureBlockArgs().",
      "::mlir::Value", "getCaptureFor", (ins "::mlir::BlockArgument":$arg),
      [{
        // Reached through the interface, not `$_op`: a default body is emitted
        // per concrete op, and `getCaptureBlockArgs` is itself an interface
        // default rather than a member of `IterElementwiseBodyOp` & co.
        auto closure =
            ::mlir::cast<::M::MAP::CapturingFusionClosure>($_op.getOperation());
        ::mlir::Block::BlockArgListType captureArgs =
            closure.getCaptureBlockArgs();
        for (unsigned i = 0, e = captureArgs.size(); i < e; ++i)
          if (captureArgs[i] == arg)
            return closure.getCaptures()[i];
        return {};
      }]
    >,
    InterfaceMethod<
      "Appends `capture` to the capture operands WITHOUT adding a shadowing "
      "block argument, segment-aware (e.g. the epilogue's trailing "
      "`$intermediateTensor`). Use when the block argument is added "
      "separately; `addCapture` adds both.",
      "void", "appendCaptureOperand", (ins "::mlir::Value":$capture),
      [{
        $_op.getCapturesMutable().append(capture);
      }]
    >,
    InterfaceMethod<
      "Appends `capture` as a new capture and returns its new shadowing "
      "block argument. Does not dedup; the capturing-closure canonicalizer "
      "does.",
      "::mlir::BlockArgument", "addCapture", (ins "::mlir::Value":$capture),
      [{
        ::mlir::BlockArgument arg = $_op.getBody().front().addArgument(
            capture.getType(), capture.getLoc());
        // Same segment-aware append as `appendCaptureOperand`; inlined here
        // because a default method cannot call another via `$_op` (a
        // default-only method is not a member of the concrete op).
        $_op.getCapturesMutable().append(capture);
        return arg;
      }]
    >,
    InterfaceMethod<
      "Erases the capture at `idx` along with its shadowing block argument.",
      "void", "eraseCaptureAt", (ins "unsigned":$idx),
      [{
        ::mlir::Block &body = $_op.getBody().front();
        unsigned offset = body.getNumArguments() - $_op.getCaptures().size();
        body.eraseArgument(offset + idx);
        $_op.getCapturesMutable().erase(idx);
      }]
    >,
  ];
}

def MAP_IterOpInterface : OpInterface<"IterOpInterface"> {
  let description = [{
    A MAP kernel: `map.iter.elementwise`, `map.iter.rowwise` or `map.iter.opaque`.
    Its body is isolated from above and starts with one block argument per
    input, followed by one per output.
  }];

  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the non-chain inputs.",
      "::mlir::OperandRange", "getInputs", (ins),
      "return $_op.getInputs();"
    >,
    InterfaceMethod<
      "Returns the body block arguments shadowing getInputs(), one to one.",
      "::mlir::Block::BlockArgListType", "getInputBlockArgs", (ins),
      "return $_op.getInputBlockArgs();"
    >,
    InterfaceMethod<
      "Returns the body block arguments of the outputs.",
      "::mlir::Block::BlockArgListType", "getOutputBlockArgs", (ins),
      "return $_op.getOutputBlockArgs();"
    >,
    InterfaceMethod<
      "Inserts `vals` as inputs at `idx`, with their block arguments.",
      "void", "insertInputsAt",
      (ins "unsigned":$idx, "::mlir::ValueRange":$vals),
      "$_op.insertInputsAt(idx, vals);"
    >,
    InterfaceMethod<
      "Erases the inputs in `[idx, idx + len)`, with their block arguments.",
      "void", "eraseInputsAt", (ins "unsigned":$idx, "size_t":$len),
      "$_op.eraseInputsAt(idx, len);"
    >,
    InterfaceMethod<
      "Returns the input chain, or null.",
      "::mlir::Value", "getInputChain", (ins),
      "return $_op.getInputChain();"
    >,
    InterfaceMethod<
      "Returns the output chain, or null.",
      "::mlir::Value", "getOutputChain", (ins),
      "return $_op.getOutputChain();"
    >,
    InterfaceMethod<
      "Sets the input chain, adding the output chain if there was none.",
      "void", "setChain", (ins "::mlir::Value":$chain),
      "$_op.setChain(chain);"
    >,
    InterfaceMethod<
      "Returns the fusion provenance, or null.",
      "::mlir::Attribute", "getFusionInfoAttr", (ins),
      "return $_op.getFusionInfoAttr();"
    >,
    InterfaceMethod<
      "Sets the fusion provenance.",
      "void", "setFusionInfoAttr", (ins "::M::MOGG::FusionInfoAttr":$info),
      "$_op.setFusionInfoAttr(info);"
    >,
    InterfaceMethod<
      "Returns whether the kernel reads memory.",
      "bool", "getHasReadEffect", (ins),
      "return $_op.getHasReadEffect();"
    >,
    InterfaceMethod<
      "Returns whether the kernel writes memory.",
      "bool", "getHasWriteEffect", (ins),
      "return $_op.getHasWriteEffect();"
    >,
    InterfaceMethod<
      "Returns how the runtime schedules the kernel.",
      "::M::MOGG::DispatchKind", "getDispatch", (ins),
      "return $_op.getDispatch();"
    >,
  ];

  let extraClassDeclaration = [{
    /// Merges duplicate inputs onto the first one's block argument and drops
    /// the rest. Returns whether anything changed.
    bool dedupeInputs(::mlir::RewriterBase &rewriter);
  }];
}

def MAP_StructuredIterOpInterface : OpInterface<"StructuredIterOpInterface"> {
  let description = [{
    A MAP kernel that runs its body over a shape: `map.iter.elementwise` or
    `map.iter.rowwise`. Its last body block argument is the index.
  }];

  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the shape the body runs over.",
      "::M::MOSH::ShapeAttr", "getShape", (ins),
      "return $_op.getShape();"
    >,
    InterfaceMethod<
      "Returns the index block argument.",
      "::mlir::BlockArgument", "getIndexArg", (ins), [{}],
      "return $_op.getBody().front().getArguments().back();"
    >,
    InterfaceMethod<
      "Creates the closure that holds this kernel's body once outlined, with "
      "`captures` as its captures and an empty body.",
      "::M::MAP::CapturingFusionClosure", "createBodyClosure",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
           "::mlir::ValueRange":$captures)
    >,
  ];
}

def MAP_IndexTransformOpInterface : OpInterface<"IndexTransformOpInterface"> {
  let description = [{
    Common interface for ops that project a structured loop's own index into
    an index addressing a view of one of the loop's inputs (today:
    `map.index.slice`).
  }];
  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the index operand this op transforms.",
      "::mlir::Value", "getIndex", (ins)
    >,
    InterfaceMethod<
      "Materializes this transform's hoisted strided view: a tensor of this "
      "transform's output shape that a load addresses with the loop's own "
      "untransformed index. `tensor` is the original load target or, when "
      "chained, an outer hop's hoisted view. This op still lives inside the "
      "closure, so its other operands (e.g. a slice's starts/stops/steps) may "
      "be capture block arguments; map each through `outerValues` before "
      "use.",
      "::mlir::Value", "createStridedView",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$tensor,
           "const ::mlir::IRMapping &":$outerValues)
    >,
  ];
}

def MAP_DynamicShapeIndexOpInterface
    : OpInterface<"DynamicShapeIndexOpInterface"> {
  let description = [{
    Common interface for the index ops in a functor body (a
    `map.iter.prologue` or `map.iter.epilogue`) that read shapes at runtime
    when their spec is dynamic -- today `map.index.reshape` and
    `map.index.slice.select`. The functor's lowering builds those shapes in
    kernel scope and threads them in as shape captures, without knowing the
    concrete op kind.
  }];
  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the shapes this op reads at runtime, in the order its lowering "
      "takes them as `$dynamicShapes`, or none when every shape is static.",
      "::llvm::SmallVector<::M::MOSH::ShapeAttr>", "getDynamicShapes", (ins)
    >,
  ];
}

def MAP_StrideViewOpInterface : OpInterface<"StrideViewOpInterface"> {
  let description = [{
    Common interface for the hoisted stride views -- `map.index.slice.stride`,
    `map.index.broadcast.stride`, `map.index.reshape.stride`,
    `map.index.transpose.stride` -- the form
    `IndexTransformOpInterface::createStridedView` materializes. Each produces
    a tensor of its view's OUTPUT shape over storage it does not own.
  }];
  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the tensor this view re-addresses. Nothing is copied, so the "
      "result holds the same elements as the view, only differently indexed.",
      "::mlir::Value", "getViewedTensor", (ins),
      "return $_op.getInput();"
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD
