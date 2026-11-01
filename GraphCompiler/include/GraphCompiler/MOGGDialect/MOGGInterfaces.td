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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_TD

include "mlir/IR/OpBase.td"

def MOGG_KernelCallInterface : OpInterface<"KernelCallInterface"> {
  let description = [{
    An interface that represents ops that call into Mojo kernels directly.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Inject a dictionary attribute for kernel parameters",
      "void", "setKernelParams", (ins "mlir::DictionaryAttr":$params),
      "$_op.setKernelParamAttr(params);"
    >,
    InterfaceMethod<
      "Returns the value that represents the lowered KGEN mojo object",
      "mlir::DictionaryAttr", "getKernelParams", (ins),
      "return $_op.getKernelParamAttr();"
    >,
    InterfaceMethod<
      "Returns the sym_name of the kernel decl this op references.",
      "mlir::StringAttr", "getKernelSymNameAttr", (ins),
      "return $_op.getKernelSymNameAttr();"
    >,
  ];
}

def MOGG_FuncCallOpInterface : OpInterface<"FuncCallOpInterface"> {
  let description = [{
    Ops that call an outlined `mogg.func` by symbol (`mogg.call`,
    `mogg.call.async`). The callee is the `$name` attribute.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Returns the symbol reference of the callee kernel stub.",
      "mlir::FlatSymbolRefAttr", "getCalleeSymbol", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]
    >,
    InterfaceMethod<
      "Sets the symbol reference of the callee kernel stub.",
      "void", "setCalleeSymbol", (ins "mlir::FlatSymbolRefAttr":$callee),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ $_op.setNameAttr(callee); }]
    >,
  ];
}

def MOGG_TensorCreateInterface : OpInterface<"TensorCreateInterface"> {
  let description = [{
    Common interface for the `mogg.tensor.create*` ops (plain and each view
    kind), which all carry `$elementType`/`$mut`.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Returns the element type of the produced tensor.",
      "::mlir::Type", "getElementType", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getElementType(); }]
    >,
    InterfaceMethod<
      "Returns whether the produced tensor is mutable.",
      "bool", "getMut", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getMut(); }]
    >,
    InterfaceMethod<
      "Sets whether the produced tensor is mutable.",
      "void", "setMut", (ins "bool":$mut), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ $_op.setMut(mut); }]
    >,
    InterfaceMethod<
      [{Clone this create op with a new mutability, adjusting the result kind
        (`!mogg.tensor` when made immutable) so the buffer-implies-mutable
        invariant holds. Returns the clone; the original is left for DCE.}],
      "::M::MOGG::TensorCreateInterface", "cloneWithMut",
      (ins "::mlir::OpBuilder &":$builder, "bool":$mut), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        auto cloned = ::mlir::cast<ConcreteOp>(builder.clone(*$_op.getOperation()));
        // Unlike a raw `setAttr`, concrete `setMut` writes the property that
        // `getMut` reads, and fails to compile if an op drops it.
        cloned.setMut(mut);
        ::mlir::Value result = cloned->getResult(0);
        if (!mut && ::mlir::isa<::M::MOGG::BufferType>(result.getType()))
          result.setType(::M::MOGG::TensorType::get(cloned->getContext()));
        return ::mlir::cast<::M::MOGG::TensorCreateInterface>(
            cloned.getOperation());
      }]
    >,
  ];

  // `getKernelOperandKind()` reads the mutable `Buffer` kind off a
  // `!mogg.buffer`, so it must be `mut`. The converse does not hold: DPS
  // outputs and mutable views are mutable `!mogg.tensor`s.
  let verify = [{
    return ::M::MOGG::verifyTensorCreateInterfaceMutability($_op);
  }];
}

def MOGG_FusedEpilogueInterface : OpInterface<"FusedEpilogueInterface"> {
  let description = [{
    A fused-output tensor whose body runs as its kernel's epilogue
    (`mogg.tensor.create.epilogue`, `mogg.tile.create.epilogue`).
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<[{
      Whether the body does work beyond the kernel's own write, which is what
      `has_epilogue_fusion` asks. `MOToMAP` gives every output a store
      epilogue, so an unfused one is a pass-through: a single store of the
      untouched value at the kernel's index, or the yield of it a compute
      lane leaves. Anything else is work the kernel would otherwise skip.
    }], "bool", "recomputesValue", (ins)>,
  ];
}

def MOGG_KernelTerminator : OpInterface<"KernelTerminator"> {
  let description = [{
    A terminator for `mogg.routine` operation.

    A kernel terminator represents a call to a particular registered mojo
    function with the following calling convention:

    ```mojo
    // mogg.kernel_terminator input(%1, %2) dps_output(%3) -> mo.opaque(...)
    def kernel[...]  (
       out : OutputTensor[...] // <- %3

       in0 : InputTensor[...]  // <- %1
       in1 : Opaque[...]       // <- %2

    ) -> Opaque[...] {         // <- kernel returns.
       ...
    }
    ```
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
     InterfaceMethod<
      "Returns the device context that this op uses. Can be nullptr if not needed.",
      "mlir::Value", "getDeviceContext", (ins)
    >,
    InterfaceMethod<
      "Returns the DPS output values of the kernel",
      "mlir::SmallVector<mlir::Value>", "getKernelDPSOutputs", (ins)
    >,
    InterfaceMethod<
      "Returns the inputs values of the kernel",
      "mlir::SmallVector<mlir::Value>", "getKernelInputs", (ins)
    >,
    InterfaceMethod<
      "Returns the type of kernel returned value (either a by_val/by_ref return)",
      "mlir::Type", "getKernelReturnOutput", (ins)
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
