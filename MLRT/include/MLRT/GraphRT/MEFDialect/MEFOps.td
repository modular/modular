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
// This defines operations in the MEF dialect.
//
//===----------------------------------------------------------------------===//

include "MLRT/GraphRT/MEFDialect/MEFDialect.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "Support/MDialect/M.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for the operation in this dialect.
class MEFOp<string mnemonic, list<Trait> traits = []> :
    Op<MEFDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CallableOp
//===----------------------------------------------------------------------===//

def CallableOp
  : MEFOp<"callable", [HasParent<"ModuleOp">, FunctionOpInterface,
                        SymbolName, SymbolVisibility, Symbol, IsolatedFromAbove,
                        SingleBlockImplicitTerminator<"OutputOp">]> {
  let summary = "Named callable in a MEF file";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getFunctionBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getFunctionBody().front();
    }

    OutputOp getOutputOp();

    /// Returns the argument types of this callable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this callable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getFunctionBody(); }
  }];

  let builders = [
    /// Create a new mef.callable with an empty body.
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$resultTypes,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>">:$attrs)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

def OpaqueOp : MEFOp<"opaque"> {
  let summary = "opaque MEF operation";
  let description = [{
    The "mef.opaque" operation represents an operation parsed from a MEF file
    without sufficient type information to form the proper MLIR operation.  This
    allows us to load and round trip arbitrary MEF files even when they don't
    have additional metadata (e.g. MLIR types, attribute names etc) required to
    properly form original operations.
  }];

  let arguments = (ins StrAttr:$opName, ArrayAttr:$attrs, UnitAttr:$nonstrict,
                       Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$body);
  let assemblyFormat = [{
    $opName (`nonstrict` $nonstrict^)? $attrs `(` $operands `)` attr-dict `:`
            functional-type($operands, $results) $body
  }];
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelOp
  : MEFOp<"model", [HasParent<"ModuleOp">, IsolatedFromAbove,
                    RegionKindInterface, HasOnlyGraphRegion, SymbolName,
                    SymbolVisibility, Symbol]> {
  let summary = "A model with 'init' and 'execute' entry points";
  let description = [{
    This op is a lowered correlary to MGP::ModelOp that is allowed to have a
    looser relationship between init and exec regions.

    Example:
    ```mlir
    mef.model @my_model
      init (%arg0: !mef.chain) -> (!mef.chain) {
        mef.output %arg0 : !mef.chain
      }
      execute (%arg0: !mef.chain, %arg1: tensor<?xf32>) -> (!mef.chain, tensor<?xf32>) {
        mef.output %arg0, %arg1 : !mef.chain, tensor<?xf32>
      }
    ```
  }];

  let arguments = (ins
                       M_DeviceRefArrayAttr:$aDevices,                 // 0
                       StrAttr:$bFrameworkName,                        // 1
                       TypeAttrOf<FunctionType>:$cInitType,            // 2
                       TypeAttrOf<FunctionType>:$dExecuteType,         // 3
                       TypeAttrOf<FunctionType>:$eModelSignature,      // 4
                       ArrayAttr:$fInputDeviceIndices,                 // 5
                       ArrayAttr:$gResultDeviceIndices,                // 6
                       ArrayAttr:$hArgumentNames,                      // 7
                       ArrayAttr:$iResultNames,                        // 8
                       SymbolNameAttr:$sym_name,                       // 9
                       OptionalAttr<StrAttr>:$sym_visibility);         // 10
  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }

    /// Get the execute block. This function is useful for callgraph utilities.
    Region &getBodyRegion() { return getExecute(); }

    StringAttr getName() { return getSymNameAttr(); }
  }];

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
      `` $sym_name `:` custom<FrameworkName>($bFrameworkName)
      $eModelSignature
      `devices` $aDevices
      `inputDeviceIndices` $fInputDeviceIndices
      `resultDeviceIndices` $gResultDeviceIndices
      `argumentNames` $hArgumentNames
      `resultNames` $iResultNames
      `init` custom<ModelRegion>($init, $cInitType)
      `execute` custom<ModelRegion>($execute, $dExecuteType)
      attr-dict
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def OutputOp : MEFOp<"output", [Terminator, ReturnLike]> {
  let summary = "MEF output operation";
  let description = [{
    The `mef.output` operation specifies the returned values for callable in
    a MEF MLIR file.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the `mef.callable`
    (or enclosing region) that contains the operation. For example:

    ```mlir
      mef.callable @foo() : i32, f8 {
        ...
        mef.output %0, %1 : i32, f8
      }
    ```
  }];

  let builders = [
    OpBuilder<(ins)>
  ];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
