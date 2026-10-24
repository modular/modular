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
// This defines interfaces in the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_TD

include "GraphCompiler/MODialect/MOInterfaces.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def RMO_TensorOpInterface : OpInterface<"RMOOp", [MO_TensorOpInterface, MO_ParamDeclaration]> {
  let description = [{
    Interface for pattern matching on all RMO ops.

    Each RMO op must have a parent `mo.graph_op`, which keeps shape parameters
    unique.
  }];
  let cppNamespace = "::M::RMO";
  let verify = [{ return verifyRMOOp(cast<RMOOp>($_op)); }];
}

def RMO_MOAnalogue : OpInterface<"MOAnalogue"> {
  let description = [{
    An RMO op with the same semantics, operands and results as an existing MO
    op, but a relaxed verifier. `inferMOTypes` reconciles the RMO types with
    types that pass the stricter MO verifier.
  }];
  let cppNamespace = "::M::RMO";
  let methods = [
    InterfaceMethod<"Returns operand and result types for the analogous MO "
    "op that pass its verifier. The RMO operands are rebound to these types, "
    "so they may differ from the RMO types only in shape. Shape parameters "
    "declared along the way go in `moInputParamDecls[i]` / "
    "`moOutputParamDecls[j]` for the ith operand / jth result, each declared "
    "only at its earliest use. The `mo*` outputs must always be sufficient to "
    "construct the MO op.",

      /*retTy=*/"void",
      /*methodName=*/"inferMOTypes",
      /*args=*/(ins "MO::GraphOp":$graphOp,
                    "::llvm::SmallVectorImpl<::mlir::Type>&":$moOperandTypes,
                    "::llvm::SmallVectorImpl<::mlir::Type>&":$moResultTypes,
                    "::llvm::SmallVectorImpl<SmallVector<::M::KGEN::ParamDeclAttr>>&":$moInputParamDecls,
                    "::llvm::SmallVectorImpl<SmallVector<::M::KGEN::ParamDeclAttr>>&":$moOutputParamDecls),
      /*methodBody=*/ [{
        typename ConcreteOp::Adaptor adaptor(
            $_op->getOperands(),
            $_op->getAttrDictionary(),
            $_op->getPropertiesStorage(),
            $_op->getRegions()
        );

        ConcreteOp::inferMOTypesImpl(
          $_op->getContext(),
          graphOp,
          adaptor,
          $_op->getResults(),
          moOperandTypes,
          moResultTypes,
          moInputParamDecls,
          moOutputParamDecls
        );
      }]
    >,

    // Guarantees all analogue ops carry the output param decls attribute.
    InterfaceMethod<"Get the list of output parameters declared by the RMO op.",
      /*retTy=*/"::llvm::ArrayRef<ParamDeclAttr>",
      /*methodName=*/"getMOAnalogueOutputParamDecls",
      /*args=*/(ins),
      [{ return $_op.getOutputParamDecls(); }]
    >,

    InterfaceMethod<"Get the name of the output parameters attribute.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getMOAnalogueOutputParamDeclsAttrName",
      /*args=*/(ins),
      [{ return $_op.getOutputParamDeclsAttrName(); }]
    >,
  ];
}

def RMO_TensorSameDTypeOperandsAndResults : OpInterface<"TensorSameDTypeOperandsAndResults"> {
  let description = [{
    All tensor operands and results have the same dtype, and there is at least
    one.
  }];
  let cppNamespace = "::M::RMO";
  let verify = [{ return verifySameDTypeOperandsAndResults(cast<RMOOp>($_op)); }];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_TD
