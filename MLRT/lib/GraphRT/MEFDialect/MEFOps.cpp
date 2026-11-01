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
// This file implements the MEF dialect operations.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace M;
using namespace MEF;

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

static ParseResult parseModelRegion(OpAsmParser &p, Region &region,
                                    TypeAttr &regionSignature) {
  llvm::SMLoc argLoc;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> outputTypes;

  if (p.getCurrentLocation(&argLoc) ||
      p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/true))
    return failure();

  if (p.parseArrowTypeList(outputTypes))
    return failure();

  if (p.parseRegion(region, args, /*enableNameShadowing=*/true))
    return failure();

  regionSignature = TypeAttr::get(
      p.getBuilder().getFunctionType(region.getArgumentTypes(), outputTypes));

  return success();
}

static void printModelRegion(OpAsmPrinter &p, const ModelOp &op, Region &region,
                             TypeAttr signature) {
  Block &primaryBlock = region.front();
  p << "(";

  llvm::interleaveComma(primaryBlock.getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printArrowTypeList(primaryBlock.getTerminator()->getOperandTypes());
  p << " ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

/// Parse an optional symbol visibility keyword, mirroring
/// `function_interface_impl::parseFunctionOp`.
static ParseResult parseSymbolVisibility(OpAsmParser &p,
                                         StringAttr &visibility) {
  StringRef keyword;
  if (succeeded(
          p.parseOptionalKeyword(&keyword, {"public", "private", "nested"})))
    visibility = p.getBuilder().getStringAttr(keyword);
  return success();
}

/// Prints the symbol visibility keyword, if present.
static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                  StringAttr visibility) {
  if (visibility)
    p << visibility.getValue() << ' ';
}

/// Parse the frameworkName attribute as a keyword.
static ParseResult parseFrameworkName(OpAsmParser &p,
                                      StringAttr &frameworkName) {
  StringRef keyword;
  if (p.parseKeyword(&keyword))
    return failure();
  frameworkName = StringAttr::get(p.getContext(), keyword);
  return success();
}

/// Prints the frameworkName attribute as a keyword.
static void printFrameworkName(OpAsmPrinter &p, const ModelOp &op,
                               StringAttr frameworkName) {
  p << frameworkName.getValue();
}

LogicalResult ModelOp::verifyRegions() {
  FunctionType initFuncType = getCInitType();
  FunctionType executeFuncType = getDExecuteType();

  // Verify that init region output types match the init_type attribute.
  Block *initBlock = getInitBlock();
  if (auto output = dyn_cast<OutputOp>(initBlock->getTerminator())) {
    if (output.getOperandTypes() != initFuncType.getResults()) {
      return emitOpError() << "init region output types "
                           << output.getOperandTypes()
                           << " do not match init_type results "
                           << initFuncType.getResults();
    }
  }

  if (!getInit().hasOneBlock()) {
    return emitOpError() << "init region has non-one count of blocks";
  }

  if (!getExecute().hasOneBlock()) {
    return emitOpError() << "execute region has non-one count of blocks";
  }

  // Verify that execute region output types match the execute_type attribute.
  Block *executeBlock = getExecuteBlock();
  if (auto output = dyn_cast<OutputOp>(executeBlock->getTerminator())) {
    if (output.getOperandTypes() != executeFuncType.getResults()) {
      return emitOpError() << "execute region output types "
                           << output.getOperandTypes()
                           << " do not match execute_type results "
                           << executeFuncType.getResults();
    }
  }

  // Verify that init output types are a prefix of execute input types.
  ArrayRef<Type> initOutputTypes = initFuncType.getResults();
  ArrayRef<Type> executeInputTypes = executeFuncType.getInputs();
  if (initOutputTypes.size() > executeInputTypes.size()) {
    return emitOpError() << "init output types have " << initOutputTypes.size()
                         << " elements, but execute input types only have "
                         << executeInputTypes.size()
                         << " elements; init outputs must be a prefix of "
                            "execute inputs";
  }
  ArrayRef<Type> executeInputPrefix =
      executeInputTypes.take_front(initOutputTypes.size());
  if (initOutputTypes != executeInputPrefix) {
    return emitOpError() << "init output types " << initOutputTypes
                         << " are not a prefix of execute input types "
                         << executeInputTypes;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallableOp
//===----------------------------------------------------------------------===//

void CallableOp::build(OpBuilder &builder, OperationState &state,
                       StringAttr name, ArrayRef<Type> inputTypes,
                       ArrayRef<Type> resultTypes,
                       ArrayRef<NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name,
                      builder.getFunctionType(inputTypes, resultTypes), attrs,
                      inputTypes);
}

void CallableOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<NamedAttribute> attrs) {
  state.addAttributes(attrs);
  state.addRegion();
}

OutputOp CallableOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

/// Parses a MEF callable.
///
// operation ::=
/// `mef.callable` function-signature function-attributes? function-body
///
ParseResult CallableOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

// Print the CallableOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back.
void CallableOp::print(OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult CallableOp::verifyRegions() {
  return checkOperandTypes(getOutputOp(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

void OutputOp::build(OpBuilder &odsBuilder, OperationState &odsState) {
  build(odsBuilder, odsState, odsBuilder.getType<MEF::ChainType>(), {});
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "MLRT/GraphRT/MEFDialect/MEF.cpp.inc"
