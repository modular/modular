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
// Device-graph ops of the MGP dialect, declared in MGPDeviceGraphOps.td.
// This file holds their assembly format.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

using namespace M;
using namespace MGP;

ParseResult DeviceGraphCollectiveOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  // The shared builder and stable-address block arguments are threaded into
  // every per-device region.
  if (parser.parseKeyword("builder"))
    return failure();

  if (parser.parseLParen())
    return failure();

  OpAsmParser::Argument builderArg;
  if (parser.parseArgument(builderArg, /*allowType=*/false))
    return failure();

  // The builder type is optional and defaults to `!mgp.device_graph.builder`.
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(builderArg.type))
      return failure();
  } else {
    builderArg.type = DeviceGraphBuilderType::get(parser.getContext());
  }

  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("stable"))
    return failure();

  SmallVector<StableAddressType> stableTypes;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            return parser.parseType(stableTypes.emplace_back());
          }))
    return failure();

  // Every region's entry block takes just the builder.
  SmallVector<OpAsmParser::Argument> regionArgs = {builderArg};

  // Each region is a `DeviceRefAttr` label then its body; label order defines
  // `$devices`.
  SmallVector<DeviceRefAttr> deviceRefs;

  auto parseDeviceRegionPair = [&]() -> ParseResult {
    DeviceRefAttr deviceRef;
    if (parser.parseCustomAttributeWithFallback(deviceRef))
      return failure();
    deviceRefs.push_back(deviceRef);

    Region *region = result.addRegion();
    if (parser.parseRegion(*region, regionArgs,
                           /*enableNameShadowing=*/false))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                     parseDeviceRegionPair))
    return failure();

  // `devices` comes from the region labels, so drop it from the attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.attributes.erase(getDevicesAttrName(result.name));
  result.attributes.erase(getStableTypesAttrName(result.name));

  auto devices = DeviceRefsAttr::get(parser.getContext(), deviceRefs);
  result.addAttribute(getDevicesAttrName(result.name), devices);
  result.addAttribute(
      getStableTypesAttrName(result.name),
      StableAddressArrayAttr::get(parser.getContext(), stableTypes));

  SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  return success();
}

void DeviceGraphCollectiveOp::print(OpAsmPrinter &p) {
  // Print the shared entry-block arguments once, from the first region; the
  // region bodies are then printed without them.
  p << " builder(";
  if (!getGraphs().empty()) {
    p.printRegionArgument(getRegionBlock(0).getArgument(0),
                          /*argAttrs=*/{}, /*omitType=*/true);
  }
  p << ") stable(";
  llvm::interleaveComma(getStableTypes(), p,
                        [&](StableAddressType type) { p.printType(type); });
  p << ')';

  if (!getDevices().empty())
    p << " ";

  llvm::interleaveComma(llvm::enumerate(getDevices()), p, [&](auto idxAndAttr) {
    auto [idx, attr] = idxAndAttr;
    p.printStrippedAttrOrType(attr);
    p << " ";
    p.printRegion(getGraphs()[idx], /*printEntryBlockArgs=*/false);
  });

  // `devices` and `stableTypes` are represented by the per-region labels and
  // the `stable(...)` clause above.
  SmallVector<StringRef> elide = {getDevicesAttrName(),
                                  getStableTypesAttrName()};
  if (getConstantResources().empty())
    elide.push_back(getConstantResourcesAttrName());
  p.printOptionalAttrDict(getOperation()->getAttrs(), elide);

  if (!getOutputs().empty()) {
    p << " : ";
    llvm::interleaveComma(getOutputs(), p,
                          [&](Value v) { p.printType(v.getType()); });
  }
}

void DeviceGraphCollectiveOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // The only entry block argument is the builder, `b`.
  for (BlockArgument arg : region.getArguments())
    setNameFn(arg, "b");
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveCreateOp
//===----------------------------------------------------------------------===//

ParseResult DeviceGraphCollectiveCreateOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  Type contextType = MGP::ContextType::get(ctx);

  // The shared `!mgp.context` is also each region's second block argument.
  OpAsmParser::UnresolvedOperand theContext;
  if (parser.parseOperand(theContext))
    return failure();

  // cache(%cache : !mgp.device_graph.cache)?
  bool hasCache = false;
  OpAsmParser::UnresolvedOperand cacheVal;
  Type cacheType;
  if (succeeded(parser.parseOptionalKeyword("cache"))) {
    hasCache = true;
    if (parser.parseLParen() || parser.parseOperand(cacheVal) ||
        parser.parseColon() || parser.parseType(cacheType) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> deviceContextVals;
  SmallVector<Type> deviceContextTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> inputVals;
  SmallVector<Type> inputTypes;

  // The body omits its entry-block arguments; the prefix clauses declare them.
  auto parseGraph = [&]() -> ParseResult {
    // graph(<deviceContext> : <type>)
    if (parser.parseKeyword("graph") || parser.parseLParen())
      return failure();

    OpAsmParser::UnresolvedOperand deviceContext;
    Type deviceContextType;
    if (parser.parseOperand(deviceContext) || parser.parseColon() ||
        parser.parseType(deviceContextType) || parser.parseRParen())
      return failure();
    deviceContextVals.push_back(deviceContext);
    deviceContextTypes.push_back(deviceContextType);

    // builder(<name>)
    if (parser.parseKeyword("builder") || parser.parseLParen())
      return failure();
    OpAsmParser::Argument builderArg;
    if (parser.parseArgument(builderArg, /*allowType=*/false))
      return failure();
    builderArg.type = DeviceGraphBuilderType::get(ctx);
    if (parser.parseRParen())
      return failure();

    // inputs(<name> : <type>, ...)
    if (parser.parseKeyword("inputs") || parser.parseLParen())
      return failure();
    SmallVector<OpAsmParser::Argument> inputArgs;
    if (succeeded(parser.parseOptionalRParen())) {
      // empty input list
    } else {
      do {
        OpAsmParser::UnresolvedOperand input;
        Type inputType;
        if (parser.parseOperand(input) || parser.parseColon() ||
            parser.parseType(inputType))
          return failure();
        inputVals.push_back(input);
        inputTypes.push_back(inputType);

        OpAsmParser::Argument inputArg;
        inputArg.ssaName = input;
        inputArg.type = inputType;
        inputArgs.push_back(inputArg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }

    // Reuse the op-level context operand's name for the block argument.
    OpAsmParser::Argument contextArg;
    contextArg.ssaName = theContext;
    contextArg.type = contextType;

    SmallVector<OpAsmParser::Argument> regionArgs;
    regionArgs.push_back(builderArg);
    regionArgs.push_back(contextArg);
    llvm::append_range(regionArgs, inputArgs);

    Region *region = result.addRegion();
    if (parser.parseRegion(*region, regionArgs,
                           /*enableNameShadowing=*/true))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(parseGraph))
    return failure();

  if (parser.resolveOperand(theContext, contextType, result.operands) ||
      parser.resolveOperands(deviceContextVals, deviceContextTypes, {},
                             result.operands) ||
      (hasCache &&
       parser.resolveOperand(cacheVal, cacheType, result.operands)) ||
      parser.resolveOperands(inputVals, inputTypes, {}, result.operands))
    return failure();

  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(deviceContextVals.size()), hasCache ? 1 : 0,
           static_cast<int32_t>(inputVals.size())}));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  return success();
}

void DeviceGraphCollectiveCreateOp::print(OpAsmPrinter &p) {
  p << ' ' << getTheContext();

  if (Value cache = getCache()) {
    p << " cache(" << cache << " : " << cache.getType() << ")";
  }

  size_t inputOffset = 0;
  for (auto [idx, region] : llvm::enumerate(getGraphs())) {
    if (idx != 0)
      p << ",";

    Block &body = region.front();
    Value deviceContext = getDeviceContexts()[idx];
    BlockArgument builderArg = body.getArgument(0);
    size_t numRegionInputs = body.getNumArguments() - 2;
    auto regionInputs =
        getInputs().drop_front(inputOffset).take_front(numRegionInputs);

    p << " graph(" << deviceContext << " : " << deviceContext.getType()
      << ") builder(";
    p.printRegionArgument(builderArg, /*argAttrs=*/{}, /*omitType=*/true);

    p << ") inputs(";
    llvm::interleaveComma(regionInputs, p, [&](Value input) {
      p << input << " : " << input.getType();
    });
    p << ") ";

    SmallVector<Value> shadowNames;
    shadowNames.reserve(1 + 1 + numRegionInputs);
    shadowNames.push_back(builderArg);
    shadowNames.push_back(getTheContext());
    llvm::append_range(shadowNames, regionInputs);
    p.shadowRegionArgs(region, shadowNames);

    p.printRegion(region, /*printEntryBlockArgs=*/false);
    inputOffset += numRegionInputs;
  }

  SmallVector<StringRef> elide = {getOperandSegmentSizesAttrName()};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elide);

  if (!getOutGraphs().empty()) {
    p << " : ";
    llvm::interleaveComma(getOutGraphs(), p,
                          [&](Value v) { p.printType(v.getType()); });
  }
}

void DeviceGraphCollectiveCreateOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  unsigned idx = region.getRegionNumber();
  setNameFn(region.front().getArgument(0), "b" + Twine(idx).str());
}
