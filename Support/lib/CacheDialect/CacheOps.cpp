//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CacheDialect/CacheOps.h"
#include "Support/CacheDialect/CacheAttrs.h"
#include "Support/CacheDialect/CacheDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace Cache;

void CacheDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "Support/CacheDialect/Cache.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SymbolOp
//===----------------------------------------------------------------------===//

ParseResult SymbolOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symbolName;
  std::string cachedOperation;
  if (parser.parseSymbolName(symbolName) ||
      parser.parseString(&cachedOperation) || parser.parseKeyword("regions") ||
      parser.parseEqual())
    return failure();

  Builder &builder = parser.getBuilder();

  // Add the symbol name, stored operation name, and function type to the attr
  // dict.
  result.addAttribute(SymbolOp::getSymNameAttrName(result.name), symbolName);
  result.addAttribute(SymbolOp::getCachedOperationAttrName(result.name),
                      builder.getStringAttr(cachedOperation));

  // Parse the region hash attrs.
  SmallVector<RegionHashAttr> regionHashes;
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        regionHashes.push_back(nullptr);
        return parser.parseAttribute(regionHashes.back());
      }))
    return failure();

  // Add the region hashes.
  result.addAttribute(SymbolOp::getRegionHashesAttrName(result.name),
                      builder.getAttr<RegionHashArrayAttr>(regionHashes));

  // Now parse out the op's original attrs.
  DictionaryAttr originalAttrs;
  if (parser.parseKeyword("original_attrs") || parser.parseEqual() ||
      parser.parseAttribute(originalAttrs))
    return failure();

  // Add the original attrs.
  result.addAttribute(SymbolOp::getOrigAttrsAttrName(result.name),
                      originalAttrs);

  return success();
}

void SymbolOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " \"" << getCachedOperation() << "\" regions=[";
  llvm::interleaveComma(getRegionHashes(), printer);
  printer << "] original_attrs=" << getOrigAttrs();
}

LogicalResult SymbolOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto parent = (*this)->getParentOfType<ModuleOp>();
  for (RegionHashAttr region : getRegionHashesAttr()) {
    for (FlatSymbolRefAttr sym : region.getSymbols()) {
      auto callee =
          symbolTable.lookupSymbolIn<mlir::SymbolOpInterface>(parent, sym);
      if (!callee)
        return emitError("undefined callee: '") << sym << "'";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ContainerOp
//===----------------------------------------------------------------------===//

void ContainerOp::build(OpBuilder &builder, OperationState &state,
                        Region &body) {
  Region *region = state.addRegion();
  region->takeBody(body);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/CacheDialect/Cache.cpp.inc"
