//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CacheDialect/CacheOps.h"
#include "Support/CacheDialect/CacheAttrs.h"
#include "Support/CacheDialect/CacheDialect.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/SHA256.h"

using namespace M;
using namespace Cache;

//===----------------------------------------------------------------------===//
// Caching-related functionality
//===----------------------------------------------------------------------===//

std::string RegionCacheKey::hashKey(RegionCacheKey::KeyTy key) {
  if (std::holds_alternative<StringRef>(key))
    return std::get<StringRef>(key).str();

  Region *r = std::get<Region *>(key);
  llvm::SHA256 sha256;
  sha256.init();

  auto hashTypeOrAttr = [&](auto t) {
    llvm::SmallString<64> tmp;
    llvm::raw_svector_ostream stringStream(tmp);
    stringStream << t;
    sha256.update(stringStream.str());
  };

  r->walk([&](Operation *op) {
    // Add the operation's name to the hash.
    sha256.update(op->getName().getStringRef());

    // Add operand and result types.
    for (Type t : op->getOperandTypes())
      hashTypeOrAttr(t);
    for (Type t : op->getResultTypes())
      hashTypeOrAttr(t);

    // And finally, attribute values and names.
    for (auto attr : op->getAttrs()) {
      sha256.update(attr.getName().getValue());

      // Ignore symbol refs.
      if (attr.getValue().isa<mlir::SymbolRefAttr, mlir::FlatSymbolRefAttr>())
        continue;

      if (auto subElementIface =
              dyn_cast<mlir::SubElementAttrInterface>(attr.getValue())) {
        subElementIface.walkSubAttrs([&](Attribute attr) {
          // We only hash the attribute name for symbol refs.
          if (attr.isa<mlir::SymbolRefAttr, mlir::FlatSymbolRefAttr>())
            return;

          // Otherwise hash the attr.
          hashTypeOrAttr(attr);
        });
      } else {
        hashTypeOrAttr(attr.getValue());
      }
    }
  });

  // Finalize the hash.
  std::array<uint8_t, 32> hash = sha256.final();
  return {hash.begin(), hash.end()};
}

/// Walk a region and:
///  - Collect all symbol references from all operations in that region.
///  - Unique the references and assign them indices.
///  - Replace symbol uses with `cache.symbol_ref`
///  - Cache the region.
static LogicalResult
cacheSingleRegion(Region &r, OpBuilder &builder,
                  SmallVectorImpl<RegionHashAttr> &hashes,
                  BlobCache<M::Cache::RegionCacheKey> &cache) {
  SmallVector<StringAttr> symbolReferences;
  SmallVector<SymbolIndexAttr> refs;

  // Now we walk the symbol and collect all symbol references.
  r.walk([&](Operation *op) {
    for (auto attr : op->getAttrs()) {
      if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr.getValue()))
        symbolReferences.push_back(symbolRef.getLeafReference());
      else if (auto flatSymbolRef =
                   dyn_cast<FlatSymbolRefAttr>(attr.getValue()))
        symbolReferences.push_back(flatSymbolRef.getAttr());
      else if (auto subElementIface =
                   dyn_cast<mlir::SubElementAttrInterface>(attr.getValue())) {
        // Walk any sub-attrs.
        subElementIface.walkSubAttrs([&](Attribute attr) {
          if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr))
            symbolReferences.push_back(symbolRef.getLeafReference());
          if (auto flatSymbolRef = dyn_cast<FlatSymbolRefAttr>(attr))
            symbolReferences.push_back(flatSymbolRef.getAttr());
        });
      }
    }
  });

  // Create a unique set of symbol references while maintaining the order.
  llvm::SetVector<StringAttr> uniqueRefs(symbolReferences.begin(),
                                         symbolReferences.end());

  // Now we'll take the uniqued list of symbols we have and replace attributes
  // with the appropriate SymbolIndexAttr.
  auto replaceSymbolRef = [&](StringAttr symRef) {
    auto found = llvm::find(uniqueRefs, symRef);
    assert(found != uniqueRefs.end());
    return builder.getAttr<SymbolIndexAttr>(
        std::distance(uniqueRefs.begin(), found));
  };

  r.walk([&](Operation *op) {
    for (auto attr : op->getAttrs()) {
      if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr.getValue()))
        attr.setValue(replaceSymbolRef(symbolRef.getLeafReference()));
      else if (auto flatSymbolRef =
                   dyn_cast<FlatSymbolRefAttr>(attr.getValue()))
        attr.setValue(replaceSymbolRef(flatSymbolRef.getAttr()));
      else if (auto subElementIface =
                   dyn_cast<mlir::SubElementAttrInterface>(attr.getValue())) {
        subElementIface.replaceSubElements(
            [&](SymbolRefAttr symbolRefAttr) {
              return replaceSymbolRef(symbolRefAttr.getLeafReference());
            },
            [&](FlatSymbolRefAttr flatSymbolRefAttr) {
              return replaceSymbolRef(flatSymbolRefAttr.getAttr());
            });
      }
    }
  });

  // Finally, we can store the region. Create an op to hang it off of so we
  // can cache it.
  auto container = builder.create<ContainerOp>(r.getLoc(), r);

  // Create a place to store the bytecode.
  std::string bytecode;
  llvm::raw_string_ostream bytecodeStream(bytecode);
  // Store the container in bytecode.
  mlir::writeBytecodeToFile(container, bytecodeStream);

  // Now create a memory buffer and store it in the cache.
  std::unique_ptr<llvm::MemoryBuffer> mbufOr =
      llvm::MemoryBuffer::getMemBuffer(bytecode);
  if (!mbufOr)
    return mlir::emitError(r.getLoc())
           << "could not get a memory buffer with the bytecode of a region "
              "for caching";

  // Store it.
  auto hashOr = cache.insert(&container.getBodyRegion(), *mbufOr);
  if (failed(hashOr))
    return mlir::emitError(r.getLoc()) << hashOr.getError();

  // Finally, erase the container.
  container.erase();

  // Get the range of unique symbols as a list of FlatSymbolRefAttr.
  auto range = llvm::map_range(
      uniqueRefs, [](StringAttr ref) { return FlatSymbolRefAttr::get(ref); });

  hashes.push_back(builder.getAttr<RegionHashAttr>(
      *hashOr, SmallVector<FlatSymbolRefAttr>(range.begin(), range.end())));
  return success();
}

FailureOr<SymbolOp>
M::Cache::deflateSymbol(Operation *symbol, mlir::SymbolTable &symtab,
                        BlobCache<M::Cache::RegionCacheKey> &cache) {
  OpBuilder builder(symbol);

  // Pull out the symbol name.
  auto symName =
      symbol->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

  SmallVector<RegionHashAttr> hashes;
  for (Region &r : symbol->getRegions())
    if (failed(cacheSingleRegion(r, builder, hashes, cache)))
      return failure();

  NamedAttrList origAttrs;
  StringAttr symNameAttrName =
      builder.getStringAttr(mlir::SymbolTable::getSymbolAttrName());
  for (auto attr : symbol->getAttrs()) {
    if (attr.getName() == symNameAttrName)
      continue;

    origAttrs.append(attr);
  }

  // Remove the original symbol from the symbol table.
  symtab.remove(symbol);

  auto newSymbol = builder.create<SymbolOp>(
      symbol->getLoc(), symbol->getResultTypes(), symbol->getOperands(),
      symName, /*cachedOperation=*/
      builder.getStringAttr(symbol->getName().getStringRef()),
      /*regionHashes=*/builder.getAttr<RegionHashArrayAttr>(hashes),
      /*origHashes=*/origAttrs.getDictionary(builder.getContext()));

  // Insert the new symbol into the symbol table.
  symtab.insert(newSymbol, builder.getInsertionPoint());
  // Erase the old symbol.
  symbol->erase();

  // And we're done!
  return newSymbol;
}

/// Inflate a single region from `regionHash` and have `r` take its body.
static LogicalResult inflateRegion(Region *r, RegionHashAttr regionHash,
                                   BlobCache<RegionCacheKey> &cache) {
  auto foundOr = cache.find(regionHash.getHash());
  if (foundOr.isError())
    return mlir::emitError(r->getLoc()) << foundOr.getError();

  // Create a dummy block that we can use to inflate container ops.
  auto *b = new Block;

  // Parse the bytecode for the region.
  std::unique_ptr<llvm::MemoryBuffer> bytecode = foundOr.takeValue();
  if (failed(mlir::readBytecodeFile(
          *bytecode, b, mlir::ParserConfig(r->getContext(), false))))
    return failure();

  // Get the container, take its body, and erase it.
  ContainerOp container = cast<ContainerOp>(b->front());
  r->takeBody(container.getBodyRegion());
  container.erase();

  // Delete the block.
  delete b;

  // Finish up by replacing symbols with their original names.
  auto replaceSymbolRef = [&](SymbolIndexAttr symRef) {
    return regionHash.getSymbols()[symRef.getIndex()];
  };
  r->walk([&](Operation *op) {
    for (auto attr : op->getAttrs()) {
      if (auto symbolRef = dyn_cast<SymbolIndexAttr>(attr.getValue()))
        attr.setValue(replaceSymbolRef(symbolRef));
      else if (auto subElementIface =
                   dyn_cast<mlir::SubElementAttrInterface>(attr.getValue())) {
        subElementIface.replaceSubElements([&](SymbolIndexAttr symbolRef) {
          return replaceSymbolRef(symbolRef);
        });
      }
    }
  });

  return success();
}

FailureOr<Operation *>
M::Cache::inflateSymbol(SymbolOp cached, mlir::SymbolTable &symtab,
                        BlobCache<RegionCacheKey> &cache) {
  // Create the original op.
  OperationState opState(cached.getLoc(), cached.getCachedOperationAttr());
  // Put the symbol name back in.
  opState.addAttribute(SymbolTable::getSymbolAttrName(),
                       cached.getSymNameAttr());
  // And add the rest of the original attrs.
  opState.addAttributes(cached.getOrigAttrs().getValue());
  // Create the regions on the op. We'll fill them in later.
  for ([[maybe_unused]] auto _ : cached.getRegionHashes())
    opState.addRegion();

  OpBuilder builder(cached);

  // Remove the cached symbol.
  symtab.remove(cached);
  // Create the output operation.
  Operation *out = builder.create(opState);
  for (auto [regionHash, region] :
       llvm::zip(cached.getRegionHashes(), out->getRegions())) {
    if (failed(inflateRegion(&region, regionHash, cache)))
      return failure();
  }
  // And insert the new operation.
  symtab.insert(out, builder.getInsertionPoint());

  // Erase the `cache.symbol`.
  cached.erase();

  // Done!
  return out;
}

//===----------------------------------------------------------------------===//
// CacheDialect::registerOps
//===----------------------------------------------------------------------===//

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
