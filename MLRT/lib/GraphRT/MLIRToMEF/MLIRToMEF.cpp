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
// This file implements the entrypoint for the MLIRToMEF library.  The converter
// is implemented in three phases.  The first phase identifies all of the
// strings and attributes that need to be emitted to the string/attribute pool.
// The second phase optimizes and emits the strings and attributes to the file
// and remembers their offsets.  The third phase emits all of the regions in the
// MLIR program.
//
// MLIR ops are converted to primitive info and stored in MEF.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/MLIRToMEF.h"
#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "MLRT/GraphRT/MEFDialect/MEFNonStrictInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>

using namespace M;

/// This holds an offset into a MEF section.
struct MEFOffset {
  size_t value;
};

/// This keeps track of the signature for an invokable (a mef.invokable or
/// region in the MLIR file).
struct RegionSignature {
  size_t numArguments, numResults;
};

/// Which attributes names should never be emitted into MEF because they
/// are not needed at runtime (and may not be encodable).
/// We need to drop these here rather than in a cleanup pass because the
/// ops which use them will generally fail validation if the attributes
/// are erased.
static bool isIgnoredAttribute(StringRef name) {
  return name == "function_type" || name == "operandSegmentSizes";
}

//===----------------------------------------------------------------------===//
// MEFEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This is the emitter that builds MEF-encoded data into an std::vector.  This
/// class contains low-level routines used by the various specific emitters.  In
/// addition to collecting the bytes contained in this piece of the MEF file,
/// this tracks the alignment requirement of the contents.  If this is a
/// subsection of the file, then the enclosing container is required to provide
/// at least this alignment.
///
/// This class works by appending into two different collections, the
/// `dataResult` vector is where individually emitted bytes get added, the
/// ultimate result is in a ConcatenationTree that is maintained as well.
/// Certain built emission operations (like including a subsection) emits the
/// current `dataResult` as node in the concatenation tree before moving on.
class MEFEmitter {
public:
  MEFEmitter() = default;
  MEFEmitter(const MEFEmitter &) = delete;
  MEFEmitter &operator=(const MEFEmitter &) = delete;

  // Return the alignment required by this chunk of a MEF file.
  unsigned getRequiredAlignment() const { return requiredAlignment; }

  /// Return the current position as a MEFOffset.
  MEFOffset getOffset() const {
    return MEFOffset{treeResult.getSize() + dataResult.size()};
  }

  /// Return a raw pointer into the result buffer for the specified offset.
  uint8_t *getRawPointer(MEFOffset offset) {
    // Cannot offset into tree data.
    auto treeSize = treeResult.getSize();
    assert(offset.value >= treeSize && "cannot get offsets into tree data");
    auto dataOffset = offset.value - treeSize;
    assert(dataOffset <= dataResult.size() && "invalid offset");
    return dataResult.data() + dataOffset;
  }

  void emitByte(uint8_t byte) { dataResult.push_back(byte); }

  void emitBytes(ArrayRef<uint8_t> bytes) {
    dataResult.insert(dataResult.end(), bytes.begin(), bytes.end());
  }

  /// Emit an array of bytes, which are owned by the caller and have a lifetime
  /// guaranteed to be longer than the MEF generation process.
  void emitOwnedBytes(ArrayRef<uint8_t> bytes) {
    if (bytes.empty())
      return;
    treeResult = ConcatenationTree::concat(
        takeContents(), ConcatenationTree::getImmortalData(bytes));
  }

  void emitStringData(StringRef str) {
    emitBytes({reinterpret_cast<const uint8_t *>(str.data()), str.size()});
  }

  void emitFloat(float value) {
    emitBytes(ArrayRef<uint8_t>((uint8_t *)&value, sizeof(float)));
  }

  void emitNulTerminatedString(StringRef str) {
    emitStringData(str);
    emitByte(0);
  }

  void alignTo(unsigned alignment) {
    // Alignment of 0 and 1 is a noop.
    if (alignment < 2)
      return;

    assert(llvm::isPowerOf2_32(alignment));

    // We need attributes to have proper alignment in the file, so figure out
    // whether we need padding before this to make sure it ends up at the right
    // address.
    size_t curOffset = getOffset().value;
    size_t neededPadding = llvm::alignTo(curOffset, alignment) - curOffset;

    // Emit dummy padding bytes to get up to the right offset.
    while (neededPadding--)
      emitByte(MEF::kAlignmentByte);

    // Keep track of the maximum required alignment.
    requiredAlignment = std::max(requiredAlignment, alignment);
  }

  // Emit a guaranteed 4-byte integer aligned to 4 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void emitInt4(uint32_t value) {
    alignTo(4);
    uint8_t data[] = {uint8_t(value & 0xFF), uint8_t((value >> 8) & 0xFF),
                      uint8_t((value >> 16) & 0xFF),
                      uint8_t((value >> 24) & 0xFF)};
    emitBytes(data);
  }

  // Emit a guaranteed 8-byte integer aligned to 8 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void emitInt8(uint64_t value) {
    alignTo(8);
    uint8_t data[] = {
        uint8_t(value & 0xFF),         uint8_t((value >> 8) & 0xFF),
        uint8_t((value >> 16) & 0xFF), uint8_t((value >> 24) & 0xFF),
        uint8_t((value >> 32) & 0xFF), uint8_t((value >> 40) & 0xFF),
        uint8_t((value >> 48) & 0xFF), uint8_t((value >> 56) & 0xFF)};
    emitBytes(data);
  }

  // Emit a VBR encoded integer of arbitrary width.
  void emitVBR(size_t value) { emitVBRImpl(value, false); }

  // Emit a VBR for a signed value.  We emit the sign bit as the low bit and
  // negate negative numbers so we can emit things like "-2" into a single byte.
  void emitSignedVBR(ssize_t value) {
    if (value >= 0)
      return emitVBR(value << 1);
    return emitVBR(((-value) << 1) | 1);
  }
  void emitInvertedVBR(size_t value);

  /// Emit an offset within a known MEF section.
  void emitOffset(MEFOffset offset) { emitVBR(offset.value); }

  /// Emit an offset to a primitive, which is known to be aligned, so we can
  /// emit a smaller number.
  void emitPrimitiveOffsetNumber(MEFOffset offset) {
    assert((offset.value & (MEF::kPrimitiveEntryAlignment - 1)) == 0 &&
           "offset to primitive isn't correctly aligned");
    emitVBR(offset.value / MEF::kPrimitiveEntryAlignment);
  }

  // Many parts of the emitter logic includes forward references into stuff
  // that hasn't been emitted and has variable size.  This is handled by making
  // nested emitters.  This helper emits the subpieces once they are
  // constructed, ensuring that alignment requirements of the nested emitter are
  // maintained correctly.
  void emitEmitter(MEFEmitter &emitter) {
    alignTo(emitter.getRequiredAlignment());

    // Append the sub-emitter's contents to our contents.
    treeResult =
        ConcatenationTree::concat(takeContents(), emitter.takeContents());
  }

  /// Emit a section record with the contents of the specified other emitter as
  /// the data within it.  This takes the contents of the emitter.
  void emitSection(MEF::SectionID sectionID, MEFEmitter &emitter) {
    // Take the data from the sub-emitter.
    ConcatenationTree sectionData = emitter.takeContents();
    unsigned alignment = emitter.getRequiredAlignment();
    assert(alignment < 256 && "alignment must fit in an uint8_t");

    // Section start with an identifier.
    emitByte(static_cast<uint8_t>(sectionID));

    // Emit a length.  The low bit indicates whether alignment padding is
    // present.  We fill it in later.
    emitVBR(sectionData.getSize() << 1);
    MEFOffset needsAlignmentPaddingOffset{getOffset().value - 1};

    // If the section body requires alignment padding, then we need to emit
    // the alignment and the padding bytes.
    if (alignment > 1) {
      if (getOffset().value & (alignment - 1)) {
        emitByte(alignment);
        alignTo(alignment);
        *getRawPointer(needsAlignmentPaddingOffset) |= 1;
      } else {
        // Otherwise, if we happen to be at a compatible offset, we just
        // remember that we need this alignment.
        requiredAlignment = std::max(requiredAlignment, alignment);
      }
    }

    // Concatenate the payload data onto ourselves.
    treeResult =
        ConcatenationTree::concat(takeContents(), std::move(sectionData));
  }

  // Take any contents we built up and return it as a single ConcatenationTree.
  ConcatenationTree takeContents() {
    auto result = ConcatenationTree::concat(
        std::move(treeResult),
        ConcatenationTree::takeVector(std::move(dataResult)));
    dataResult.clear();
    return result;
  }

private:
  void emitVBRImpl(size_t value, bool isHighPart);
  // Keep track of the alignment required for the start of this object.
  unsigned requiredAlignment = 1;

  // These are the results that we are building up as we go.  At any point in
  // time, any data in 'treeResult' is assumed to precede any data in
  // 'dataResult'.
  ConcatenationTree treeResult;
  std::vector<uint8_t> dataResult;
};

} // namespace

/// Our fundamental unit is a bytestream, but we want to be able to emit large
/// values as well.  We use a VBR encoding, where the high bit set indicates
/// that this is only a portion of the value.
void MEFEmitter::emitVBRImpl(size_t value, bool isHighPart) {
  if ((value >> 7) != 0)
    emitVBRImpl(value >> 7, /*isHighPart=*/true);

  emitByte(static_cast<uint8_t>((value & 127) | (isHighPart ? 128 : 0)));
}

/// Emit the specified value as a "backwards" VBR, which allows clients to
/// decode/parse it from the end of the value back to the front.  This is used
/// for things like string/array lengths, where we want a pointer to the string
/// data to indicate both the data and the length, allowing the executor to
/// reconstruct a StringRef/ArrayRef.
void MEFEmitter::emitInvertedVBR(size_t value) {
  emitByte(uint8_t(value & 0x7F));
  value >>= 7;
  while (value != 0) {
    emitByte(uint8_t((value & 0x7F) | 0x80));
    value >>= 7;
  }
}

//===----------------------------------------------------------------------===//
// EntityTable
//===----------------------------------------------------------------------===//

namespace {

/// This table keeps track of the interesting entities (attributes, types, other
/// strings) that we care about.  This is built in the first pass.
class EntityTable {
public:
  /// Walk the IR to build the symbol tables, this returns true on failure.
  bool collect(ModuleOp module, bool enableMEFMetadata);

  unsigned getTypeIndex(Type type) const {
    auto it = typeIDs.find(type);
    assert(it != typeIDs.end() && "unregistered type");
    return it->second;
  }

  ssize_t getRegionID(const Region &region) const {
    auto it = regionIDs.find(&region);
    return it == regionIDs.end() ? -1 : it->second;
  }

  unsigned getPrimitiveID(Operation *primitive) const {
    StringAttr name;
    if (auto opaque = dyn_cast<MEF::OpaqueOp>(primitive))
      name = opaque.getOpNameAttr();
    else
      name = primitive->getName().getIdentifier();

    auto it = primitiveIDs.find(name);
    assert(it != primitiveIDs.end() && "Unknown primitive");
    return it->second;
  }

  MEFOffset getStringOffset(StringAttr str) const {
    auto it = stringOffsets.find(str);
    assert(it != stringOffsets.end() && "unknown string");
    return it->second;
  }

  MEFOffset getMetadataStringOffset(StringAttr str) const {
    auto it = metadataStringOffsets.find(str);
    assert(it != metadataStringOffsets.end() && "unknown metadata string");
    return it->second;
  }

  MEFOffset getAttributeOffset(Attribute attr) const {
    auto it = attributeOffsets.find(attr);
    assert(it != attributeOffsets.end() && "unknown attribute");
    return it->second;
  }

  void addRegionOffset(MEFOffset bodyOffset, RegionSignature signature) {
    regionIndexEntries.push_back({bodyOffset, signature});
  }

  /// Return the MEFOffset for the specified location entry.
  MEFOffset getLocationOffset(Location loc) const {
    auto locInfo = decomposeLocation(loc);
    auto it = locationOffsets.find(locInfo);
    assert(it != locationOffsets.end() && "unknown location");
    return it->second;
  }

  InFlightDiagnostic emitError(Location loc) {
    hadError = true;
    return mlir::emitError(loc);
  }

private:
  using LocationTuple = std::tuple<StringAttr, unsigned, unsigned>;
  static LocationTuple decomposeLocation(Location loc);
  void addString(StringAttr string) { strings.insert(string); }
  void addMetadataString(StringAttr string) { metadataStrings.insert(string); }
  void addType(Type type);
  LogicalResult addRegion(Region *region);
  void addPrimitiveOpcode(StringAttr opcode);
  void addLocation(Location loc);

public:
  bool hadError = false;

  /// Uniquing set of attributes we need to emit, kept in order so we always
  /// produce a deterministic output file.  This only maintains the attributes
  /// that are references by the MLIR operations, it doesn't hold the
  /// recursive elements.
  llvm::SetVector<Attribute> attributes;
  SmallVector<Location> attributeLocs;

  /// After the attributes section is emitted, this holds the offset for
  /// each attribute that is emitted.
  DenseMap<Attribute, MEFOffset> attributeOffsets;

  /// Uniquing set of the primitives that we need to emit.
  std::vector<StringAttr> primitives;
  DenseMap<StringAttr, unsigned> primitiveIDs;

  /// A list of regions we need to emit.
  std::vector<Region *> regions;
  DenseMap<Region *, unsigned> regionIDs;

  struct InvokableIndexEntry {
    MEFOffset bodyOffset;
    RegionSignature signature;
  };

  // This follows the format of the InvokableIndex section.
  std::vector<InvokableIndexEntry> regionIndexEntries;

  // All of the strings we need to emit the MEF file.
  llvm::SetVector<StringAttr> strings;

  // All of the strings we need to emit for metadata in the MEF file, e.g.
  // attribute names.
  llvm::SetVector<StringAttr> metadataStrings;

  // All of the types we need when emitting metadata in the MEF file.
  std::vector<Type> types;
  DenseMap<Type, unsigned> typeIDs;

  /// This is the offset of each known string in the STRINGS and STRINGS_MD
  /// section.
  DenseMap<StringAttr, MEFOffset> stringOffsets, metadataStringOffsets;

  // This is all of the filenames referred to by locations in the file.
  llvm::SetVector<StringAttr> locationFilenames;

  // This is a list of locations within the file, the first element of the
  // tuple is a index into locationFilenames, the second and third are line/col
  // information.
  llvm::SetVector<LocationTuple> locations;

  /// This is the offset of a given location record in the LOCATION_POSITIONS
  /// Section of the MEF file.
  DenseMap<LocationTuple, MEFOffset> locationOffsets;
};

} // namespace

void EntityTable::addType(Type type) {
  // Ignore the type if we've seen it before.
  if (!typeIDs.insert({type, types.size()}).second)
    return;
  types.push_back(type);
}

static RegionSignature getRegionSignature(Region *region) {
  // Emit information about the type of the invokable.
  auto *block = &region->getBlocks().front();
  if (block->empty())
    return RegionSignature{0, 0};

  // MEF regions produce results with the "mef.output" operation.
  auto output = dyn_cast<MEF::OutputOp>(&block->back());
  size_t numResults = output ? output->getNumOperands() : 0;
  return RegionSignature{block->getNumArguments(), numResults};
}

LogicalResult EntityTable::addRegion(Region *region) {
  // Check to see if we support this region kind.
  if (region->getBlocks().size() != 1) {
    mlir::emitError(region->getLoc())
        << "multi-block regions cannot be emitted to MEF files";
    return failure();
  }

  for (auto argTy : region->front().getArgumentTypes())
    addType(argTy);

  regionIDs[region] = regions.size();
  regions.emplace_back(region);
  return success();
}

void EntityTable::addPrimitiveOpcode(StringAttr opcode) {
  // Remember the primitive.
  if (!primitiveIDs.insert({opcode, primitives.size()}).second)
    return;
  primitives.push_back(opcode);

  // If we haven't seen it already, add it to the string table.
  addString(opcode);
}

auto EntityTable::decomposeLocation(Location loc) -> LocationTuple {
  StringAttr filename = nullptr;
  unsigned line = 0, col = 0;
  // TODO: could support more powerful location approach.
  if (auto fileLineCol = dyn_cast<FileLineColLoc>(loc)) {
    filename = fileLineCol.getFilename();
    line = fileLineCol.getLine();
    col = fileLineCol.getColumn();
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // If we are dealing with a fused location just pick one, we can't pass both
    // to the MEF and we can't disambiguate further.
    for (Location loc : fusedLoc.getLocations()) {
      if (auto fileLineCol = dyn_cast<FileLineColLoc>(loc)) {
        filename = fileLineCol.getFilename();
        line = fileLineCol.getLine();
        col = fileLineCol.getColumn();
        break;
      }
    }
  }

  if (!filename) {
    // TODO Could look through fancy locations to get a FileLineCol in most
    // cases.
    filename = StringAttr::get(loc.getContext());
  }
  return LocationTuple{filename, line, col};
}

void EntityTable::addLocation(Location loc) {
  auto locationInfo = decomposeLocation(loc);

  locationFilenames.insert(std::get<0>(locationInfo));
  locations.insert(locationInfo);
}

/// Walk the IR to build the symbol tables, this returns true on failure.
bool EntityTable::collect(ModuleOp module, bool enableMEFMetadata) {
  // Insert an empty string in the strings list
  addString(StringAttr::get(module->getContext()));

  std::vector<std::pair<SymbolRefAttr, Location>> symbolRefAttrs;
  std::vector<std::pair<DenseResourceElementsAttr, Location>> resourceAttrs;

  auto opWalker = [&](Operation *op) -> WalkResult {
    // Ignore the OutputOp: it gets special handling, instead of being treated
    // like a primitive.  Similarly, ignore ef.invokable.
    if (isa<MEF::OutputOp>(op))
      return WalkResult::advance();

    addLocation(op->getLoc());

    auto *curBlock = op->getBlock();

    // Notice the result types of the ops.  Argument types can only be arguments
    // to the invokables and results of other ops.
    for (auto result : op->getResults())
      addType(result.getType());

    for (auto operand : op->getOperands()) {
      // Verify that the operand is defined inside the current region.  We
      // don't support references to outer regions.
      if (operand.getParentBlock() != curBlock)
        return op->emitError()
               << "MEF executor only supports references to primitives within"
               << " the current region";
    }

    llvm::unique_function<void(StringAttr, Attribute, bool)> processAttr =
        [&](StringAttr name, Attribute attr, bool insideDict) {
          // Add attribute names if requested.
          if (enableMEFMetadata && !insideDict)
            addMetadataString(name);

          // Keep track of invokable references specially so we can diagnose
          // them.
          if (auto fnAttr = dyn_cast<SymbolRefAttr>(attr))
            symbolRefAttrs.emplace_back(fnAttr, op->getLoc());

          // Delay inserting resource attributes if we can (i.e. if they aren't
          // nested). We could do more complex organization, but big resources
          // generally only appear at the top level, so delay that until we
          // actually know we need it.
          if (!insideDict) {
            if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
              resourceAttrs.emplace_back(resourceAttr, op->getLoc());
              return;
            }
          }

          // Recurse into ArrayAttr/DictionaryAttr. Perform a post-order
          // traversal so that we can use the offsets from the values inside
          // the collection when emitting the collection.
          if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
            for (Attribute element : arrayAttr)
              processAttr(StringAttr::get(op->getContext()), element, true);
          }
          if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
            for (NamedAttribute namedAttr : dictAttr)
              processAttr(namedAttr.getName(), namedAttr.getValue(), true);
          }

          // If we're handling a dictionary, then insert the name into the
          // attributes list.
          if (insideDict && attributes.insert(name))
            attributeLocs.push_back(op->getLoc());

          // We just keep track of the top level attributes in the
          // "attributes" list: do not recursively process them.  This is
          // because we want to emit arrays of scalars as "flat" attributes
          // where all the values are inline.
          if (attributes.insert(attr))
            // If this attribute is new, remember what used it.
            attributeLocs.push_back(op->getLoc());
        };

    // Opaque MEF operations encode some fields in a special way.
    if (auto opaque = dyn_cast<MEF::OpaqueOp>(op)) {
      addPrimitiveOpcode(opaque.getOpNameAttr());

      // Encode the attributes in the attrs array, not all the attributes on the
      // OpaqueOp itself.
      for (auto attr : opaque.getAttrs())
        processAttr(StringAttr::get(op->getContext()), attr, false);
    } else {
      addPrimitiveOpcode(op->getName().getIdentifier());
      // If an op is a symbol, we also have to add the symbol name string.
      if (auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(op))
        if (StringAttr symbolName = symbolOp.getNameAttr())
          addString(symbolName);

      // Keep track of any attributes used by this op. If the attr is the
      // function type attr that symbols have, then don't write that.
      for (auto attr : op->getAttrs()) {
        if (isIgnoredAttribute(attr.getName().getValue()))
          continue;
        processAttr(attr.getName(), attr.getValue(), false);
      }
    }

    // Add any regions used by this op as MEF invokables.
    for (auto &region : op->getRegions()) {
      if (failed(addRegion(&region)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  // Walk the whole module finding all the things. Walk in pre-order so the
  // first primitive stored is the module itself.
  if (module.walk(opWalker).wasInterrupted())
    return true;

  // Process the delayed resource attributes.
  for (auto [resource, loc] : resourceAttrs)
    if (attributes.insert(resource))
      attributeLocs.push_back(loc);

  // If we're successful, check to make sure that all regions can be resolved.
  for (auto attrAndLoc : symbolRefAttrs) {
    Operation *symbolOp =
        module.lookupSymbol(attrAndLoc.first.getLeafReference());
    for (auto &r : symbolOp->getRegions()) {
      if (getRegionID(r) == -1) {
        mlir::emitError(attrAndLoc.second)
            << "invokable " << attrAndLoc.first << " not defined";
        return true;
      }
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Attribute Emission
//===----------------------------------------------------------------------===//

/// This is the emitter that builds the attributes section of a MEF.
///
/// This is a bit trickier than it may seem because we don't always recursively
/// process all elements - we leave flat attributes inline, and don't
/// redundantly emit the scalar values.
namespace {
class AttributesEmitter : public MEFEmitter {
public:
  explicit AttributesEmitter(EntityTable &entities) : entities(entities) {}

  MEFOffset emitAttribute(Attribute attr, Location loc);

private:
  void emitAttributeKind(MEF::AttributeKind kind) { emitByte((uint8_t)kind); }
  FailureOr<MEF::AttributeKind> deriveAttributeKind(Type eltType,
                                                    Location loc) const;
  void moveMetadataImmediatelyBeforeValue(MEFOffset metadataStartOffset,
                                          MEFOffset metadataEndOffset,
                                          MEFOffset valueStartOffset);
  MEFOffset emitAttributeInternal(Attribute attr, Location loc);
  MEFOffset emitIntegerValue(const APInt &value, Type eltType, Location loc);
  MEFOffset emitStringAttribute(StringRef value);
  MEFOffset emitSymbolAttribute(StringRef value);

  /// Emits numElements of kind using emitRangeFn. Returns the offset at which
  /// the actual data begins. If numElements is non-zero then the returned
  /// offset will be aligned to at least alignment. However so such alignment
  /// is enforced if numElements is zero.
  MEFOffset emitIntOrFloatRange(MEF::AttributeKind kind, size_t numElements,
                                size_t alignment,
                                function_ref<void()> emitRangeFn);

  MEFOffset emitArrayElementsAttr(ArrayElementsAttr value, Location loc);
  MEFOffset
  emitDenseResourceElementsAttribute(mlir::DenseResourceElementsAttr value,
                                     Location loc);
  MEFOffset emitAlignedBytesAttribute(AlignedBytesAttr attr, Location loc);
  MEFOffset emitArrayAttribute(ArrayAttr value, Location loc);
  MEFOffset emitDictionaryAttribute(DictionaryAttr value, Location loc);
  MEFOffset emitTypeAttribute(Type type, Location loc);
  MEFOffset emitMultiLineStringAttribute(MultiLineStringAttr attr,
                                         Location loc);
  MEFOffset emitDeviceSpecCollectionAttribute(DeviceSpecCollectionAttr attr,
                                              Location loc);
  MEFOffset emitDeviceRefArrayAttribute(llvm::ArrayRef<Attribute> attr,
                                        Location loc);
  MEFOffset emitDeviceRefAttribute(DeviceRefAttr attr, Location loc);
  MEFOffset emitInOutSignatureAttribute(InOutSignatureAttr attr, Location loc);
  EntityTable &entities;

  friend class TypeEmitter;
};
} // namespace

/// Given a MEF attribute category and its bitwidth, construct the MEF attribute
/// kind.
static MEF::AttributeKind getAttributeKind(MEF::AttrCategory category,
                                           size_t bitwidth) {
  return MEF::AttributeKind(uint8_t(category) << 1 | llvm::Log2_32(bitwidth / 8)
                                                         << 3);
}

/// Given an MLIR type, construct the corresponding MEF attribute type.
FailureOr<MEF::AttributeKind>
AttributesEmitter::deriveAttributeKind(Type eltType, Location loc) const {
  if (eltType.isSignlessInteger(1))
    return MEF::AttributeKind::kBool;

  MEF::AttrCategory category;
  if (eltType.isSignedInteger())
    category = MEF::AttrCategory::Signed;
  else if (eltType.isUnsignedInteger())
    category = MEF::AttrCategory::Unsigned;
  else if (eltType.isSignlessInteger())
    return entities.emitError(loc)
           << "signless integer attributes not supported";
  else if (eltType.isF32() || eltType.isF64())
    category = MEF::AttrCategory::Float;
  else
    return entities.emitError(loc) << "unknown type " << eltType;

  size_t bitwidth = eltType.getIntOrFloatBitWidth();
  if (bitwidth < 8 || bitwidth > 64 || !llvm::isPowerOf2_32(bitwidth))
    return entities.emitError(loc) << "invalid bitwidth " << bitwidth;

  return getAttributeKind(category, bitwidth);
}

/// In some cases, we need to emit some metadata immediately before a given
/// value. That metadata might be an array size or attribute kind. Often, the
/// size of that metadata is not known until after it is emitted.
///
/// We also need to ensure that the data for a given value is properly aligned.
///
/// So we first emit the metadata, then align for the value (both in the caller
/// of this function), then here we move the metadata immediately before the
/// start of the value, if necessary.
void AttributesEmitter::moveMetadataImmediatelyBeforeValue(
    MEFOffset metadataStartOffset, MEFOffset metadataEndOffset,
    MEFOffset valueStartOffset) {
  if (metadataEndOffset.value != valueStartOffset.value) {
    size_t sizeOfArraySizeInBytes =
        metadataEndOffset.value - metadataStartOffset.value;
    memmove(getRawPointer(valueStartOffset) - sizeOfArraySizeInBytes,
            getRawPointer(metadataStartOffset), sizeOfArraySizeInBytes);
    // Set the unused bytes to the alignment filler.
    memset(getRawPointer(metadataStartOffset), MEF::kAlignmentByte,
           valueStartOffset.value - metadataEndOffset.value);
  }
}

/// Emit a single integer value of supported type.
MEFOffset AttributesEmitter::emitIntegerValue(const APInt &value, Type eltType,
                                              Location loc) {
  auto kindOr = deriveAttributeKind(eltType, loc);
  if (failed(kindOr))
    return {0};

  auto kindStart = getOffset();

  // Emit the kind of the data.
  emitAttributeKind(*kindOr);

  auto kindEnd = getOffset();

  size_t bytes = llvm::divideCeil(value.getBitWidth(), 8);
  alignTo(bytes);

  auto valueOffset = getOffset();

  moveMetadataImmediatelyBeforeValue(kindStart, kindEnd, valueOffset);

  uint64_t v = value.getLimitedValue();
  emitBytes({reinterpret_cast<const uint8_t *>(&v), bytes});
  return valueOffset;
}

MEFOffset AttributesEmitter::emitStringAttribute(StringRef value) {
  // In order to support strings of data with NUL characters in them, we emit
  // strings as modified Pascal-style strings, that start with a VBR length. The
  // string passed to the function points to the start of the data, but the
  // bytes before it specify how long the string is.  To enforce this notion, we
  // don't emit a zero byte at the end of the string data.
  emitInvertedVBR(value.size());
  emitAttributeKind(MEF::AttributeKind::kString);
  auto offset = getOffset();
  emitStringData(value);
  return offset;
}

MEFOffset AttributesEmitter::emitSymbolAttribute(StringRef value) {
  // In order to support strings of data with NUL characters in them, we emit
  // strings as modified Pascal-style strings, that start with a VBR length. The
  // string passed to the function points to the start of the data, but the
  // bytes before it specify how long the string is.  To enforce this notion, we
  // don't emit a zero byte at the end of the string data.
  emitInvertedVBR(value.size());
  emitAttributeKind(MEF::AttributeKind::kSymbolRef);
  auto offset = getOffset();
  emitStringData(value);
  return offset;
}

MEFOffset
AttributesEmitter::emitIntOrFloatRange(MEF::AttributeKind kind,
                                       size_t numElements, size_t alignment,
                                       function_ref<void()> emitRangeFn) {
  MEFOffset sizeStart = getOffset();

  // Emit the number of elements first and then emit all of the attributes
  // consecutively, returning the offset of the first element in the attribute
  // array.
  emitInvertedVBR(numElements);

  // Emit the byte for the attribute kind just after the inverted VBR for the
  // elements.
  emitAttributeKind(MEF::getArrayOf(kind));

  MEFOffset sizeAndKindEnd = getOffset();

  // The attribute array is empty. We just return the offset past the size.
  if (numElements == 0)
    return sizeAndKindEnd;

  // Emit a range of integer values representing the elements of the array.
  alignTo(alignment);
  MEFOffset startOffset = getOffset();

  // If there is a gap between the size and the first attribute value, we
  // move the size to be immediately before the first attribute value to
  // remove the gap.
  moveMetadataImmediatelyBeforeValue(sizeStart, sizeAndKindEnd, startOffset);

  emitRangeFn();
  return startOffset;
}

MEFOffset AttributesEmitter::emitDenseResourceElementsAttribute(
    mlir::DenseResourceElementsAttr value, Location loc) {
  Type elementType = value.getElementType();

  // We only support integer and FP element types.
  if (!elementType.isIntOrFloat()) {
    entities.emitError(loc) << "unsupported element type";
    return {0};
  }

  // Get the attribute kind for the elements of the array.
  auto kindOr = deriveAttributeKind(elementType, loc);
  if (failed(kindOr))
    return {0};

  // Verify that the width of the array is of a form we support.
  size_t bitwidth = elementType.getIntOrFloatBitWidth();
  if (!llvm::isPowerOf2_64(bitwidth) || bitwidth > 64 || bitwidth == 2 ||
      bitwidth == 4) {
    entities.emitError(loc) << "unsupported element width " << bitwidth;
    return {0};
  }

  // Check if the data for this resource is present, we can't emit if it isn't.
  mlir::AsmResourceBlob *resourceBlob = value.getRawHandle().getBlob();
  if (!resourceBlob) {
    entities.emitError(loc) << "cannot emit resource without underlying data: "
                            << value.getRawHandle().getKey();
    return {0};
  }

  // The attribute may include an alignment in its type annotation.
  // Otherwise use the blob's alignment.
  uint64_t align = dyn_cast<HasAlignedBytesInterface>(value)
                       .getAlignedBytesType()
                       .getAlign();

  return emitIntOrFloatRange(*kindOr, value.size(), align, [&]() {
    emitOwnedBytes(resourceBlob->getDataAs<uint8_t>());
  });
}

MEFOffset AttributesEmitter::emitAlignedBytesAttribute(AlignedBytesAttr attr,
                                                       Location loc) {
  auto kindOr = deriveAttributeKind(attr.getElementType(), loc);
  if (failed(kindOr))
    return {0};

  // All encoded AlignedBytes attributes are a uint8_t array of bytes.
  return emitIntOrFloatRange(*kindOr, attr.getNumElements(),
                             static_cast<size_t>(attr.getAlign()),
                             [&]() { emitOwnedBytes(attr.getData()); });
}

MEFOffset AttributesEmitter::emitArrayElementsAttr(ArrayElementsAttr value,
                                                   Location loc) {
  auto kindOr = deriveAttributeKind(value.getElementType(), loc);
  if (failed(kindOr))
    return {0};

  // The attribute may include an alignment in its type annotation.
  // Otherwise use a default according to the element type.
  size_t align = value.getAlignedBytesType().getAlign();

  return emitIntOrFloatRange(*kindOr, value.size(), align,
                             [&]() { emitOwnedBytes(value.getRawData()); });
}

MEFOffset AttributesEmitter::emitArrayAttribute(ArrayAttr value, Location loc) {
  // Emit the size of the array and the attribute kind.
  emitInvertedVBR(value.size());
  emitAttributeKind(MEF::AttributeKind::kList);

  MEFOffset startOffset = getOffset();

  for (Attribute element : value) {
    // Emit the offsets to the name and data as an offset relative to the
    // dictionary offset. They are therefore implicitly negative!
    size_t elementOffset =
        startOffset.value - entities.getAttributeOffset(element).value;
    emitOffset(MEFOffset{elementOffset});
  }

  return startOffset;
}

MEFOffset AttributesEmitter::emitDictionaryAttribute(DictionaryAttr value,
                                                     Location loc) {
  // Emit the size of the dict and the attribute kind.
  emitInvertedVBR(value.size());
  emitAttributeKind(MEF::AttributeKind::kDictionary);

  // This is the offset into the dict we want to provide - essentially a table
  // of name:offset pairs that we can use when parsing.
  MEFOffset nameAndOffsetStart = getOffset();

  for (NamedAttribute namedAttr : value) {
    // Emit the offsets to the name and data as an offset relative to the
    // dictionary offset. They are therefore implicitly negative!
    size_t relativeNameOffset =
        nameAndOffsetStart.value -
        entities.getAttributeOffset(namedAttr.getName()).value;

    size_t relativeAttrOffset =
        nameAndOffsetStart.value -
        entities.getAttributeOffset(namedAttr.getValue()).value;

    emitOffset(MEFOffset{relativeNameOffset});
    emitOffset(MEFOffset{relativeAttrOffset});
  }

  return nameAndOffsetStart;
}

MEFOffset
AttributesEmitter::emitMultiLineStringAttribute(MultiLineStringAttr attr,
                                                Location loc) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  for (auto stringAttr : attr.getValue())
    os << stringAttr.getValue() << '\n';
  os << '\0';
  return emitStringAttribute(storage);
}

MEFOffset AttributesEmitter::emitDeviceSpecCollectionAttribute(
    DeviceSpecCollectionAttr attr, Location loc) {
  ErrorOr<DeviceSpecCollection> deviceSpecsOr = toRuntimeDeviceSpecs(attr);
  if (deviceSpecsOr) {
    entities.emitError(loc)
        << "unable to represent DeviceSpecsAttr: " << deviceSpecsOr.getError();
    return {0};
  }
  // Encode the attribute in JSON. We'll decode at runtime into a
  // DeviceSpecCollection.
  return emitStringAttribute(deviceSpecsOr->serializeToJSON());
}

MEFOffset
AttributesEmitter::emitDeviceRefArrayAttribute(llvm::ArrayRef<Attribute> attrs,
                                               Location loc) {
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::json::OStream json(os);
  json.arrayBegin();
  for (auto attr : attrs) {
    auto devAttr = dyn_cast<DeviceRefAttr>(attr);
    if (!devAttr) {
      entities.emitError(loc) << "DeviceRefArrayAttr is not all DeviceRefAttrs";
      return {0};
    }
    ErrorOr<DeviceRef> deviceOr = toRuntimeDeviceRef(devAttr);
    if (deviceOr.isError()) {
      entities.emitError(loc)
          << "unable to represent DeviceRefAttr: " << deviceOr.takeError();
      return {0};
    }
    deviceOr->serializeToJSON(json);
  }
  json.arrayEnd();
  return emitStringAttribute(s);
}

MEFOffset AttributesEmitter::emitDeviceRefAttribute(DeviceRefAttr attr,
                                                    Location loc) {
  ErrorOr<DeviceRef> deviceOr = toRuntimeDeviceRef(attr);
  if (deviceOr.isError()) {
    entities.emitError(loc)
        << "unable to represent DeviceRefAttr: " << deviceOr.takeError();
    return {0};
  }
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::json::OStream json(os);
  deviceOr->serializeToJSON(json);
  return emitStringAttribute(s);
}

MEFOffset
AttributesEmitter::emitInOutSignatureAttribute(InOutSignatureAttr attr,
                                               Location loc) {
  // Just use the underlying signature string directly.
  return emitStringAttribute(attr.getSignature());
}

namespace {
class TypeEmitter {
public:
  TypeEmitter(AttributesEmitter *emitter, Location loc)
      : emitter(emitter), loc(loc) {}
  LogicalResult emitType(Type type);

private:
  ErrorOr<MEF::TypeKind> getFloatTypeEncoding(FloatType type);
  ErrorOr<MEF::TypeKind> getIntegerTypeEncoding(IntegerType type);
  ErrorOr<MEF::TypeKind> getIntFPComplexTypeEncoding(Type type);
  LogicalResult emitIntFPComplexType(Type type);
  LogicalResult emitShapedType(ShapedType type);
  LogicalResult emitFunctionType(FunctionType type);
  LogicalResult emitUniformQuantizedType(UniformQuantizedType type);
  LogicalResult
  emitUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type);
  LogicalResult emitStorageAndExpressedTypes(QuantizedType type);

  AttributesEmitter *emitter;
  Location loc;
};
} // namespace

ErrorOr<MEF::TypeKind> TypeEmitter::getFloatTypeEncoding(FloatType type) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  if (isa<MLIR_TYPE>(type))                                                    \
    return MEF::TypeKind::SHORT_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  return Error("unknown float type");
}

/// Emit an integer type we know how to encode into a MEF file.
ErrorOr<MEF::TypeKind> TypeEmitter::getIntegerTypeEncoding(IntegerType type) {
  unsigned bitWidth = type.getWidth();

  if (bitWidth == 1 && type.isSignless())
    return MEF::TypeKind::kBool;

  if (type.isSignless())
    return Error("signless integers not supported");

  auto eltType = DType::getInt(bitWidth, type.isSigned());
  if (failed(eltType))
    return Error("integer cannot be converted to MEF type");

  return MEF::TypeKind(eltType->getValue());
}

/// Helper to get int, float or complex encoding.
ErrorOr<MEF::TypeKind> TypeEmitter::getIntFPComplexTypeEncoding(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return getFloatTypeEncoding(floatType);
  if (auto intType = dyn_cast<IntegerType>(type))
    return getIntegerTypeEncoding(intType);
  if (auto complexType = dyn_cast<mlir::ComplexType>(type)) {
    auto eltTypeOr = getIntFPComplexTypeEncoding(complexType.getElementType());
    if (eltTypeOr.isError())
      return eltTypeOr.takeError();
    return MEF::TypeKind(DType::getComplex(*eltTypeOr).getValue());
  }

  std::string str;
  llvm::raw_string_ostream os(str);
  os << "the MLIR type " << type << " is not supported by MEF";
  return Error(str);
}

/// Emit an int, float, or complex type.
LogicalResult TypeEmitter::emitIntFPComplexType(Type type) {
  // mgp.tensor_type uses NoneType to represent unknown dtype - encode it as
  // MEF::TypeKind::invalid.
  if (isa<NoneType>(type)) {
    emitter->emitByte(MEF::TypeKind::invalid);
    return success();
  }

  auto eltEncoding = getIntFPComplexTypeEncoding(type);
  if (eltEncoding.isError()) {
    emitter->entities.emitError(loc) << eltEncoding.getError() << ": " << type;
    return failure();
  }
  emitter->emitByte(eltEncoding->getValue());
  return success();
}

/// Emits a shaped type, such as `tensor<1x3xf32>`, into MEF.
/// The type is encoded in the following form:
///   TypeKind::tensor | element TypeKind | rank | VBR encoded dims
/// For the example tensor above, the output format would look like this:
///   TypeKind::tensor | TypeKind::f32 | 2 | 1, 3
/// Unranked tensors are encoded with a rank of
/// std::numeric_limits<uint8_t>::max() and no dimension entries.
LogicalResult TypeEmitter::emitShapedType(ShapedType type) {
  emitter->emitByte(MEF::TypeKind::tensor);
  Type elementType = type.getElementType();

  if (failed(emitType(elementType)))
    return failure();

  if (type.hasRank()) {
    ArrayRef<int64_t> shape = type.getShape();
    assert(shape.size() <= kMaxRank);
    emitter->emitByte(static_cast<uint8_t>(shape.size()));
    for (int64_t dimOrDynamic : shape) {
      // Note that dimOrDynamic may be ShapedType::kDynamic, which will be
      // encoded in unsigned form, then decoded back to ssize_t at runtime.
      emitter->emitVBR(static_cast<size_t>(dimOrDynamic));
    }
  } else {
    emitter->emitByte(Detail::kDynamicRank);
  }
  return success();
}

LogicalResult TypeEmitter::emitStorageAndExpressedTypes(QuantizedType type) {
  // QuantizedType stores the signedness information in a flag, but it is easier
  // to emit a signed or unsigned type.
  auto storageType = IntegerType::get(
      type.getContext(), type.getStorageTypeIntegralWidth(),
      type.isSigned() ? IntegerType::Signed : IntegerType::Unsigned);
  if (failed(emitIntFPComplexType(storageType)))
    return failure();

  // In practice these are floats, but we could emit other expressed types too.
  return emitIntFPComplexType(type.getExpressedType());
}

/// emitUniformQuantizedType emits a uniform, not per-axis quantized type such
/// as `!quant.uniform<i8:f32, 0.1:-128>` into MEF. The type is encoded in the
/// following form:
///   TypeKind::quantized | storage TypeKind | expressed TypeKind |
///   scale as float | zero point in VBR
/// For the example type above, the output format would look like this:
///   TypeKind::quantized | TypeKind::si8 | TypeKind::f32 | 0.1 | -128
LogicalResult TypeEmitter::emitUniformQuantizedType(UniformQuantizedType type) {
  emitter->emitByte(MEF::TypeKind::quantized);

  if (failed(emitStorageAndExpressedTypes(type)))
    return failure();

  // Ensure that we emit the float aligned to 4 bytes.
  emitter->alignTo(4);

  // We emit the scale as is.
  emitter->emitFloat((float)type.getScale());

  // The zero point is emitted in VBR.
  emitter->emitSignedVBR(type.getZeroPoint());

  // FIXME: need to handle non-default storageTypeMin/Max
  return success();
}

/// emitUniformQuantizedPerAxisType emits a uniform, per-axis quantized type
/// such as `!quant.uniform<i16:f32:1, {0.2:128, 0.1:-128}>` into MEF. The type
/// is encoded in the following form:
///   TypeKind::quantizedPerAxis | storage TypeKind | expressed TypeKind |
///   quantized dim | number of scales (or zero points) in VBR |
///   scales (each as float) | zero points (each in VBR)
/// The number of scales is always equal to the number of zero points.
/// For the example type above, the output format would look like this:
///   TypeKind::quantizedPerAxis | TypeKind::si16 | TypeKind::f32 |
///   1 | 2 | 0.2 | 0.1 | 128 | -128
LogicalResult
TypeEmitter::emitUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type) {
  emitter->emitByte(MEF::TypeKind::quantizedPerAxis);

  if (failed(emitStorageAndExpressedTypes(type)))
    return failure();

  // We first emit the index of the quantized dimension.
  emitter->emitByte((uint8_t)type.getQuantizedDimension());

  // Then we emit the number of scales (or zero points) in VBR.
  auto scales = type.getScales();
  emitter->emitVBR(scales.size());

  // Scales are emitted directly, in order.
  for (auto const scale : scales)
    emitter->emitFloat(float(scale));

  // Zero points are emitted in VBR, in order.
  for (auto const zeroPoint : type.getZeroPoints())
    emitter->emitSignedVBR(zeroPoint);

  return success();
}

LogicalResult TypeEmitter::emitFunctionType(FunctionType type) {
  emitter->emitByte(MEF::TypeKind::function);
  emitter->emitVBR(type.getNumInputs());
  emitter->emitVBR(type.getNumResults());
  for (auto input : type.getInputs()) {
    if (failed(emitType(input)))
      return failure();
  }

  for (auto result : type.getResults()) {
    if (failed(emitType(result)))
      return failure();
  }
  return success();
}

LogicalResult TypeEmitter::emitType(Type type) {
  if (auto tensorType = dyn_cast<ShapedType>(type))
    return emitShapedType(tensorType);

  if (auto qType = dyn_cast<UniformQuantizedType>(type))
    return emitUniformQuantizedType(qType);

  if (auto fnType = dyn_cast<FunctionType>(type))
    return emitFunctionType(fnType);

  if (auto qType = dyn_cast<UniformQuantizedPerAxisType>(type))
    return emitUniformQuantizedPerAxisType(qType);

  // Chains are super easy, just emit the byte that says "I'm a chain".
  if (isa<MEF::ChainType>(type)) {
    emitter->emitByte(MEF::TypeKind::chain);
    return success();
  }

  // Interface for foreign types.
  if (isa<MEF::MEFAnyTypeInterface>(type)) {
    emitter->emitByte(MEF::TypeKind::any);
    return success();
  }

  // FIXME: We're ignoring CalibratedQuantizedType.  Nothing in the compiler
  // uses this, why are we generating it in the first place?
  if (auto qType = dyn_cast<CalibratedQuantizedType>(type))
    return emitType(qType.getExpressedType());

  return emitIntFPComplexType(type);
}

/// emitTypeAttribute handles type emission for floats, integers, and ranked
/// tensor types and returns the start offset for the attribute in question.
MEFOffset AttributesEmitter::emitTypeAttribute(Type type, Location loc) {
  emitAttributeKind(MEF::AttributeKind::kType);
  MEFOffset startOffset = getOffset();

  TypeEmitter typeEmitter(this, loc);
  if (failed(typeEmitter.emitType(type))) {
    // The emitter will have recorded a diagnostic at the origin of failure.
    return {0};
  }
  return startOffset;
}

MEFOffset AttributesEmitter::emitAttributeInternal(Attribute attr,
                                                   Location loc) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return emitIntegerValue(intAttr.getValue(), intAttr.getType(), loc);

  if (auto fpAttr = dyn_cast<FloatAttr>(attr))
    return emitIntegerValue(fpAttr.getValue().bitcastToAPInt(),
                            fpAttr.getType(), loc);

  if (auto string = dyn_cast<StringAttr>(attr))
    return emitStringAttribute(string.getValue());

  if (auto symRef = dyn_cast<SymbolRefAttr>(attr))
    return emitSymbolAttribute(symRef.getLeafReference());

  if (auto arrayElements = dyn_cast<ArrayElementsAttr>(attr))
    return emitArrayElementsAttr(arrayElements, loc);

  if (auto denseElements = dyn_cast<mlir::DenseResourceElementsAttr>(attr))
    return emitDenseResourceElementsAttribute(denseElements, loc);

  if (auto alignedBytesAttr = dyn_cast<AlignedBytesAttr>(attr))
    return emitAlignedBytesAttribute(alignedBytesAttr, loc);

  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    if (array.size() > 0 && dyn_cast<DeviceRefAttr>(array.getValue().front()))
      return emitDeviceRefArrayAttribute(array.getValue(), loc);
    return emitArrayAttribute(array, loc);
  }

  if (auto dict = dyn_cast<DictionaryAttr>(attr))
    return emitDictionaryAttribute(dict, loc);

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    return emitTypeAttribute(typeAttr.getValue(), loc);

  if (auto multilineAttr = dyn_cast<MultiLineStringAttr>(attr))
    return emitMultiLineStringAttribute(multilineAttr, loc);

  if (auto deviceSpecCollectionAttr = dyn_cast<DeviceSpecCollectionAttr>(attr))
    return emitDeviceSpecCollectionAttribute(deviceSpecCollectionAttr, loc);

  if (auto inOutSigAttr = dyn_cast<InOutSignatureAttr>(attr))
    return emitInOutSignatureAttribute(inOutSigAttr, loc);

  if (auto dev = dyn_cast<DeviceRefAttr>(attr))
    return emitDeviceRefAttribute(dev, loc);

  entities.emitError(loc) << "unsupported attribute " << attr;
  return {0};
}

MEFOffset AttributesEmitter::emitAttribute(Attribute attr, Location loc) {
  assert(!entities.attributeOffsets.count(attr) && "Attr already emitted?");
  auto offset = emitAttributeInternal(attr, loc);
  entities.attributeOffsets[attr] = offset;
  return offset;
}

//===----------------------------------------------------------------------===//
// InvokableBodyEmitter
//===----------------------------------------------------------------------===//

namespace {
// This is the emitter that builds the invokable body of a MEF.
class RegionEmitter : public MEFEmitter {
public:
  RegionEmitter(EntityTable &entities, MEFEmitter *attributeNamesSection,
                MEFEmitter *mlirTypesSection)
      : entities(entities), attributeNamesSection(attributeNamesSection),
        mlirTypesSection(mlirTypesSection) {}

  void emitRegion(Region *region);

private:
  void emitRegisterTable(Block *block);
  unsigned getNumNonOutputUsers(Value result) const;
  void emitPrimitiveResultUsers(Value result, MEFEmitter &primitiveBody) const;
  void emitArgumentsPseudoOp(Block *block, MEFEmitter &primitiveList) const;
  void emitPrimitive(Operation *op, MEFEmitter &primitiveList) const;

  unsigned getRegisterNumber(Value reg) const {
    auto it = registerNumbers.find(reg);
    assert(it != registerNumbers.end() && "Unknown register");
    return it->second;
  }

  DenseMap<Value, unsigned> registerNumbers;
  DenseMap<Operation *, unsigned> primitiveIndices;

  EntityTable &entities;

  // If optional metadata is enabled, this section collects information about
  // the names that correspond to primitive attributes.
  MEFEmitter *attributeNamesSection;
  // If optional metadata is enabled, this section collects information about
  // the MLIR type for each argument and primitive result.
  MEFEmitter *mlirTypesSection;
};
} // namespace

void RegionEmitter::emitRegion(Region *region) {
  assert(region->getBlocks().size() == 1 && "should have a single block");
  auto &block = region->getBlocks().front();

  // Get a dense numbering of primitives.
  unsigned numPrimitives = 0;

  // If the invokable has arguments, we emit a pseudo-op that provides the
  // argument values.
  if (block.getNumArguments() != 0)
    ++numPrimitives;

  for (auto &op : block.getOperations()) {
    if (!isa<MEF::OutputOp>(op))
      primitiveIndices[&op] = numPrimitives++;
  }

  // Emit a count of primitives.
  emitVBR(numPrimitives);

  // Emit the register table.
  emitRegisterTable(&block);

  // Next is an offset for each primitive (from the start of the primitive
  // list) then each primitive is emitted in turn.
  MEFEmitter primitiveList;

  // If we have arguments, emit a pseudo op (with no opcode) that produces the
  // registers on entry to the invokable.
  if (block.getNumArguments() != 0) {
    // Offset of the primitive in the list.
    emitPrimitiveOffsetNumber(primitiveList.getOffset());
    // Pseudo has zero operands that need to be available.
    emitVBR(0);
    emitArgumentsPseudoOp(&block, primitiveList);
  }

  for (auto &op : block) {
    // Output primitive gets special processing.
    if (isa<MEF::OutputOp>(op))
      continue;

    bool isNonStrict = false;
    if (auto opaque = dyn_cast<MEF::OpaqueOp>(op))
      isNonStrict = opaque.getNonstrict();
    else
      isNonStrict = isa<MEF::NonStrictInterface>(op);

    // Offset of the primitive in the list.
    primitiveList.alignTo(MEF::kPrimitiveEntryAlignment);
    emitPrimitiveOffsetNumber(primitiveList.getOffset());
    // Number of operands that need to be available before it is ready to go.
    auto numOperandsBeforeRunning = op.getNumOperands();

    // We set the number to 1 for non-strict primitives so they get kicked off
    // as soon as any argument is available.  We use 1 instead of zero because
    // primitives with no operands ready are likely to just wait anyway.
    if (isNonStrict && numOperandsBeforeRunning)
      numOperandsBeforeRunning = 1;

    emitVBR(numOperandsBeforeRunning);
    emitPrimitive(&op, primitiveList);
  }

  // Emit the result registers list at the end, if present.
  if (!block.empty()) {
    if (auto outputOp = dyn_cast<MEF::OutputOp>(block.back())) {
      for (auto operand : outputOp.getOperands())
        emitVBR(getRegisterNumber(operand));
    }
  }

  // Once we're done, we can emit the primitive data after the primitive index
  // list. Note that primitive entries are fixed32 integers with 4-byte
  // alignment.
  alignTo(MEF::kPrimitiveEntryAlignment);
  emitEmitter(primitiveList);

  primitiveIndices.clear();
}

void RegionEmitter::emitRegisterTable(Block *block) {
  MEFEmitter regTable;
  unsigned numRegisters = 0;

  auto emitRegister = [&](Value reg) {
    // Then the use-count.
    regTable.emitVBR(std::distance(reg.use_begin(), reg.use_end()));
    registerNumbers[reg] = numRegisters++;
  };

  for (auto arg : block->getArguments())
    emitRegister(arg);

  for (auto &op : *block)
    for (auto result : op.getResults())
      emitRegister(result);

  // Emit the number of registers, then the register table.
  emitVBR(numRegisters);
  emitEmitter(regTable);
}

unsigned RegionEmitter::getNumNonOutputUsers(Value result) const {
  int numUsers = 0;
  for (auto *user : result.getUsers()) {
    // Ignore the 'mef.output' op, it gets special handling.
    if (!isa<MEF::OutputOp>(user))
      ++numUsers;
  }
  return numUsers;
}

void RegionEmitter::emitPrimitiveResultUsers(Value result,
                                             MEFEmitter &primitiveBody) const {
  for (auto *user : result.getUsers()) {
    // Ignore the 'mef.output' op, it gets special handling.
    if (isa<MEF::OutputOp>(user))
      continue;

    auto it = primitiveIndices.find(user);
    assert(it != primitiveIndices.end() && "Invalid user");
    primitiveBody.emitInt4(it->second);
  }
}

void RegionEmitter::emitArgumentsPseudoOp(Block *block,
                                          MEFEmitter &primitiveList) const {
  // This primitive starts with a dummy code and a dummy location. And this
  // primitive only has results and UsedBys in its body.
  primitiveList.emitInt4(0);                        // code
  primitiveList.emitInt4(0);                        // location
  primitiveList.emitInt4(0);                        // # operands
  primitiveList.emitInt4(0);                        // # attributes
  primitiveList.emitInt4(0);                        // # invokables
  primitiveList.emitInt4(block->getNumArguments()); // # results

  // Only emit a single register number for the decoder for consistency.  It
  // should always be zero since this is the first result of the first
  // primitive.
  assert(block->getNumArguments() >= 1);
  assert(getRegisterNumber(block->getArguments().front()) == 0);
  primitiveList.emitInt4(0);

  for (auto arg : block->getArguments())
    primitiveList.emitInt4(getNumNonOutputUsers(arg));

  for (auto arg : block->getArguments()) {
    emitPrimitiveResultUsers(arg, primitiveList);
    // If extra metadata is enabled, remember the type for each result.
    if (mlirTypesSection)
      mlirTypesSection->emitVBR(entities.getTypeIndex(arg.getType()));
  }
}

void RegionEmitter::emitPrimitive(Operation *op,
                                  MEFEmitter &primitiveList) const {
  // Collect the regions and attributes to emit.
  SmallVector<uint32_t> regionsToEmit;
  SmallVector<uint64_t> attributesToEmit;

  // Grab the module so we can use it to lookup symbols when necessary.
  ModuleOp theModule = op->getParentOfType<ModuleOp>();

  auto processAttr = [&](StringAttr name, Attribute value) {
    // Symbol ref attrs are emitted as invokables.
    if (auto fnAttr = dyn_cast<FlatSymbolRefAttr>(value)) {
      // Make sure we emit every region in the callee.
      Operation *callee = theModule.lookupSymbol(fnAttr.getAttr());
      for (auto &r : callee->getRegions())
        regionsToEmit.emplace_back(entities.getRegionID(r));
    }

    attributesToEmit.push_back(entities.getAttributeOffset(value).value);
    if (attributeNamesSection)
      attributeNamesSection->emitOffset(entities.getMetadataStringOffset(name));
  };

  // mef.opaque operands store attributes in an explicit way since we don't know
  // the attribute names.
  if (auto opaque = dyn_cast<MEF::OpaqueOp>(op)) {
    // All encoded attributes are stored in a single array.
    for (auto attr : opaque.getAttrs())
      processAttr(StringAttr::get(op->getContext()), attr);
  } else {
    for (auto attrNamePair : op->getAttrs()) {
      if (isIgnoredAttribute(attrNamePair.getName()))
        continue;
      processAttr(attrNamePair.getName(), attrNamePair.getValue());
    }
  }

  // Regions on the op are also considered to be MEF invokables.
  for (auto &region : op->getRegions())
    regionsToEmit.push_back(entities.getRegionID(region));

  // Emit the PRIMITIVE_HEADER records.
  primitiveList.emitInt4(entities.getPrimitiveID(op));
  primitiveList.emitInt4(entities.getLocationOffset(op->getLoc()).value);
  primitiveList.emitInt4(op->getNumOperands());
  primitiveList.emitInt4(attributesToEmit.size());
  primitiveList.emitInt4(regionsToEmit.size());
  primitiveList.emitInt4(op->getNumResults());

  // Emit the PRIMITIVE_BODY contents.
  for (auto attrOffset : attributesToEmit)
    primitiveList.emitInt8(attrOffset);
  for (auto operand : op->getOperands())
    primitiveList.emitInt4(getRegisterNumber(operand));
  for (auto fnOffset : regionsToEmit)
    primitiveList.emitInt4(fnOffset);

  // Emit a single result number if we have one or more results.  They are
  // numbered consecutively, so we don't need to emit them all.
  if (!op->getResults().empty())
    primitiveList.emitInt4(getRegisterNumber(op->getResults()[0]));
  // Emit the counts of each results' users...
  for (auto result : op->getResults())
    primitiveList.emitInt4(getNumNonOutputUsers(result));
  // ...followed by the primitives that use them.
  for (auto result : op->getResults()) {
    emitPrimitiveResultUsers(result, primitiveList);

    // If extra metadata is enabled, remember the type for each result.
    if (mlirTypesSection)
      mlirTypesSection->emitVBR(entities.getTypeIndex(result.getType()));
  }
}

//===----------------------------------------------------------------------===//
// MEFModuleEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This emitter generates the top level structure of the MEF file.
class MEFModuleEmitter : public MEFEmitter {
public:
  explicit MEFModuleEmitter(ModuleOp module) : module(module) {}

  /// Walk the IR to build the symbol tables, this returns true on failure.
  bool collectEntities(bool enableMEFMetadata) {
    return entities.collect(module, enableMEFMetadata);
  }
  void extractLocationInfo(MEFEmitter &locationFilenames,
                           MEFEmitter &locationPositions);
  void emitStrings();
  void extractStringsMD(MEFEmitter &stringsMDSection);
  void extractTypesMD(MEFEmitter &typeStringsMDSection);
  void emitAttributesAndExtractIndexMD(MEFEmitter &attrMDSection);
  void emitPrimitiveNames();
  void emitInvokablesAndExtractMD(MEFEmitter *attributeNamesSection,
                                  MEFEmitter *mlirTypesSection);
  void emitInvokableIndex();

  bool hadError() const { return entities.hadError; }

private:
  ModuleOp module;
  EntityTable entities;
};
} // namespace

void MEFModuleEmitter::extractLocationInfo(MEFEmitter &locationFilenames,
                                           MEFEmitter &locationPositions) {
  DenseMap<StringAttr, MEFOffset> filenameOffsets;
  for (StringAttr filename : entities.locationFilenames) {
    filenameOffsets[filename] = locationFilenames.getOffset();
    locationFilenames.emitNulTerminatedString(filename.getValue());
  }

  // Emit each of the positions and remember the offsets within the section.
  for (auto position : entities.locations) {
    entities.locationOffsets[position] = locationPositions.getOffset();
    const auto &[fname, line, col] = position;
    locationPositions.emitOffset(filenameOffsets[fname]);
    locationPositions.emitVBR(line);
    locationPositions.emitVBR(col);
  }
}

void MEFModuleEmitter::emitStrings() {
  // Emit the strings to the string section and remember their offsets.
  MEFEmitter stringSection;
  for (StringAttr string : entities.strings) {
    entities.stringOffsets[string] = stringSection.getOffset();
    stringSection.emitNulTerminatedString(string ? string.getValue() : "");
  }

  emitSection(MEF::SectionID::kStrings, stringSection);
}

void MEFModuleEmitter::extractStringsMD(MEFEmitter &stringsMDSection) {
  // Emit the strings to the metadata string section and remember their offsets.
  for (StringAttr string : entities.metadataStrings) {
    entities.metadataStringOffsets[string] = stringsMDSection.getOffset();
    stringsMDSection.emitNulTerminatedString(string);
  }
}

void MEFModuleEmitter::extractTypesMD(MEFEmitter &typeStringsMDSection) {
  // Emit the type strings to the type string section.
  SmallVector<char, 64> result;
  for (Type type : entities.types) {
    result.clear();
    llvm::raw_svector_ostream os(result);
    type.print(os);
    typeStringsMDSection.emitNulTerminatedString(os.str());
  }
}

/// Emit the ATTRIBUTES and the ATTRIBUTE_INDEX_MD sections.
void MEFModuleEmitter::emitAttributesAndExtractIndexMD(
    MEFEmitter &attrMDSection) {
  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  AttributesEmitter attributesSection(entities);
  for (const auto &[attr, attrLoc] :
       llvm::zip(entities.attributes, entities.attributeLocs)) {
    auto offset = attributesSection.emitAttribute(attr, attrLoc);
    // Emit metadata for each attribute we put out so MEF2MLIR can interpret the
    // data correctly.
    attrMDSection.emitOffset(offset);
  }

  emitSection(MEF::SectionID::kAttributes, attributesSection);
}

void MEFModuleEmitter::emitPrimitiveNames() {
  // The primitives are already in a stable order, so just emit them in the
  // order they were found.
  MEFEmitter primitivesNamesSection;
  primitivesNamesSection.emitVBR(entities.primitives.size());
  for (auto op : entities.primitives)
    primitivesNamesSection.emitOffset(entities.getStringOffset(op));

  emitSection(MEF::SectionID::kPrimitiveNames, primitivesNamesSection);
}

/// Emit the INVOKABLES_BODIES section.  If optional metadata is enabled,
/// the attributeNamesSection/mlirTypesSection will be non-null and should be
/// filled in.
void MEFModuleEmitter::emitInvokablesAndExtractMD(
    MEFEmitter *attributeNamesSection, MEFEmitter *mlirTypesSection) {
  RegionEmitter invokablesSection(entities, attributeNamesSection,
                                  mlirTypesSection);

  MLIRContext *ctx = module->getContext();
  for (auto region : entities.regions) {
    StringAttr opcode = StringAttr::get(ctx);
    StringAttr symbol = StringAttr::get(ctx);

    // If the region belongs to a symbol, then fill in the primitive opcode and
    // symbol names.
    Operation *parent = region->getParentOp();
    if (auto parentSym = dyn_cast<mlir::SymbolOpInterface>(parent)) {
      if (StringAttr sym = parentSym.getNameAttr()) {
        opcode = parent->getName().getIdentifier();
        symbol = sym;
      }
    }

    // Remember that we emitted this region to this offset.
    entities.addRegionOffset(invokablesSection.getOffset(),
                             getRegionSignature(region));
    invokablesSection.emitRegion(region);
  }

  emitSection(MEF::SectionID::kTopLevelOpBodies, invokablesSection);
}

void MEFModuleEmitter::emitInvokableIndex() {
  MEFEmitter invokableIndexSection;
  invokableIndexSection.emitVBR(entities.regionIndexEntries.size());
  for (const auto &entry : entities.regionIndexEntries) {
    invokableIndexSection.emitOffset(entry.bodyOffset);
    invokableIndexSection.emitVBR(entry.signature.numArguments);
    invokableIndexSection.emitVBR(entry.signature.numResults);
  }

  emitSection(MEF::SectionID::kTopLevelOpIndex, invokableIndexSection);
}

//===----------------------------------------------------------------------===//
// Top level emission logic
//===----------------------------------------------------------------------===//

/// This function converts the specified MLIR module containing a executor
/// compatible program to the BinaryExecutableFormat (MEF) format, which is
/// the binary format that the executor takes.  When
/// `disableOptionalSections` is true, this omits metadata sections required
/// to reverse a MEF file back into a .mlir file.
///
/// On error, this emits the error message through the MLIR error handler,
/// and returns an empty vector.
ConcatenationTree M::translateMLIRToMEF(ModuleOp module,
                                        bool enableMEFMetadata) {
  MEFModuleEmitter emitter(module);

  // Build the entities table.
  if (emitter.collectEntities(enableMEFMetadata))
    return {};

  // Sections describing location information.  We collect this early, but emit
  // it near the end of the file since it isn't typically needed by the
  // MEFExecutor.
  MEFEmitter locationFilenames, locationPositions;

  // If optional section emission is enabled, these collect the data to emit
  // into the respective sections.
  MEFEmitter attrIndexMDSection;
  MEFEmitter stringsMDSection, typeStringsMDSection, attrNamesSection,
      mlirTypesSection;

  // Emit magic numbers and format version.
  emitter.emitBytes({MEF::kMagic1, MEF::kMagic2 + MEF::kVersion0});
  emitter.extractLocationInfo(locationFilenames, locationPositions);
  emitter.emitStrings();
  if (enableMEFMetadata) {
    emitter.extractStringsMD(stringsMDSection);
    emitter.extractTypesMD(typeStringsMDSection);
  }

  // Emitting the attributes can fail if there's an unsupported attribute or
  // attribute type. The diagnostic will be recorded and reported by MLIR.
  emitter.emitAttributesAndExtractIndexMD(attrIndexMDSection);

  emitter.emitPrimitiveNames();
  emitter.emitInvokablesAndExtractMD(
      enableMEFMetadata ? &attrNamesSection : nullptr,
      enableMEFMetadata ? &mlirTypesSection : nullptr);
  emitter.emitInvokableIndex();

  // Now that we've emitted all of the critical sections used in the common case
  // of the MEF emitter, emit metadata sections.
  emitter.emitSection(MEF::SectionID::kLocationFilenames, locationFilenames);
  emitter.emitSection(MEF::SectionID::kLocationPositions, locationPositions);

  // Unconditionally emit metadata that describes the attribute section, this
  // is the minimal metadata needed by MEF2MLIR.
  emitter.emitSection(MEF::SectionID::kAttributeIndexMD, attrIndexMDSection);

  // Emit all the optional sections at the end if they're requested.
  if (enableMEFMetadata) {
    emitter.emitSection(MEF::SectionID::kMLIRTypeMD, mlirTypesSection);
    emitter.emitSection(MEF::SectionID::kStringsMD, stringsMDSection);
    emitter.emitSection(MEF::SectionID::kAttributeNamesMD, attrNamesSection);
    emitter.emitSection(MEF::SectionID::kMLIRTypeStringsMD,
                        typeStringsMDSection);
  }

  // If emission diagnosed an error along the way and emitted a diagnostic,
  // then we should propagate the failure up.
  if (emitter.hadError())
    return {};

  return emitter.takeContents();
}
