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
// This file implements the main entrypoint for the MEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"

#include "AsyncRT/Runtime/CPUDevice.h"
#include "MEFFileImpl.h"
#include "MLRT/GraphRT/MEF/MEFReader.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/ML/SizeUtils.h"
#include "Support/RCRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <utility>

#define DEBUG_TYPE "mef"

using namespace M;

//===----------------------------------------------------------------------===//
// MEFFileReader
//===----------------------------------------------------------------------===//

namespace {
/// This class decodes the key sections of the specified MEF file, capturing
/// some of their ranges into MEFFileImpl and completely decoding others.
///
/// The convention here is that these methods return false on success and
/// returns true and fill in `errorMessage` on failure.
class MEFFileReader : public MEFReader {
public:
  MEFFileReader(ArrayRef<uint8_t> file, MEFFileImpl &mefFile)
      : MEFReader(file,
                  [this](Error message) { this->error = std::move(message); }),
        mefFile(mefFile) {}

  std::optional<Error> &getError() { return error; }

  /// Decode the key sections of the specified MEF file, capturing some of their
  /// ranges into MEFFileImpl and completely decoding others.
  bool readFileContents();

private:
  /// Decode the next section and remember it if we care about it.
  bool readNextSection();

  /// Decode the invokable index section.
  bool readInvokableIndexSection(MEFReader &stringReader);

  /// This is filled in if an error occurs.
  std::optional<Error> error;

  ArrayRef<uint8_t> invokableIndex;

  // This is the file structure we're reading.
  MEFFileImpl &mefFile;
};
} // namespace

/// Decode the key sections of the specified MEF file, capturing some of their
/// ranges into MEFFileImpl and completely decoding others.  Return true and
/// fill in errorMessage on failure.
bool MEFFileReader::readFileContents() {
  // Start by reading all the sections, capturing the ones we care about in
  // MEFFileInfo.
  while (!empty()) {
    if (readNextSection())
      return true;
  }

  MEFReader stringReader(mefFile.strings, [this](Error message) {
    this->error = std::move(message);
  });

  // Now that we've figured out the contents of the sections, resolve the
  // things we parse ahead of time.
  return readInvokableIndexSection(stringReader);
}

/// Decode the invokable index section.
bool MEFFileReader::readInvokableIndexSection(MEFReader &stringReader) {
  setContents(invokableIndex);

  size_t numInvokables;
  if (parseVBR(numInvokables))
    return true;

  if (numInvokables < 1)
    return emitError("MEF file must have at least one invokable");

  auto *fnMemory =
      (MEFInvokableImpl *)malloc(numInvokables * sizeof(MEFInvokableImpl));

  // Read until we run out of invokable descriptors.
  size_t regionNumber = 0;
  auto reportIncorrectInvokableCount = [&]() -> bool {
    // Destroy the partially constructed invokables array.
    while (regionNumber)
      fnMemory[--regionNumber].~MEFInvokableImpl();
    free(fnMemory);
    return emitError("invokable count is incorrect");
  };

  while (!empty()) {
    if (regionNumber > numInvokables)
      return reportIncorrectInvokableCount();

    size_t bodyOffset, numArgs, numResults;
    if (parseVBR(bodyOffset) || parseVBR(numArgs) || parseVBR(numResults))
      return true;

    /// Create the actual MEFInvokable.
    (void)new (&fnMemory[regionNumber++])
        MEFInvokableImpl(numArgs, numResults, bodyOffset, &mefFile);
  }

  // Check that the correct number of invokables was actually present.
  if (regionNumber != numInvokables)
    return reportIncorrectInvokableCount();
  // Otherwise, we succeeded!
  mefFile.invokables = fnMemory;
  mefFile.numInvokables = numInvokables;
  return false;
}

/// Decode the next section and remember it if we care about it.
bool MEFFileReader::readNextSection() {
  // Decode the section ID and find the section contents.
  uint8_t sectionID;
  ArrayRef<uint8_t> sectionContents;
  if (readSection(sectionID, sectionContents))
    return true;

  // Remember the sections we care about either in MEFFileInfo (if they get
  // decoded lazily by the invokable executor) or in MEFFileReader (if they're
  // decoded eagerly at file load time).  Ignore sections we don't care about.
  switch (static_cast<MEF::SectionID>(sectionID)) {
  default: // Ignore unknown section.
    return false;

  case MEF::SectionID::kLocationFilenames:
    mefFile.locationFilenames = sectionContents;
    return false;

  // We just remember the locations of the kLocationPositions section - it is
  // directly indexed with a byte offset.
  case MEF::SectionID::kLocationPositions:
    mefFile.locationPositions = sectionContents;
    return false;

  // We just remember the locations of the kStrings section - it is directly
  // indexed with a byte offset.
  case MEF::SectionID::kStrings:
    mefFile.strings = sectionContents;
    return false;

  // We just remember the locations of the kAttributes section - it is directly
  // indexed with a byte offset.
  case MEF::SectionID::kAttributes:
    mefFile.attributes = sectionContents;
    return false;

  case MEF::SectionID::kPrimitiveNames:
    mefFile.primitiveNames = sectionContents;
    return false;

  // We just remember the locations of the kInvokablesBodies section - it is
  // directly indexed with a byte offset from the kInvokablesIndex section.
  case MEF::SectionID::kTopLevelOpBodies:
    mefFile.invokableBodies = sectionContents;
    return false;

  case MEF::SectionID::kTopLevelOpIndex:
    invokableIndex = sectionContents;
    return false;
  }
}

//===----------------------------------------------------------------------===//
// MEFFileImpl
//===----------------------------------------------------------------------===//

MEFFileImpl::~MEFFileImpl() {
  if constexpr (PrimitiveProfilerEntry::isEnabled()) {
    // Profilers only exist in their owning `Runtime`.
    if (auto cpuDevice = AsyncRT::CPUDevice::getCurrentCPUDeviceOrNull()) {
      // Make sure any borrowed strings accumulated during profiling are
      // converted to owned strings before the MEFFile they are borrowed from is
      // destroyed.
      if (std::optional<TimeTraceProfiler> &profiler = cpuDevice->getProfiler())
        profiler->intern();
    }
  }

  free(static_cast<void *>(invokables));
}

/// This method enumerates the primitive name table, calling the specified
/// callable on each entry.  The callable function should return a bool, and
/// a `true` result indicates that iteration should stop or `false` to
/// continue.
ErrorOrSuccess MEFFileImpl::enumeratePrimitiveNames(
    const std::function<ErrorOrSuccess(StringRef)> &callbackFn) {
  ErrorOrSuccess errorResult;
  MEFReader primNameReader(primitiveNames,
                           [&](Error err) { errorResult = std::move(err); });
  MEFReader stringReader(strings,
                         [&](Error err) { errorResult = std::move(err); });

  size_t numPrimitives;
  if (primNameReader.parseVBR(numPrimitives))
    return errorResult;
  primitives.reserve(numPrimitives);
  decodedPrimitiveNames.reserve(numPrimitives);

  // Read until we run out of names.
  while (!primNameReader.empty()) {
    // Each primitive name is encoded as an offset into the string table.
    size_t stringOffset;
    StringRef name;
    if (primNameReader.parseVBR(stringOffset) ||
        stringReader.parseNULTerminatedStringAt(stringOffset, name))
      return errorResult;

    // Invoke the callback, propagating an error if needed.
    if (auto err = callbackFn(name))
      return err;
  }
  return success();
}

/// Return the invokable with the specified Invokable ID.
MEFInvokable *MEFFileImpl::getInvokableByID(size_t number) const {
  assert(number < numInvokables && "invalid invokable");
  // Invokables are stored consecutively in memory.
  return invokables + number;
}

//===----------------------------------------------------------------------===//
// MEFFile implementation
//===----------------------------------------------------------------------===//

namespace {
/// A fake 'model' used by MEFInvokable::walkPrimitives to determine the value
/// of the mgp.model's frameworkName attribute.
class MGPModel final {
public:
  explicit MGPModel(std::string frameworkName)
      : frameworkName(std::move(frameworkName)) {}

  static StringLiteral getEntityName() { return "mgp.model"; }

  static ErrorOr<std::unique_ptr<MGPModel>>
  getFromDescriptor(const PrimitiveDescriptor &descriptor) {
    StringRef frameworkName = *descriptor.getAttr<StringRef>(1);
    return std::make_unique<MGPModel>(frameworkName.str());
  }

  std::string frameworkName;
};

} // namespace

/// Return true if the specified file contents look like it is a MEF file, by
/// checking the magic number.
bool M::isMEFFile(llvm::StringRef contents) {
  return MEF::hasMEFMagicNumber(contents);
}

/// Home the vtable in a .cpp file.
MEFFile::~MEFFile() = default;

/// This method inspects the contents of the MEF file to make an educated
/// guess of what it contains, allowing for better QoI and usability of tools.
CompiledFrameworkLabel MEFFile::sniffFileContents() const {
  bool isMopModel = false;
  CompiledFrameworkLabel result{CompiledFrameworkLabel::kUnknown};

  // Method 1: Look for a distinguished model constructor op name.
  (void)((MEFFileImpl *)const_cast<MEFFile *>(this))
      ->enumeratePrimitiveNames([&](StringRef primitiveName) -> ErrorOrSuccess {
        if (primitiveName == "mgp.model") {
          // We'll need to use method 2 below.
          isMopModel = true;
        } else if (CompiledFrameworkLabel::isValidOpName(primitiveName)) {
          result = CompiledFrameworkLabel::getLabelForOpName(primitiveName);
        } else {
          // Keep looking.
          return success();
        }
        // Stop iteration early (we'll ignore the outer error).
        return Error("done");
      });

  if (isMopModel) {
    // Method 2: Look at the name of the mgp.model's frameworkName attribute.
    SmallVector<std::unique_ptr<MGPModel>> models;
    ErrorOrSuccess err = getTopLevelInvokable()->walkPrimitives<MGPModel>(
        [&models](std::unique_ptr<MGPModel> &&model) {
          models.emplace_back(std::move(model));
          return success();
        });
    if (!err.isError() && !models.empty())
      result = CompiledFrameworkLabel::getLabelForOpName(
          "mgp.model", models.front()->frameworkName);
  }

  return result;
}

/// Open and read a MEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On failure, an error
/// message is emitted to the errorHandler and a null RCRef is returned.
///
ErrorOr<RCRef<MEFFile>> MEFFile::open(StringRef file) {
  if (auto maxAttrAlign = maxAttributeAlignment();
      !isAddrAligned(maxAttrAlign, file.data()))
    return Error(Twine("MEF file (0x") +
                 Twine::utohexstr(reinterpret_cast<uintptr_t>(file.data())) +
                 Twine(") must be aligned to: 0x") +
                 Twine::utohexstr(maxAttrAlign.value()));

  if (!MEF::hasMEFMagicNumber(file))
    return Error("not a MEF file");
  file = file.drop_front(2); // Drop the magic number.

  auto mefFilePtr = RCRef<MEFFileImpl>::create();

  ArrayRef<uint8_t> fileData((const uint8_t *)file.data(), file.size());
  MEFFileReader reader(fileData, *mefFilePtr);
  if (reader.readFileContents()) {
    assert(reader.getError() && "Failed but didn't produce an error!");
    return std::move(*reader.getError());
  }

  return mefFilePtr;
}

/// Check that the primitives name table is valid and can be resolved to
/// entries in the specified registry.
ErrorOrSuccess MEFFile::installPrimitives(const PrimitiveRegistry &registry) {
  auto *mefFile = static_cast<MEFFileImpl *>(this);
  return mefFile->enumeratePrimitiveNames(
      [&](StringRef primitiveName) -> ErrorOrSuccess {
        auto *primitiveImpl = registry.get(primitiveName);
        if (!primitiveImpl)
          return Error("unknown primitive '" + primitiveName + "'");

        // Otherwise remember it.
        LLVM_DEBUG(llvm::dbgs()
                   << "registering primitive '" << primitiveName
                   << "' with opcode " << mefFile->primitives.size() << "\n");
        mefFile->primitives.push_back(primitiveImpl);
        mefFile->decodedPrimitiveNames.push_back(primitiveName);
        return success();
      });
}

/// Open and read a MEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On an initial parse
/// failure, an error message is returned in the `ErrorOr`.  MEF file format
/// errors discovered lazily when running invokables, so they are returned
/// through the invokable's result AsyncValue's.
///
/// Note that this doesn't set up the primitive table, it requires an explicit
/// call to `installPrimitives` to do so.
///
/// TODO: This should (optionally) manage ownership of the underlying data
/// passed in.  For example, we could take a closure to run when the lifetime
/// of the MEFFile is done, or have a reference counted abstraction for
/// holding memory buffers.
ErrorOr<RCRef<MEFFile>> MEFFile::open(StringRef file,
                                      const PrimitiveRegistry &registry) {
  auto resultOr = open(file);
  if (resultOr.isError())
    return resultOr.takeError();
  auto result = std::move(*resultOr);
  if (auto err = result->installPrimitives(registry))
    return err.takeError();
  return result;
}

const MEFInvokable *MEFFile::getTopLevelInvokable() const {
  auto mefFile = static_cast<const MEFFileImpl *>(this);
  assert(mefFile->numInvokables > 0 && "Expected at least 1 invokable");
  // The last invokable is the module. We are guaranteed to have at least one
  // invokable, because parsing the file will fail if we don't.
  return &mefFile->invokables[mefFile->numInvokables - 1];
}
