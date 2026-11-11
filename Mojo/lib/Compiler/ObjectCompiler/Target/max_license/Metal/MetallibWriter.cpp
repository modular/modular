//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Container layout, little-endian, as emitted by `metallib` 32023.921:
//
//   0x00 "MTLB" u16 platform, u16 version major, u16 version minor,
//        u8 file type, u8 OS target, u16 OS major, u16 OS minor, u64 file size
//   0x18 (u64 offset, u64 size) for: function list, public metadata,
//        private metadata, bitcode. The function list size excludes its count.
//   0x58 u32 function count, then per function: u32 entry size (including
//        itself) and tags ending in ENDT. A tag is a 4-byte name, a u16
//        payload size, and the payload.
//        A header-extension tag list follows the function list; the loader
//        rejects files without it.
//
// `metallib` additionally writes a library name (HDYN) and per-function
// reflection (RLST/RFLT). Neither is needed to load a library or to build a
// pipeline with argument reflection, so they are omitted.
//
//===----------------------------------------------------------------------===//

#include "Target/max_license/Metal/MetallibWriter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <array>
#include <cstdint>

namespace M::KGEN {
namespace {

constexpr uint16_t kPlatformMacOS = 0x8001;
constexpr uint8_t kOSTargetMacOS = 0x81;
// The container format version, independent of the AIR version in `VERS`.
constexpr uint16_t kFileVersionMajor = 2;
constexpr uint16_t kFileVersionMinor = 9;
constexpr uint8_t kFileTypeExecutable = 0;
constexpr uint8_t kFunctionTypeKernel = 2;
constexpr size_t kHeaderSize = 0x58;

class TagWriter {
public:
  explicit TagWriter(std::string &out)
      : os(out), le(os, llvm::endianness::little) {}

  void tag(llvm::StringRef name, llvm::ArrayRef<uint8_t> payload) {
    os << name;
    le.write<uint16_t>(payload.size());
    os.write(reinterpret_cast<const char *>(payload.data()), payload.size());
  }
  void tag(llvm::StringRef name, llvm::StringRef payload) {
    tag(name,
        llvm::ArrayRef<uint8_t>(
            reinterpret_cast<const uint8_t *>(payload.data()), payload.size()));
  }
  template <typename... Ts>
  void tagInts(llvm::StringRef name, Ts... values) {
    std::string payload;
    llvm::raw_string_ostream pos(payload);
    llvm::support::endian::Writer pw(pos, llvm::endianness::little);
    (pw.write(values), ...);
    tag(name, payload);
  }
  void end() { os << "ENDT"; }

  llvm::raw_string_ostream os;
  llvm::support::endian::Writer le;
};

uint32_t mdInt(const llvm::MDNode *node, unsigned idx) {
  if (!node || idx >= node->getNumOperands())
    return 0;
  if (auto *c = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(
          node->getOperand(idx)))
    return c->getZExtValue();
  return 0;
}

const llvm::MDNode *firstNamedOperand(const llvm::Module &module,
                                      llvm::StringRef name) {
  const llvm::NamedMDNode *md = module.getNamedMetadata(name);
  return md && md->getNumOperands() ? md->getOperand(0) : nullptr;
}

} // namespace

llvm::Expected<std::string> writeMetallib(const llvm::Module &module,
                                          llvm::StringRef airBitcode) {
  llvm::SmallVector<llvm::StringRef> kernels;
  if (const llvm::NamedMDNode *md = module.getNamedMetadata("air.kernel")) {
    for (const llvm::MDNode *kernel : md->operands()) {
      auto *fn = kernel->getNumOperands()
                     ? llvm::mdconst::dyn_extract_or_null<llvm::Function>(
                           kernel->getOperand(0))
                     : nullptr;
      if (!fn)
        return llvm::createStringError("malformed !air.kernel entry");
      kernels.push_back(fn->getName());
    }
  }
  if (kernels.empty())
    return llvm::createStringError("module has no !air.kernel functions");

  // !air.version = {major, minor, patch}
  // !air.language_version = {"Metal", major, minor, patch}
  const llvm::MDNode *airVersion = firstNamedOperand(module, "air.version");
  const llvm::MDNode *langVersion =
      firstNamedOperand(module, "air.language_version");
  if (!airVersion || !langVersion)
    return llvm::createStringError(
        "module lacks !air.version or !air.language_version");

  llvm::VersionTuple osVersion =
      llvm::Triple(module.getTargetTriple()).getOSVersion();

  std::array<uint8_t, 32> hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(airBitcode.data()), airBitcode.size()));

  std::string entries;
  for (llvm::StringRef name : kernels) {
    std::string tags;
    TagWriter tw(tags);
    tw.tag("NAME", (name + llvm::StringRef("\0", 1)).str());
    tw.tagInts("TYPE", kFunctionTypeKernel);
    tw.tag("HASH", hash);
    // Public metadata, private metadata, and bitcode offsets within their
    // sections. All kernels share the one module.
    tw.tagInts("OFFT", uint64_t{0}, uint64_t{0}, uint64_t{0});
    tw.tagInts("VERS", static_cast<uint16_t>(mdInt(airVersion, 0)),
               static_cast<uint16_t>(mdInt(airVersion, 1)),
               static_cast<uint16_t>(mdInt(langVersion, 1)),
               static_cast<uint16_t>(mdInt(langVersion, 2)));
    tw.tagInts("MDSZ", static_cast<uint64_t>(airBitcode.size()));
    tw.end();
    tw.os.flush();

    llvm::raw_string_ostream eos(entries);
    llvm::support::endian::write<uint32_t>(eos, tags.size() + sizeof(uint32_t),
                                           llvm::endianness::little);
    eos << tags;
  }

  // A deterministic UUID keeps the build hermetic.
  std::string extension;
  {
    TagWriter tw(extension);
    tw.tag("UUID", llvm::ArrayRef<uint8_t>(hash).take_front(16));
    tw.end();
  }

  std::string emptyMetadata;
  {
    llvm::raw_string_ostream mos(emptyMetadata);
    llvm::support::endian::write<uint32_t>(mos, 8, llvm::endianness::little);
    mos << "ENDT";
  }

  const uint64_t functionListOffset = kHeaderSize;
  const uint64_t publicMetadataOffset =
      functionListOffset + sizeof(uint32_t) + entries.size() + extension.size();
  const uint64_t privateMetadataOffset =
      publicMetadataOffset + emptyMetadata.size();
  const uint64_t bitcodeOffset = privateMetadataOffset + emptyMetadata.size();
  const uint64_t fileSize = bitcodeOffset + airBitcode.size();

  std::string out;
  out.reserve(fileSize);
  llvm::raw_string_ostream os(out);
  llvm::support::endian::Writer le(os, llvm::endianness::little);
  os << "MTLB";
  le.write<uint16_t>(kPlatformMacOS);
  le.write<uint16_t>(kFileVersionMajor);
  le.write<uint16_t>(kFileVersionMinor);
  le.write<uint8_t>(kFileTypeExecutable);
  le.write<uint8_t>(kOSTargetMacOS);
  le.write<uint16_t>(osVersion.getMajor());
  le.write<uint16_t>(osVersion.getMinor().value_or(0));
  le.write<uint64_t>(fileSize);
  for (uint64_t v : {functionListOffset, uint64_t(entries.size()),
                     publicMetadataOffset, uint64_t(emptyMetadata.size()),
                     privateMetadataOffset, uint64_t(emptyMetadata.size()),
                     bitcodeOffset, uint64_t(airBitcode.size())})
    le.write<uint64_t>(v);
  le.write<uint32_t>(kernels.size());
  os << entries << extension << emptyMetadata << emptyMetadata << airBitcode;
  os.flush();
  if (out.size() != fileSize)
    return llvm::createStringError("metallib size mismatch");
  return out;
}

} // namespace M::KGEN
