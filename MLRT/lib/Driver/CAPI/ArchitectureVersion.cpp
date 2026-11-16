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

#include "MLRT/Driver/CAPI/ArchitectureVersion.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace M;
using namespace M::Driver;

namespace {

Error malformedArchitectureName(llvm::StringRef api,
                                llvm::StringRef architectureName) {
  return Error("cannot read an architecture version from the " + api +
               " architecture name '" + architectureName + "'");
}

/// Consumes a nonempty run of decimal digits from the front of `text`.
bool consumeDecimal(llvm::StringRef &text, uint32_t &out) {
  size_t digits = 0;
  while (digits < text.size() && llvm::isDigit(text[digits]))
    ++digits;
  if (digits == 0 || text.take_front(digits).getAsInteger(10, out))
    return false;
  text = text.drop_front(digits);
  return true;
}

/// `sm_<major><minor>[a|f]`, where the minor is the last digit.
ErrorOr<ArchitectureVersion> parseCudaArchitecture(llvm::StringRef name) {
  llvm::StringRef rest = name;
  uint32_t cc = 0;
  if (!rest.consume_front("sm_") || !consumeDecimal(rest, cc) || cc < 10)
    return malformedArchitectureName(Device::cudaAPI, name);
  if (!rest.empty() && rest != "a" && rest != "f")
    return malformedArchitectureName(Device::cudaAPI, name);
  return ArchitectureVersion{cc / 10, cc % 10, 0};
}

/// `gfx<major><minor><stepping>[:feature...]`, where the minor is one decimal
/// digit and the stepping one hexadecimal digit.
ErrorOr<ArchitectureVersion> parseHipArchitecture(llvm::StringRef name) {
  llvm::StringRef rest = name.split(':').first;
  if (!rest.consume_front("gfx") || rest.size() < 3)
    return malformedArchitectureName(Device::hipAPI, name);
  llvm::StringRef majorDigits = rest.drop_back(2);
  const char minorDigit = rest[rest.size() - 2];
  const char steppingDigit = rest[rest.size() - 1];
  uint32_t major = 0;
  if (!consumeDecimal(majorDigits, major) || !majorDigits.empty() ||
      !llvm::isDigit(minorDigit) || !llvm::isHexDigit(steppingDigit))
    return malformedArchitectureName(Device::hipAPI, name);
  return ArchitectureVersion{major, static_cast<uint32_t>(minorDigit - '0'),
                             llvm::hexDigitValue(steppingDigit)};
}

/// `<generation>[-<metal language>]`.
ErrorOr<ArchitectureVersion> parseMetalArchitecture(llvm::StringRef name) {
  llvm::StringRef rest = name;
  uint32_t generation = 0;
  if (!consumeDecimal(rest, generation) ||
      !(rest.empty() || rest.starts_with("-")))
    return malformedArchitectureName(Device::metalAPI, name);
  return ArchitectureVersion{generation, 0, 0};
}

} // namespace

ErrorOr<ArchitectureVersion>
M::Driver::parseArchitectureVersion(llvm::StringRef api,
                                    llvm::StringRef architectureName) {
  if (api == Device::cudaAPI)
    return parseCudaArchitecture(architectureName);
  if (api == Device::hipAPI)
    return parseHipArchitecture(architectureName);
  if (api == Device::metalAPI)
    return parseMetalArchitecture(architectureName);
  if (api == Device::cpuAPI)
    return Error("the CPU device has no architecture version");
  return Error("no architecture version is known for the " + api + " API");
}

ErrorOr<ArchitectureVersion>
M::Driver::getArchitectureVersion(const Device &device) {
  if (device.api() == Device::cpuAPI)
    return Error("the CPU device has no architecture version");
  ErrorOr<const char *> name = device.getArchName();
  if (name.isError())
    return name.takeError();
  return parseArchitectureVersion(device.api(), name.get());
}
