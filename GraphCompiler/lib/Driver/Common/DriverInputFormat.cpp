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

#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <regex>

// Only the inline hasMEFMagicNumber is used, so the MEF library isn't linked.
#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "mlir/Bytecode/BytecodeReader.h"

using namespace M;

//===----------------------------------------------------------------------===//
// sniffInputFormat
//===----------------------------------------------------------------------===//

FileFormat M::sniffInputFormat(StringRef fileExtension,
                               StringRef fileContents) {

  if (fileContents.empty())
    return FileFormat::kUnknown;

  if (MEF::hasMEFMagicNumber(fileContents))
    return FileFormat::kMEFFile;

  // Fall back to recognizing a textual file of a familiar dialect.
  if (llvm::isASCII(fileContents)) {
    if (fileContents.contains("mo.graph"))
      return FileFormat::kMODialect;
    if (size_t start = fileContents.find("mef.model");
        start != StringRef::npos) {
      // Look for `mef.model @<name> : <framework>`.
      StringRef suffix = fileContents.drop_front(start);
      std::regex regex(R"(mef\.model\s+@\w+\s*:\s*(\w+))");
      std::match_results<StringRef::iterator> m;
      std::regex_search(suffix.begin(), suffix.end(), m, regex);
      if (!m.empty() && m.size() == 2) {
        switch (CompiledFrameworkLabel::getLabelForOpName("mef.model",
                                                          StringRef(m[1]))
                    .value) {
        case CompiledFrameworkLabel::kUnknown:
          return FileFormat::kUnknown;
        case CompiledFrameworkLabel::kModularModel:
          return FileFormat::kMGPDialect;
        }
      }
    }
  } else if (mlir::isBytecode(llvm::MemoryBufferRef(fileContents, ""))) {
    // Look for the dialect id on a word boundary.
    // TODO(#26701): Replace this with a reliable / fast dialect ID encoding.
    constexpr int maxSniffedBytes = 1024 * 1024;
    std::match_results<StringRef::iterator> m;
    if (std::regex_search(fileContents.begin(),
                          std::min(fileContents.end(),
                                   fileContents.begin() + maxSniffedBytes),
                          m, std::regex(R"(\bmo\.)"))) {
      return FileFormat::kMODialect;
    }
  }
  return FileFormat::kUnknown;
}

FileFormat
DriverInputFormatOptions::getInputFormat(StringRef fileExtension,
                                         StringRef fileContents) const {
  if (inputFormat != FileFormat::kUnknown)
    return inputFormat;
  return sniffInputFormat(fileExtension, fileContents);
}
