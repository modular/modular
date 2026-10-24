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

#ifndef GENERICML_COMMONDRIVER_DRIVERINPUTFORMAT_H
#define GENERICML_COMMONDRIVER_DRIVERINPUTFORMAT_H

#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <filesystem>

namespace llvm {
class MemoryBuffer;
}

namespace M {

enum class FileFormat : uint8_t {
  kUnknown = 0,

  /// This looks like a MEF file of unknown contents.
  kMEFFile,

  /// This is a MLIR file with the mo dialect in it.
  kMODialect,

  /// This is a MLIR file with the mgp dialect and no fallbacks in it.
  kMGPDialect,
};

inline FileFormat frameworkToFileFormat() { return FileFormat::kMODialect; }

/// Guess the format of the input file contents.
FileFormat sniffInputFormat(StringRef fileExtension, StringRef fileContents);

struct DriverInputFormatOptions {
public:
  /// User override of the input format; autosensed by default.
  FileFormat inputFormat{FileFormat::kUnknown};

  /// Returns the user-specified input format, or an autosensed one.
  FileFormat getInputFormat(StringRef fileExtension,
                            StringRef fileContents) const;
};

class DriverInputFormatCLOptions {

public:
  DriverInputFormatOptions &options;
  DriverInputFormatCLOptions(DriverInputFormatOptions &opts) : options(opts) {}

private:
  llvm::cl::OptionCategory DriverInputFormatCategory{
      "Driver input format options"};
  /// User override of the input format; autosensed by default.
  M::cl::MOpt<FileFormat, true> inputFormat{
      "input-format", cl::desc("input file format override"),
      cl::values(clEnumValN(FileFormat::kMODialect, "mo", "Input is a MO file"),
                 clEnumValN(FileFormat::kMEFFile, "mef",
                            "Input is an executable MEF file"),
                 clEnumValN(FileFormat::kMGPDialect, "mgp",
                            "Input is an mgp dialect file")),
      llvm::cl::location(options.inputFormat),
      llvm::cl::cat(DriverInputFormatCategory)};
};
} // namespace M

#endif // GENERICML_COMMONDRIVER_DRIVERINPUTFORMAT_H
