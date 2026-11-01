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

#ifndef GENERICML_FRAMEWORKFRONTEND_H
#define GENERICML_FRAMEWORKFRONTEND_H

#include "Support/Buffer.h"
#include "Support/Context.h"
#include "Support/SymbolExport.h"

#include <memory>

namespace M {

class DriverConfig;
class DriverOptions;
class ExecutorDriverOptions;
enum class FileFormat : uint8_t;
namespace MO {
struct GCPipelineOptions;
}

// TODO: Avoid exporting these; mt currently needs the symbols.
#define EXPORT_FOR_MT DRIVER_VISIBILITY_EXPORT

class MOFContext final {
public:
  explicit MOFContext(ContextRef);
  EXPORT_FOR_MT MOFContext(ContextRef, int &argc, char **&argv);
  EXPORT_FOR_MT ~MOFContext();

  // TODO: setOptions is used only by mt
  EXPORT_FOR_MT ErrorOrSuccess setOptions(ArrayRef<std::string> opts);

  // TODO: getExecutorDriverOptions is only used by mt
  EXPORT_FOR_MT const ExecutorDriverOptions *getExecutorDriverOptions() const;

  // TODO: setDriverOptions is only used by mt
  EXPORT_FOR_MT void setDriverOptions(const DriverOptions &options);

  // TODO: overrideLoweringOptions is only used by mt
  EXPORT_FOR_MT void overrideLoweringOptions(MO::GCPipelineOptions &);

  ErrorOrSuccess setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs);

  /// Compile an input buffer of a known format to an MEF model binary.
  EXPORT_FOR_MT ErrorOr<M::BufferRef> compileToBinary(M::BufferRef file,
                                                      FileFormat format);

  /// Compile an opaque input of a known format, passed through to the
  /// framework driver, to an MEF model binary.
  ErrorOr<M::BufferRef> compileToBinary(void *source, FileFormat format);

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};

#undef EXPORT_FOR_MT

} // namespace M

#endif // GENERICML_FRAMEWORKFRONTEND_H
