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

#ifndef GENERICML_API_SOURCEMODEL_H
#define GENERICML_API_SOURCEMODEL_H

#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include <filesystem>
#include <type_traits>

namespace M {

/// Opaque representation for compiling a model.
struct ModelSource {
  void *source;
  M::FileFormat fileFormat;
};
static_assert(std::is_standard_layout_v<ModelSource> &&
                  std::is_trivial_v<ModelSource>,
              "ModelSource must be C ABI compatible");

/// Represents a model that has not yet been compiled, such as a TensorFlow
/// saved model.
struct SourceModel {
  SourceModel(ModelSource source)
      : modelPath(std::nullopt), modelSource{source} {}

  SourceModel(std::filesystem::path path, FileFormat format)
      : modelPath(std::move(path)), modelSource(ModelSource{nullptr, format}) {}

  FileFormat format() const { return modelSource.fileFormat; }

  std::optional<std::filesystem::path> modelPath;
  // Always stores the FileFormat.
  // Compilation prefers to compile from `ModelSource.source` when non-null.
  ModelSource modelSource;
};

} // namespace M

#endif // GENERICML_API_SOURCEMODEL_H
