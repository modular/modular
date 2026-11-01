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
// This file implements an in-memory representation for a model specification.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_MODELSPEC_H
#define GENERICML_SUPPORT_MODELSPEC_H

#include "GraphCompiler/Support/QuantSpec.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace M {

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

/// Framework-independent description of a model: name, I/O tensor specs, and
/// I/O layer names. Keep in sync with Models/docs/README.md.
class ModelSpec {
public:
  /// Per-tensor specification
  struct ModelIOSpec {
    std::string name;
    M::QuantTensorSpec spec;

    /// How to initialize this tensor: a keyword (zeros|ones|negones|random) or
    /// a filename (absolute or relative to the manifest).
    std::optional<std::string> data;

    std::optional<std::string> size;
    std::optional<std::string> sha1sum;

    /// TODO: Temporary, for the mo shape inference bring-up.
    std::optional<bool> legacyShapeInference;

    ModelIOSpec() = default;
    ModelIOSpec(std::string name, M::QuantTensorSpec spec,
                std::optional<std::string> data = {},
                std::optional<std::string> size = {},
                std::optional<std::string> sha1sum = {})
        : name(std::move(name)), spec(std::move(spec)), data(std::move(data)),
          size(std::move(size)), sha1sum(std::move(sha1sum)) {}
  };

  /// Maps model-spec ('preferred') I/O names to the model format's ('native')
  /// names, since equivalent models differ trivially across formats (e.g.
  /// tensorflow vs ONNX runtime). Unmapped names match. Only significant for
  /// formats with named inputs or outputs.
  struct NameSubstitutions {
    llvm::StringMap<std::string> preferredToNativeMap;

    /// Returns the 'native' name for `preferredName`, which must come from the
    /// model spec; the result then lives as long as the spec and is
    /// null-terminated.
    StringRef getPreferredName(StringRef preferredName) const;
  };

  /// Describes how to fetch the model in its native binary format.
  struct ModelBinary {
    std::string format;
    std::string url;
    std::string size;
    std::string sha1sum;
    std::string metadata;
    SmallVector<std::string, 1> dependencies;
    NameSubstitutions inputSubstitutions;
    NameSubstitutions outputSubstitutions;
    /// TODO: Temporary, for the mo shape inference bring-up.
    std::optional<bool> legacyShapeInference;
  };

  // Metric used to verify the accuracy of the model.
  struct Metric {
    std::string name;
    std::optional<double> expected;
    std::optional<double> atol;
    std::optional<double> rtol;
  };

  ModelSpec() = default;
  ModelSpec(const ModelSpec &) = default;
  ModelSpec(ModelSpec &&) = default;
  ModelSpec &operator=(ModelSpec &&) = default;

  ~ModelSpec();

  /// Returns the model binary corresponding to the given format.
  ErrorOr<const ModelBinary *> getModelBinary(StringRef format) const;

  std::string name;
  std::string task;
  std::string description;
  std::string data_name;
  std::string data_description;
  SmallVector<ModelBinary, 3> model;
  SmallVector<ModelIOSpec, 5> inputs, outputs;
  SmallVector<ModelIOSpec, 5> compile_specs;
  SmallVector<Metric, 5> metrics;
};
} // namespace M

#endif // GENERICML_SUPPORT_MODELSPEC_H
