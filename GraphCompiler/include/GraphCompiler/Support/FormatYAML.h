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
// This file provides YAMLIO serialization and deserialization interfaces
// for types found in GraphCompiler such as TensorSpec and QuantSpec.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_FORMATYAML_H
#define GENERICML_FORMATYAML_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/Support/Dict.h"
#include "GraphCompiler/Support/List.h"
#include "GraphCompiler/Support/ResultInfo.h"
#include "GraphCompiler/Support/Value.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LLVMYAMLForwardDecls.h"
#include "llvm/Support/YAMLTraits.h"
#include <optional>
#include <string>

namespace llvm {
class MemoryBuffer;
}

namespace M {
class ModelSpec;
class QuantTensorSpec;
class ResultInfo;
class TensorResultInfo;

ErrorOr<ModelSpec>
loadModelSpecFromYAML(std::unique_ptr<llvm::MemoryBuffer> yamlBuffer);

} // namespace M

/// Provide the ModelSpec mapping trait for YAML conversion.
namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec> {
  static void mapping(IO &io, M::ModelSpec &modelSpec);
  static std::string validate(IO &io, M::ModelSpec &modelSpec);
};
} // namespace llvm::yaml

/// Provide the QuantTensorSpec mapping trait for YAML conversion.
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::QuantTensorSpec)

LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(M::QuantTensorSpec)

LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(M::TensorResultInfo)

template <>
struct llvm::yaml::MappingTraits<M::TensorResultInfo> {
  static void mapping(IO &io, M::TensorResultInfo &info);
};

template <>
struct llvm::yaml::MappingTraits<M::VariantResultInfo> {
  static void mapping(IO &io, M::VariantResultInfo &info);
};

LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(std::shared_ptr<M::ResultInfo>)

template <>
struct llvm::yaml::MappingTraits<std::shared_ptr<M::ResultInfo>> {
  static void mapping(IO &io, std::shared_ptr<M::ResultInfo> &info);
};

namespace M {
using YamlDictKeyValuePair = std::pair<M::GML::Variant, M::GML::Variant>;

/// Shallow-copies a composed model output `out`, replacing `Tensor` leaves with
/// `TensorResultInfo`s for YAML and collecting the tensors in
/// `processedResults` for binary output.
ErrorOr<AnyAsyncValueRef>
getResultForBinaryOutput(const M::AnyAsyncValueRef &out,
                         M::ProcessedResults &processedResults);
} // namespace M

LLVM_YAML_IS_SEQUENCE_VECTOR(M::GML::Variant)
LLVM_YAML_IS_SEQUENCE_VECTOR(M::YamlDictKeyValuePair)

template <>
struct llvm::yaml::MappingTraits<M::GML::Variant> {
  class NormalizedVariant {
  public:
    NormalizedVariant(IO &);
    NormalizedVariant(IO &, M::GML::Variant &variant);
    M::GML::Variant denormalize(IO &);
    std::optional<std::string> stringValue;
    std::optional<double> doubleValue;
    std::optional<int64_t> longValue;
    std::optional<bool> boolValue;
    std::optional<M::GML::None> noneValue;
    std::optional<M::GML::List> listValue;
    std::optional<M::GML::Tuple> tupleValue;
    std::optional<M::GML::Dict> dictValue;
    std::optional<M::TensorResultInfo> tensorResultInfoValue;
  };

  static void mapping(IO &io, M::GML::Variant &variant);
};

namespace llvm::yaml {
template <>
struct MappingTraits<M::GML::None> {
  static void mapping(IO &io, M::GML::None &list);
};

template <>
struct MappingTraits<M::GML::List> {
  static void mapping(IO &io, M::GML::List &list);
};

template <>
struct MappingTraits<M::GML::Tuple> {
  static void mapping(IO &io, M::GML::Tuple &tuple);
};

template <>
struct MappingTraits<M::YamlDictKeyValuePair> {
  static void mapping(IO &io, M::YamlDictKeyValuePair &kv);
};

template <>
struct MappingTraits<M::GML::Dict> {
  class NormalizedDict {
  public:
    NormalizedDict(IO &);
    NormalizedDict(IO &, M::GML::Dict &dict);
    M::GML::Dict denormalize(IO &);
    std::vector<M::YamlDictKeyValuePair> normalizedDict;
  };

  static void mapping(IO &io, M::GML::Dict &dict);
};
} // namespace llvm::yaml

#endif // GENERICML_FORMATYAML_H
