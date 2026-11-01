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
// This is a file serializes and deserializes the types found in GraphCompiler
// such as TensorSpec, QuantSpec, and ModelSpec.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Support/FormatYAML.h"
#include "GraphCompiler/Support/ModelSpec.h"
#include "GraphCompiler/Support/QuantSpec.h"
#include "GraphCompiler/Support/ResultInfo.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <utility>

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

//===----------------------------------------------------------------------===//
// QuantTensorSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
/// Provide a forwarding from ssize_t to int64_t for YAML ScalarTraits.
template <typename T>
struct ScalarTraits<
    T, std::enable_if_t<!llvm::is_detected<ScalarTraits, T>::value &&
                        std::is_same_v<T, ssize_t>>> {
  static void output(const ssize_t &val, void *p, llvm::raw_ostream &out) {
    ScalarTraits<int64_t>::output((int64_t)val, p, out);
  }
  static StringRef input(StringRef scalar, void *p, ssize_t &value) {
    int64_t v;
    auto out = ScalarTraits<int64_t>::input(scalar, p, v);
    value = v;
    return out;
  }
  static QuotingType mustQuote(StringRef s) {
    return ScalarTraits<int64_t>::mustQuote(s);
  }
};

/// Provide the QuantSpec::ScaleZeroPointPair mapping trait for YAML conversion.
template <>
struct MappingTraits<M::QuantSpec::ScaleZeroPointPair> {
  static void mapping(IO &io,
                      M::QuantSpec::ScaleZeroPointPair &scaleZeroPointPair) {
    io.mapRequired("scale", scaleZeroPointPair.scale);
    io.mapRequired("zero_point", scaleZeroPointPair.zeroPoint);
  }
};
} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(M::QuantSpec::ScaleZeroPointPair)

static void fromYAML(llvm::yaml::IO &io, QuantTensorSpec &quantTensorSpec) {
  std::string shapeStr;
  io.mapOptional("shape", shapeStr);
  if (shapeStr.empty()) {
    quantTensorSpec = QuantTensorSpec();
    return;
  }
  auto errorOrTensorSpec = TensorSpec::parseFromString(shapeStr);
  if (failed(errorOrTensorSpec)) {
    io.setError(errorOrTensorSpec.getError());
    return;
  }
  TensorSpec tensorSpec = *errorOrTensorSpec;

  bool isQuantized = false;
  io.mapOptional("quantized", isQuantized, false);

  if (!isQuantized) {
    quantTensorSpec = QuantTensorSpec(std::move(tensorSpec));
    return;
  }

  StringRef kindStr("");
  io.mapRequired("kind", kindStr);
  auto kind = QuantSpec::kindFromString(kindStr);

  // Only present for M::QuantSpec::Kind::uniformPerAxis.
  uint8_t quantDim = 0;
  io.mapOptional("quant_dim", quantDim, 0);

  SmallVector<M::QuantSpec::ScaleZeroPointPair> scaleZeroPointPairs;
  io.mapOptional("params", scaleZeroPointPairs);

  quantTensorSpec = QuantTensorSpec(
      std::move(tensorSpec), QuantSpec(kind, quantDim, scaleZeroPointPairs));
}

static void formatYAML(llvm::yaml::IO &io, QuantTensorSpec &quantTensorSpec) {
  if (!io.outputting())
    return fromYAML(io, quantTensorSpec);

  bool isQuantized = quantTensorSpec.isQuantized();
  // Serialize as a plain TensorSpec so the shape string (dim0x...xdtype)
  // excludes quantization info.
  std::string shapeStr =
      static_cast<M::TensorSpec>(quantTensorSpec).getAsString();
  io.mapRequired("shape", shapeStr);
  io.mapOptional("quantized", isQuantized, false);
  if (quantTensorSpec.isQuantized()) {
    auto &quantSpec = quantTensorSpec.getQuantSpec();
    auto params = quantSpec.getParams();
    auto kind = M::QuantSpec::kindAsString(quantSpec.getKind());
    io.mapRequired("kind", kind);
    if (quantSpec.getKind() == M::QuantSpec::Kind::uniformPerAxis) {
      auto quantDim = quantSpec.getQuantDim();
      io.mapRequired("quant_dim", quantDim);
    }
    if (!params.empty()) {
      // yamlize doesn't accept an ArrayRef.
      SmallVector<M::QuantSpec::ScaleZeroPointPair> paramsVec(params.begin(),
                                                              params.end());
      io.mapRequired("params", paramsVec);
    }
  }
}

void llvm::yaml::MappingTraits<M::QuantTensorSpec>::mapping(
    IO &io, M::QuantTensorSpec &quantTensorSpec) {
  return formatYAML(io, quantTensorSpec);
}

//===----------------------------------------------------------------------===//
// ModelIOSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::ModelIOSpec> {
  static void mapping(IO &io, M::ModelSpec::ModelIOSpec &inputOutput) {
    io.mapOptional("name", inputOutput.name);
    io.mapOptional("data", inputOutput.data);
    io.mapOptional("size", inputOutput.size);
    io.mapOptional("sha1sum", inputOutput.sha1sum);

    // Avoids another nesting level in the YAML dictionary.
    formatYAML(io, inputOutput.spec);
  }
};
} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::ModelIOSpec)

//===----------------------------------------------------------------------===//
// M::ModelSpec::MetricSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::Metric> {
  static void mapping(IO &io, M::ModelSpec::Metric &metric) {
    io.mapRequired("name", metric.name);
    io.mapOptional("expected", metric.expected);
    io.mapOptional("atol", metric.atol);
    io.mapOptional("rtol", metric.rtol);
  }
};
} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::Metric)

//===----------------------------------------------------------------------===//
// M::ModelSpec::NameSubstitutions
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct CustomMappingTraits<M::ModelSpec::NameSubstitutions> {
  static void inputOne(IO &io, StringRef key,
                       M::ModelSpec::NameSubstitutions &nameSubstitutions) {
    // The YAML machinery guarantees unique keys.
    assert(nameSubstitutions.preferredToNativeMap.count(key) == 0 &&
           "model name already in substitution");
    io.mapRequired(key.str().c_str(),
                   nameSubstitutions.preferredToNativeMap[key]);
  }

  static void output(IO &io,
                     M::ModelSpec::NameSubstitutions &nameSubstitutions) {
    for (auto &kv : nameSubstitutions.preferredToNativeMap)
      io.mapRequired(kv.getKey().str().c_str(), kv.getValue());
  }
};
} // namespace llvm::yaml

//===----------------------------------------------------------------------===//
// M::ModelSpec::ModelBinary
//===----------------------------------------------------------------------===//

static bool isValidModelFormat(StringRef format) {
  return llvm::is_contained({"mef", "torchscript"}, format);
}

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::ModelBinary> {
  static void mapping(IO &io, M::ModelSpec::ModelBinary &binary) {
    io.mapRequired("format", binary.format);
    io.mapRequired("url", binary.url);

    if (!io.outputting() || !binary.size.empty())
      io.mapOptional("size", binary.size);

    if (!io.outputting() || !binary.sha1sum.empty())
      io.mapOptional("sha1sum", binary.sha1sum);

    if (!io.outputting() || !binary.metadata.empty())
      io.mapOptional("metadata", binary.metadata);

    if (!io.outputting() || !binary.dependencies.empty())
      io.mapOptional("dependencies", binary.dependencies);

    io.mapOptional("input_substitutions", binary.inputSubstitutions);
    io.mapOptional("output_substitutions", binary.outputSubstitutions);
  }
  static std::string validate(IO &io, M::ModelSpec::ModelBinary &binary) {
    // TODO: Make the format an enum.
    if (!isValidModelFormat(binary.format))
      return std::string("the ") + binary.format +
             " format specified is not valid";

    llvm::Regex re(R"(^(https?|file|s3)://)");
    if (!re.match(binary.url))
      return std::string("the ") + binary.url + " URL specified is not valid";

    return std::string{};
  }
};
} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::ModelBinary)
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::NameSubstitutions)

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

void llvm::yaml::MappingTraits<M::ModelSpec>::mapping(IO &io,
                                                      M::ModelSpec &modelSpec) {
  io.mapRequired("name", modelSpec.name);
  if (!io.outputting() || !modelSpec.task.empty())
    io.mapOptional("task", modelSpec.task);
  if (!io.outputting() || !modelSpec.description.empty())
    io.mapOptional("description", modelSpec.description);
  if (!io.outputting() || !modelSpec.data_name.empty())
    io.mapOptional("data_name", modelSpec.data_name);
  if (!io.outputting() || !modelSpec.data_description.empty())
    io.mapOptional("data_description", modelSpec.data_description);
  io.mapRequired("model", modelSpec.model);
  io.mapRequired("inputs", modelSpec.inputs);
  io.mapRequired("outputs", modelSpec.outputs);
  io.mapOptional("metrics", modelSpec.metrics);
  io.mapOptional("compile_specs", modelSpec.compile_specs);
}

namespace {
/// Returns a non-empty error if the name substitutions don't fit `names`.
std::string validateNameSubstitutions(
    StringRef format, StringRef inputOrOutput,
    ArrayRef<M::ModelSpec::ModelIOSpec> names,
    const M::ModelSpec::NameSubstitutions &nameSubstitutions) {
  llvm::StringMap<std::string> nativeToPreferredMap;
  for (const auto &kv : nameSubstitutions.preferredToNativeMap) {
    if (!contains_if(names, [&kv](const M::ModelSpec::ModelIOSpec &spec) {
          return spec.name == kv.getKey();
        })) {
      return (Twine("No such preferred ") + inputOrOutput + " name '" +
              kv.getKey() + "' for " + inputOrOutput +
              " substitution in format " + format)
          .str();
    }
    // The substitution must be bijective.
    if (!nativeToPreferredMap.try_emplace(kv.getValue(), kv.getKey()).second) {
      return (Twine("Already substituted native ") + inputOrOutput + " name '" +
              kv.getValue() + "' in format " + format)
          .str();
    };
  }
  return {};
}
} // namespace

std::string
llvm::yaml::MappingTraits<M::ModelSpec>::validate(IO &io,
                                                  M::ModelSpec &modelSpec) {
  for (const auto &modelBinary : modelSpec.model) {
    std::string err =
        validateNameSubstitutions(modelBinary.format, "input", modelSpec.inputs,
                                  modelBinary.inputSubstitutions);
    if (!err.empty())
      return err;

    err = validateNameSubstitutions(modelBinary.format, "output",
                                    modelSpec.outputs,
                                    modelBinary.outputSubstitutions);
    if (!err.empty())
      return err;
  }
  return {};
}

M::ErrorOr<M::ModelSpec>
M::loadModelSpecFromYAML(std::unique_ptr<llvm::MemoryBuffer> yamlBuffer) {
  llvm::yaml::Input yin(yamlBuffer->getBuffer());
  M::ModelSpec modelSpec;
  yin >> modelSpec;
  if (yin.error())
    return Error("unable to parse the model spec from the YAML buffer");
  return modelSpec;
}

void llvm::yaml::MappingTraits<M::TensorResultInfo>::mapping(
    IO &io, M::TensorResultInfo &info) {
  io.mapRequired("name", info.name);
  io.mapOptional("shape", info.shape);
}

void llvm::yaml::MappingTraits<M::VariantResultInfo>::mapping(
    IO &io, M::VariantResultInfo &info) {
  io.mapRequired("name", info.name);
  io.mapRequired("variant", info.variant);
}

void llvm::yaml::MappingTraits<std::shared_ptr<M::ResultInfo>>::mapping(
    IO &io, std::shared_ptr<M::ResultInfo> &info) {
  info->map(io);
}

ErrorOr<AsyncRT::AnyAsyncValueRef>
M::getResultForBinaryOutput(const AsyncRT::AnyAsyncValueRef &out,
                            ProcessedResults &processedResults) {
  M::AsyncRT::CompactCPUDevicePtr cpuDevice = out.getCPUDevice();

  if (GML::List *castListValue = dyn_cast<GML::List>(out)) {
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<GML::List>(cpuDevice);
    out.get<GML::List>().getElements().reserve(
        castListValue->getElements().size());
    for (GML::Variant &elem : castListValue->getElements()) {
      ErrorOr<AsyncRT::AnyAsyncValueRef> newElemOr =
          getResultForBinaryOutput(elem.getValue(), processedResults);
      if (newElemOr->isError())
        return newElemOr.takeError();
      out.get<GML::List>().getElements().push_back(newElemOr.takeValue());
    }
    return out;
  }
  if (GML::Tuple *castTupleValue = dyn_cast<GML::Tuple>(out)) {
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<GML::Tuple>(cpuDevice);
    out.get<GML::Tuple>().getElements().reserve(
        castTupleValue->getElements().size());
    for (GML::Variant &elem : castTupleValue->getElements()) {
      ErrorOr<AsyncRT::AnyAsyncValueRef> newElemOr =
          getResultForBinaryOutput(elem.getValue(), processedResults);
      if (newElemOr->isError())
        return newElemOr.takeError();
      out.get<GML::Tuple>().getElements().push_back(newElemOr.takeValue());
    }
    return out;
  }
  if (GML::Dict *castDictValue = dyn_cast<GML::Dict>(out)) {
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<GML::Dict>(cpuDevice);
    out.get<GML::Dict>().getElements().reserve(
        castDictValue->getElements().size());
    for (auto &[key, value] : castDictValue->getElements()) {
      ErrorOr<AsyncRT::AnyAsyncValueRef> newKeyOr =
          getResultForBinaryOutput(key.getValue(), processedResults);
      ErrorOr<AsyncRT::AnyAsyncValueRef> newValueOr =
          getResultForBinaryOutput(value.getValue(), processedResults);
      if (newKeyOr.isError())
        return newKeyOr.takeError();
      if (newValueOr.isError())
        return newValueOr.takeError();
      out.get<GML::Dict>().getElements().insert(
          {newKeyOr.takeValue(), newValueOr.takeValue()});
    }
    return out;
  }
  if (Tensor *castTensorValue = dyn_cast<Tensor>(out)) {
    std::string name = processedResults.insertTensor(castTensorValue);
    // Tensors sharing a buffer (pointer and size) are the same result even if
    // a reshape changed their shape.
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<TensorResultInfo>(
            cpuDevice, name, *castTensorValue);
    return out;
  }
  // Copy references to primitive types (string, double, long, bool, None).
  if (isa<std::string>(out) || isa<float>(out) || isa<double>(out) ||
      isa<bool>(out) || isa<int32_t>(out) || isa<int64_t>(out) ||
      isa<uint32_t>(out) || isa<uint64_t>(out) || isa<GML::None>(out))
    return out.copy();
  else
    return Error("unknown result");
}

llvm::yaml::MappingTraits<
    M::GML::Variant>::NormalizedVariant::NormalizedVariant(IO &) {}

llvm::yaml::MappingTraits<M::GML::Variant>::NormalizedVariant::
    NormalizedVariant(IO &, M::GML::Variant &variant) {
  if (auto *castStringValue = dyn_cast<std::string>(variant.getValue()))
    stringValue = *castStringValue;
  else if (auto *castDoubleValue = dyn_cast<double>(variant.getValue()))
    doubleValue = *castDoubleValue;
  else if (auto *castLongValue = dyn_cast<int64_t>(variant.getValue()))
    longValue = *castLongValue;
  else if (auto *castBoolValue = dyn_cast<bool>(variant.getValue()))
    boolValue = *castBoolValue;
  else if (auto *castListValue = dyn_cast<GML::List>(variant.getValue()))
    listValue = *castListValue;
  else if (auto *castTupleValue = dyn_cast<GML::Tuple>(variant.getValue()))
    tupleValue = *castTupleValue;
  else if (auto *castDictValue = dyn_cast<GML::Dict>(variant.getValue()))
    dictValue = *castDictValue;
  else if (auto *castNoneValue = dyn_cast<GML::None>(variant.getValue()))
    noneValue = *castNoneValue;
  else if (auto *castTensorResultInfo =
               dyn_cast<TensorResultInfo>(variant.getValue()))
    tensorResultInfoValue = *castTensorResultInfo;
  // Unsupported types are not mapped.
}

M::GML::Variant
llvm::yaml::MappingTraits<M::GML::Variant>::NormalizedVariant::denormalize(
    IO &) {
  AsyncRT::CPUDevice *cpuDevice = CPUDevice::getCurrentCPUDeviceOrNull();
  M::GML::Variant variant;
  if (stringValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<std::string>(cpuDevice,
                                                                  *stringValue);
  } else if (doubleValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<double>(cpuDevice, *doubleValue);
  } else if (longValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<int64_t>(cpuDevice, *longValue);
  } else if (boolValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<bool>(cpuDevice, *boolValue);
  } else if (listValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<GML::List>(cpuDevice,
                                                                *listValue);
  } else if (tupleValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<GML::Tuple>(cpuDevice,
                                                                 *tupleValue);
  } else if (dictValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<GML::Dict>(cpuDevice,
                                                                *dictValue);
  } else if (noneValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<GML::None>(cpuDevice,
                                                                *noneValue);
  }
  return variant;
}

void llvm::yaml::MappingTraits<M::GML::Variant>::mapping(
    IO &io, M::GML::Variant &variant) {
  MappingNormalization<NormalizedVariant, M::GML::Variant> keys(io, variant);
  io.mapOptional("string", keys->stringValue, std::optional<std::string>());
  io.mapOptional("double", keys->doubleValue, std::optional<double>());
  io.mapOptional("long", keys->longValue, std::optional<int64_t>());
  io.mapOptional("bool", keys->boolValue, std::optional<bool>());
  io.mapOptional("none", keys->noneValue, std::optional<M::GML::None>());
  io.mapOptional("list", keys->listValue, std::optional<M::GML::List>());
  io.mapOptional("tuple", keys->tupleValue, std::optional<M::GML::Tuple>());
  io.mapOptional("dict", keys->dictValue, std::optional<M::GML::Dict>());
  io.mapOptional("tensor-result-info", keys->tensorResultInfoValue,
                 std::optional<TensorResultInfo>());
}

void llvm::yaml::MappingTraits<M::GML::None>::mapping(IO &io,
                                                      M::GML::None &list) {}

void llvm::yaml::MappingTraits<M::GML::List>::mapping(IO &io,
                                                      M::GML::List &list) {
  io.mapRequired("list", list.getElements());
}

void llvm::yaml::MappingTraits<M::GML::Tuple>::mapping(IO &io,
                                                       M::GML::Tuple &tuple) {
  io.mapRequired("tuple", tuple.getElements());
}

void llvm::yaml::MappingTraits<YamlDictKeyValuePair>::mapping(
    IO &io, YamlDictKeyValuePair &kv) {
  io.mapRequired("key", kv.first);
  io.mapRequired("value", kv.second);
}

llvm::yaml::MappingTraits<M::GML::Dict>::NormalizedDict::NormalizedDict(IO &) {}

llvm::yaml::MappingTraits<M::GML::Dict>::NormalizedDict::NormalizedDict(
    IO &, M::GML::Dict &dict) {
  for (auto &entry : dict.getElements())
    normalizedDict.emplace_back(entry.first, entry.second);
}

M::GML::Dict
llvm::yaml::MappingTraits<M::GML::Dict>::NormalizedDict::denormalize(IO &) {
  M::GML::Dict dict;
  for (auto &entry : normalizedDict) {
    dict.getElements().insert(std::make_pair(entry.first, entry.second));
  }
  return dict;
}

void llvm::yaml::MappingTraits<M::GML::Dict>::mapping(IO &io,
                                                      M::GML::Dict &dict) {
  MappingNormalization<NormalizedDict, M::GML::Dict> keys(io, dict);
  io.mapRequired("dict", keys->normalizedDict);
}
