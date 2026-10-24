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
// This file declares the ResultInfo and related classes, which hold
// model result information for outputs in a framework-independent way.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_RESULTINFO_H
#define GENERICML_SUPPORT_RESULTINFO_H

#include "GraphCompiler/Support/Value.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/ML/TensorSpec.h"
#include <map>

namespace M {

/// Output information written to YAML when --result-output-style=binary.
class ResultInfo {
public:
  ResultInfo() = default;
  ResultInfo(std::string name);
  virtual ~ResultInfo() = default;

  std::string name;

  virtual void map(llvm::yaml::IO &io) = 0;
};

class TensorResultInfo : public ResultInfo {
public:
  TensorResultInfo() = default;
  TensorResultInfo(std::string name, std::string shape);
  TensorResultInfo(const TensorResultInfo &other) = default;
  TensorResultInfo(std::string name, const MLRT::Tensor &tensor);

  TensorResultInfo operator=(TensorResultInfo other);
  TensorResultInfo &operator=(TensorResultInfo &other);

  void map(llvm::yaml::IO &io) override;

  std::string shape;
};

class VariantResultInfo : public M::ResultInfo {
public:
  VariantResultInfo(std::string name, M::GML::Variant variant);

  VariantResultInfo(const VariantResultInfo &) = default;
  VariantResultInfo(VariantResultInfo &&) = default;
  VariantResultInfo &operator=(const VariantResultInfo &) = default;
  VariantResultInfo &operator=(VariantResultInfo &&) = default;

  void map(llvm::yaml::IO &io) override;

  M::GML::Variant variant;
};

/// Per-output `ResultInfo`s plus a flat map of the unique tensors "unzipped"
/// from nested (Dict, List, Tuple) model outputs for export. A tensor output
/// keeps the output's name; a leaf tensor is assigned one.
class ProcessedResults {
public:
  ProcessedResults();

  /// Inserts a `TensorResultInfo` (tensor output) or `VariantResultInfo`
  /// (composed output).
  void insertResultInfo(std::shared_ptr<ResultInfo> info);

  void insertTensor(const std::string &name, MLRT::Tensor *tensor);

  /// Names and inserts `tensor`, returning the name; reuses the existing name
  /// of a tensor with the same buffer.
  std::string insertTensor(MLRT::Tensor *tensor);

  /// Returns true if a tensor with the same buffer exists in the results.
  bool containsTensor(MLRT::Tensor *tensor) const;

  /// Returns the first name registered for a tensor with the same buffer, or
  /// an empty string.
  std::string getTensorName(MLRT::Tensor *) const;

  SmallVector<std::shared_ptr<ResultInfo>> &getResultInfos();

  const std::map<std::string, MLRT::Tensor *> &getTensors() const;

private:
  SmallVector<std::shared_ptr<ResultInfo>> resultInfos;
  // Counts unnamed binary outputs.
  size_t counter;
  std::map<std::string, MLRT::Tensor *> tensors;
};

} // namespace M

#endif // GENERICML_SUPPORT_RESULTINFO_H
