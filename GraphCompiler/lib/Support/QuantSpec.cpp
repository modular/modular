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

#include "GraphCompiler/Support/QuantSpec.h"
#include "MLRT/GraphRT/MEF/MEFTypes.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Stringification and printing methods
//===----------------------------------------------------------------------===//

StringRef QuantSpec::kindAsString(QuantSpec::Kind kind) {
  switch (kind) {
  case QuantSpec::Kind::invalid:
    return "invalid";
  case QuantSpec::Kind::notQuantized:
    return "notQuantized";
  case QuantSpec::Kind::uniform:
    return "uniform";
  case QuantSpec::Kind::uniformPerAxis:
    return "uniformPerAxis";
  }
  llvm_unreachable("unknown QuantSpec kind");
}

QuantSpec::Kind QuantSpec::kindFromString(StringRef kind) {
  return llvm::StringSwitch<QuantSpec::Kind>(kind)
      .Case("invalid", QuantSpec::Kind::invalid)
      .Case("notQuantized", QuantSpec::Kind::notQuantized)
      .Case("uniform", QuantSpec::Kind::uniform)
      .Case("uniformPerAxis", QuantSpec::Kind::uniformPerAxis)
      .Default(QuantSpec::Kind::invalid);
}

/// Returns the kind, plus the quantized dimension if relevant to the kind.
static std::string getKindAndDimAsString(const QuantSpec *qSpec) {
  StringRef kind = QuantSpec::kindAsString(qSpec->getKind());
  if (qSpec->getKind() == QuantSpec::Kind::uniformPerAxis)
    return (Twine(kind) + ":" + Twine(qSpec->getQuantDim())).str();
  return kind.str();
}

void QuantSpec::print(raw_ostream &os) const {
  os << getKindAndDimAsString(this);
  auto params = getParams();
  if (!params.empty()) {
    os << " {";
    llvm::interleave(
        getParams(),
        [&](ScaleZeroPointPair p) { os << p.scale << ":" << p.zeroPoint; },
        [&]() { os << ", "; });
    os << "}";
  }
}

std::string QuantSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void QuantSpec::dump() const { print(llvm::errs()); }

void QuantTensorSpec::print(raw_ostream &os) const {
  TensorSpec::print(os);
  if (isQuantized()) {
    os << " ";
    getQuantSpec().print(os);
  }
}

std::string QuantTensorSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void QuantTensorSpec::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// Static helpers
//===----------------------------------------------------------------------===//

QuantSpec QuantSpec::getUniformPerAxis(uint8_t quantDim, ArrayRef<float> scales,
                                       ArrayRef<int64_t> zeroPoints) {
  // TODO: change this to take ArrayRef<ScaleZeroPointPair>, and change the
  // MEF encoding to encode similarly.
  return QuantSpec(
      Kind::uniformPerAxis, quantDim,
      llvm::map_to_vector(llvm::zip(scales, zeroPoints), [](auto paramPair) {
        auto [scale, zeroPoint] = paramPair;
        return ScaleZeroPointPair{scale, static_cast<int16_t>(zeroPoint)};
      }));
}

ErrorOr<QuantTensorSpec> QuantTensorSpec::get(MEFType type) {
  auto tensorType = dyn_cast<MEFTensorType>(type);
  if (!tensorType)
    return Error("MEFType cannot be cast to MEFTensorType");

  bool hasRank;
  SmallVector<int64_t> dims;
  auto mefType = tensorType->decode(hasRank, dims);
  TensorRankStyle style = hasRank ? kStaticallyRanked : kDynamicallyRanked;

  if (auto qType = dyn_cast<MEFQuantizedType>(mefType)) {
    float scale;
    int64_t zeroPoint;
    auto [storageType, expressedType] = qType->decode(scale, zeroPoint);
    return QuantTensorSpec(TensorSpec(dims, style, storageType.getKind()),
                           QuantSpec::getUniform(scale, zeroPoint));
  }
  if (auto qType = dyn_cast<MEFQuantizedPerAxisType>(mefType)) {
    SmallVector<float> scales;
    SmallVector<int64_t> zeroPoints;
    uint8_t quantDim;
    auto [storageType, expressedType] =
        qType->decode(scales, zeroPoints, quantDim);

    return QuantTensorSpec(
        TensorSpec(dims, style, storageType.getKind()),
        QuantSpec::getUniformPerAxis(quantDim, scales, zeroPoints));
  }
  // MEFType kinds are GraphCompiler DType kinds.
  return QuantTensorSpec(TensorSpec(dims, style, mefType.getKind()), {});
}
