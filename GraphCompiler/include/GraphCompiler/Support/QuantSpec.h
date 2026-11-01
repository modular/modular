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
// This file declares the QuantSpec and related classes, which hold
// quantization information for tensors in a framework-independent way.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_QUANTSPEC_H
#define GENERICML_SUPPORT_QUANTSPEC_H

#include "Support/ML/TensorSpec.h"

namespace M {
class MEFType;

/// Framework-independent quantization info for a single tensor.
class QuantSpec {
public:
  enum Kind : uint8_t {
    invalid,
    notQuantized,
    uniform,
    uniformPerAxis,
  };

  /// Affine quantization parameters mapping a stored value `v_q` to the
  /// expressed value `v = (v_q - zeroPoint) * scale`. `zeroPoint` is typically,
  /// but not necessarily, within the storage type's range.
  struct ScaleZeroPointPair {
    float scale;
    int16_t zeroPoint;
  };

  QuantSpec() : kind(notQuantized) {}
  QuantSpec(Kind kind, uint8_t quantDim, ArrayRef<ScaleZeroPointPair> params)
      : kind(kind), quantDim(quantDim), params(params.begin(), params.end()) {}

  /// Creates a QuantSpec of uniform kind from a single scale and zero point.
  static inline QuantSpec getUniform(float scale, int16_t zeroPoint) {
    return QuantSpec(Kind::uniform, 0, {{scale, zeroPoint}});
  }

  /// Creates a QuantSpec of uniformPerAxis kind from arrays of scales and zero
  /// points.
  static QuantSpec getUniformPerAxis(uint8_t quantDim, ArrayRef<float> scales,
                                     ArrayRef<int64_t> zeroPoints);

  /// Returns a string representation of the kind enum.
  static StringRef kindAsString(QuantSpec::Kind kind);

  /// Returns a QuantSpec::Kind enum given a string representation.
  static QuantSpec::Kind kindFromString(StringRef kind);

  /// Returns the paired scales and zero points.
  ArrayRef<ScaleZeroPointPair> getParams() const { return params; }

  Kind getKind() const { return kind; }

  /// The quantized dimension is undefined when kind is invalid or notQuantized.
  uint8_t getQuantDim() const { return quantDim; }

  bool isValid() const { return kind != Kind::invalid; }

  /// Returns false if the QuantSpec is invalid or not quantized.
  bool isQuantized() const { return isValid() && kind != notQuantized; }

  std::string getAsString() const;
  void print(raw_ostream &os) const;
  void dump() const;

private:
  Kind kind;
  uint8_t quantDim = 0;

  // Sized for the common uniform case: one scale and one zero point.
  SmallVector<ScaleZeroPointPair, 1> params;
};

inline raw_ostream &operator<<(raw_ostream &os, const QuantSpec &value) {
  value.print(os);
  return os;
}

/// TensorSpec plus quantization info, which may be notQuantized.
class QuantTensorSpec : public TensorSpec {
private:
  QuantTensorSpec(const QuantTensorSpec &spec, const TensorShape &newShape)
      : TensorSpec(newShape, spec.getEltType()),
        quantSpec(spec.getQuantSpec()) {}

public:
  QuantTensorSpec() : TensorSpec() { setEltType(DType::invalid); }
  QuantTensorSpec(TensorSpec &&tensorSpec) : TensorSpec(tensorSpec) {}
  QuantTensorSpec(TensorSpec &&tensorSpec, QuantSpec &&quantSpec)
      : TensorSpec(tensorSpec), quantSpec(quantSpec) {}

  const QuantSpec &getQuantSpec() const { return quantSpec; }

  /// Returns true if QuantTensorSpec describes a truly quantized tensor.
  bool isQuantized() const { return quantSpec.isQuantized(); }

  /// Returns new QuantTensorSpec with identical attributes save for newShape.
  QuantTensorSpec cloneWithNewShape(const TensorShape &newShape) const {
    return QuantTensorSpec(*this, newShape);
  }

  /// Helper method to decode a MEFType into a TensorSpec, if possible.
  static ErrorOr<QuantTensorSpec> get(MEFType type);

  std::string getAsString() const;
  void print(raw_ostream &os) const;
  void dump() const;

private:
  QuantSpec quantSpec;
};

inline raw_ostream &operator<<(raw_ostream &os, const QuantTensorSpec &value) {
  value.print(os);
  return os;
}

} // namespace M

#endif // GENERICML_SUPPORT_QUANTSPEC_H
