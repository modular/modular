//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_CABIAAPCS_H
#define KGEN_LIB_KGENTOLLVM_CABIAAPCS_H

#include "CABILowering.h"
#include <optional>

namespace M::KGEN {

/// Implementation of ARM64 AAPCS calling convention.
///
/// This class handles classification and coercion of struct types according
/// to the AAPCS rules, which are the standard on ARM64 (Linux, macOS, iOS).
///
/// Key rules:
/// - Structs ≤8 bytes (all-integer): coerce to i64 (iN when returned)
/// - Structs 9-16 bytes: two registers (X0-X1), as [2 x i64] or i128
/// - Structs >16 bytes: pass by pointer (indirect)
/// - HFA (Homogeneous Float Aggregate): up to 4 float/vector fields in V0-V3
/// - **Darwin vs Linux difference for variadic HFA:**
///   Darwin (macOS/iOS): va_list is flat (GP/stack only); HFA variadic args
///   must be coerced to integer so va_arg reads from GP save area.
///   Linux AAPCS64: va_list has separate GP and VR save areas; HFA variadic
///   args stay HFAs so va_arg reads correctly from VR save area.
// TODO: A proposed LLVM-native ABI lowering library (LLVMABI) would replace
// this class. See:
// https://discourse.llvm.org/t/rfc-an-abi-lowering-library-for-llvm/84495
class AAPCSABIInfo : public CABIInfo {
public:
  AAPCSABIInfo(mlir::MLIRContext *ctx, const LLVMDataLayout &dataLayout,
               bool isDarwinOS);

protected:
  CoercionInfo classifyArgumentType(mlir::Type type, mlir::Location loc,
                                    bool isVariadicArg) const override;

  CoercionInfo classifyReturnType(mlir::Type type,
                                  mlir::Location loc) const override;

private:
  //===--------------------------------------------------------------------===//
  // Phase 2: Two-register classification (9-16 byte structs)
  //===--------------------------------------------------------------------===//

  /// Classify a 9-16 byte struct into two registers (Phase 2 implementation).
  ///
  /// ARM64 AAPCS is simpler than x86-64: the struct takes two integer
  /// registers (X0-X7), whatever its fields. It is passed as one value, to
  /// prevent a split between a register and the stack.
  ///
  /// \param type The struct (9-16 bytes, not an HFA or HVA)
  /// \return Coercion with coercedType
  CoercionInfo classifyTwoRegisterStruct(mlir::LLVM::LLVMStructType type) const;

  bool isDarwin; // true = Darwin (macOS/iOS), false = Linux/other

  //===--------------------------------------------------------------------===//
  // HFA and HVA detection
  //===--------------------------------------------------------------------===//

  /// A homogeneous aggregate: a struct whose members are all alike, so it is
  /// passed in SIMD registers, one member per register. It is an HFA if the
  /// members are floats and an HVA if they are vectors.
  struct HomogeneousAggregate {
    /// The type of the first member. The others have its size.
    mlir::Type base;
    /// The number of members, 1 to 4.
    int64_t numMembers;
  };

  /// Classify an HFA or HVA argument: an array of its members.
  static CoercionInfo
  classifyHomogeneousAggregate(const HomogeneousAggregate &aggregate);

  /// Helper that detects `Array[T, 0]`, `Array[Array[T, 0], N]` and so on.
  /// A struct with such a member is not an HFA or HVA.
  static bool isZeroLengthArray(mlir::Type type);

  /// Check whether a struct is an HFA or HVA: 1 to 4 members that are all
  /// floats of one size, or all vectors of 64 or 128 bits. Members are
  /// counted through nested structs and arrays.
  ///
  /// \return The first member's type and the number of members, or
  ///         std::nullopt if the struct is not an HFA or HVA
  std::optional<HomogeneousAggregate>
  getHomogeneousAggregate(mlir::LLVM::LLVMStructType type) const;
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABIAAPCS_H
