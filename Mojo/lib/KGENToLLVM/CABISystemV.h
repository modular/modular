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

#ifndef KGEN_LIB_KGENTOLLVM_CABISYSTEMV_H
#define KGEN_LIB_KGENTOLLVM_CABISYSTEMV_H

#include "CABILowering.h"

namespace M::KGEN {

/// Implementation of x86-64 System V AMD64 ABI calling convention.
///
/// This class handles classification and coercion of struct types according
/// to the System V ABI rules, which are the standard on Linux, BSD, and macOS.
///
/// Key rules:
/// - Structs ≤8 bytes (all-integer): coerce to i8/i16/i32/i64
/// - Structs 9-16 bytes: split into two eightbytes, classify each
/// - Structs >16 bytes: pass by pointer (MEMORY class)
/// - Float/SSE handling: classify per eightbyte
/// - Variadic arguments: on x86-64, variadic args always use stack
class SystemVABIInfo : public CABIInfo {
public:
  SystemVABIInfo(mlir::MLIRContext *ctx, const LLVMDataLayout &dataLayout);

protected:
  /// Classify a whole signature, applying the rollback-to-stack rule: an
  /// aggregate that does not fit in the remaining argument registers is
  /// passed entirely in memory rather than split.
  SignatureClassification classifySignature(mlir::TypeRange argTypes,
                                            mlir::Type retType,
                                            mlir::Location loc,
                                            size_t numFixedArgs) const override;

  CoercionInfo classifyArgumentType(mlir::Type type, mlir::Location loc,
                                    bool isVariadicArg) const override;

  CoercionInfo classifyReturnType(mlir::Type type,
                                  mlir::Location loc) const override;

private:
  /// Number of integer and SSE argument registers a classified argument
  /// consumes. Identity (scalar/pointer) args are derived from their type.
  std::pair<unsigned, unsigned> argRegisterUsage(const CoercionInfo &info,
                                                 mlir::Type llvmType) const;

  /// Common classification logic for both arguments and return values.
  /// \param useSRet If true, >16 byte structs use sret; otherwise indirect.
  CoercionInfo classifyStructType(mlir::Type type, mlir::Location loc,
                                  bool useSRet) const;

  //===--------------------------------------------------------------------===//
  // Eightbyte classification (structs up to 16 bytes)
  //===--------------------------------------------------------------------===//

  /// Classification for each 8-byte chunk of a struct.
  enum class EightbyteClass {
    NoClass, // Uninitialized
    Integer, // Integer types (passed in GP registers)
    SSE,     // Float/double types (passed in SSE registers)
    SSEUp,   // Upper half of a 128-bit vector, in the same SSE register
    Memory   // Must use memory (stack)
  };

  /// Get the LLVM type to use for an eightbyte.
  ///
  /// \param eightbyteClass The classification (Integer or SSE)
  /// \param size Number of bytes in this eightbyte (1-8)
  /// \return Appropriate integer (i8/i16/i32/i64) or float (f32/f64) type
  mlir::Type getEightbyteType(EightbyteClass eightbyteClass,
                              int64_t size) const;

  /// Merge the class of one more member into its eightbyte's class
  /// (System V ABI 3.2.3p2, rule 4).
  ///
  /// Members that share an eightbyte share one register, so the higher class
  /// wins: NoClass < SSEUp < SSE < Integer < Memory.
  ///
  /// \param accum The class of the members already in the eightbyte
  /// \param field The class of the new member
  /// \return The combined class
  static EightbyteClass mergeEightbyteClass(EightbyteClass accum,
                                            EightbyteClass field);

  /// Classify one member of a struct of at most 16 bytes.
  ///
  /// Floats up to 64 bits are SSE. Vectors are classified by size: up to 32
  /// bits Integer, 64 bits SSE, 128 bits SSE with SSEUp for the upper half,
  /// other sizes Memory. Anything else is Integer. A member that is not
  /// aligned is Memory.
  ///
  /// \param type The member type: a scalar, a vector, or a type without size
  /// \param offset Byte offset of the member in the struct
  /// \return The classes the member gives to bytes 0-7 and to bytes 8-15;
  ///         NoClass for an eightbyte it does not cover
  std::pair<EightbyteClass, EightbyteClass>
  classifyMember(mlir::Type type, int64_t offset) const;
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABISYSTEMV_H
