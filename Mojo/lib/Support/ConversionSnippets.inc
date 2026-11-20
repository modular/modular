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
//
// GENERATED FILE - DO NOT EDIT. Regeneration steps: PrecompiledMLIR.cpp.
//
//===----------------------------------------------------------------------===//

namespace {

struct ConversionSnippet {
  DType::Cases from;
  DType::Cases to;
  llvm::StringLiteral symbolName;
  llvm::StringLiteral mlirText;
};

static constexpr llvm::StringLiteral kFloat8ToF32_f8e4m3fn = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e4m3fn_to_f32(%arg0: i8 {llvm.noundef}) -> f32 {
    %0 = llvm.mlir.constant(3 : i8) : i8
    %1 = llvm.mlir.constant(15 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(20 : i64) : i64
    %5 = llvm.mlir.constant(120 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(3 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(7 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(127 : i8) : i8
    %20 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %21 = llvm.mlir.constant(-2147483649 : i64) : i64
    %22 = llvm.mlir.constant(2147483648 : i64) : i64
    %23 = llvm.lshr %arg0, %0 : i8
    %24 = llvm.and %23, %1 : i8
    %25 = llvm.zext %24 : i8 to i64
    %26 = llvm.sext %arg0 : i8 to i64
    %27 = llvm.and %26, %13 : i64
    %28 = llvm.xor %27, %11 : i64
    %29 = llvm.icmp "slt" %27, %8 : i64
    %30 = llvm.select %29, %28, %27 fastmath<contract> : i1, i64
    %31 = llvm.call_intrinsic "llvm.ctlz"(%30, %9) : (i64, i1) -> i64
    %32 = llvm.sub %12, %31 : i64
    %33 = llvm.sub %7, %32 : i64
    %34 = llvm.add %33, %6 : i64
    %35 = llvm.shl %27, %34 : i64
    %36 = llvm.and %35, %13 : i64
    %37 = llvm.sub %25, %33 : i64
    %38 = llvm.icmp "eq" %27, %8 : i64
    %39 = llvm.xor %38, %10 : i1
    %40 = llvm.icmp "eq" %25, %8 : i64
    %41 = llvm.select %40, %39, %9 fastmath<contract> : i1, i1
    %42 = llvm.select %41, %37, %25 fastmath<contract> : i1, i64
    %43 = llvm.select %41, %36, %27 fastmath<contract> : i1, i64
    %44 = llvm.add %42, %5 : i64
    %45 = llvm.shl %43, %4 : i64
    %46 = llvm.shl %44, %15 : i64
    %47 = llvm.and %46, %14 : i64
    %48 = llvm.or %47, %8 : i64
    %49 = llvm.trunc %48 : i64 to i32
    %50 = llvm.sext %49 : i32 to i64
    %51 = llvm.and %50, %16 : i64
    %52 = llvm.and %45, %17 : i64
    %53 = llvm.or %51, %52 : i64
    %54 = llvm.trunc %53 : i64 to i32
    %55 = llvm.bitcast %54 : i32 to f32
    %56 = llvm.and %arg0, %19 : i8
    %57 = llvm.icmp "eq" %56, %2 : i8
    %58 = llvm.select %57, %3, %55 fastmath<contract> : i1, f32
    %59 = llvm.icmp "eq" %56, %19 : i8
    %60 = llvm.select %59, %20, %58 fastmath<contract> : i1, f32
    %61 = llvm.and %26, %18 : i64
    %62 = llvm.icmp "eq" %61, %8 : i64
    %63 = llvm.xor %62, %10 : i1
    %64 = llvm.bitcast %60 : f32 to i32
    %65 = llvm.sext %64 : i32 to i64
    %66 = llvm.and %65, %21 : i64
    %67 = llvm.or %66, %22 : i64
    %68 = llvm.select %63, %67, %66 fastmath<contract> : i1, i64
    %69 = llvm.trunc %68 : i64 to i32
    %70 = llvm.bitcast %69 : i32 to f32
    llvm.return %70 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kFloat8ToF32_f8e5m2 = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e5m2_to_f32(%arg0: i8 {llvm.noundef}) -> f32 {
    %0 = llvm.mlir.constant(2 : i8) : i8
    %1 = llvm.mlir.constant(31 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(21 : i64) : i64
    %5 = llvm.mlir.constant(112 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(3 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(124 : i8) : i8
    %20 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %21 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %22 = llvm.mlir.constant(-2147483649 : i64) : i64
    %23 = llvm.mlir.constant(2147483648 : i64) : i64
    %24 = llvm.mlir.constant(127 : i8) : i8
    %25 = llvm.lshr %arg0, %0 : i8
    %26 = llvm.and %25, %1 : i8
    %27 = llvm.zext %26 : i8 to i64
    %28 = llvm.sext %arg0 : i8 to i64
    %29 = llvm.and %28, %13 : i64
    %30 = llvm.xor %29, %11 : i64
    %31 = llvm.icmp "slt" %29, %8 : i64
    %32 = llvm.select %31, %30, %29 fastmath<contract> : i1, i64
    %33 = llvm.call_intrinsic "llvm.ctlz"(%32, %9) : (i64, i1) -> i64
    %34 = llvm.sub %12, %33 : i64
    %35 = llvm.sub %7, %34 : i64
    %36 = llvm.add %35, %6 : i64
    %37 = llvm.shl %29, %36 : i64
    %38 = llvm.and %37, %13 : i64
    %39 = llvm.sub %27, %35 : i64
    %40 = llvm.icmp "eq" %29, %8 : i64
    %41 = llvm.xor %40, %10 : i1
    %42 = llvm.icmp "eq" %27, %8 : i64
    %43 = llvm.select %42, %41, %9 fastmath<contract> : i1, i1
    %44 = llvm.select %43, %39, %27 fastmath<contract> : i1, i64
    %45 = llvm.select %43, %38, %29 fastmath<contract> : i1, i64
    %46 = llvm.add %44, %5 : i64
    %47 = llvm.shl %45, %4 : i64
    %48 = llvm.shl %46, %15 : i64
    %49 = llvm.and %48, %14 : i64
    %50 = llvm.or %49, %8 : i64
    %51 = llvm.trunc %50 : i64 to i32
    %52 = llvm.sext %51 : i32 to i64
    %53 = llvm.and %52, %16 : i64
    %54 = llvm.and %47, %17 : i64
    %55 = llvm.or %53, %54 : i64
    %56 = llvm.trunc %55 : i64 to i32
    %57 = llvm.bitcast %56 : i32 to f32
    %58 = llvm.and %arg0, %24 : i8
    %59 = llvm.icmp "ugt" %58, %19 : i8
    %60 = llvm.icmp "eq" %58, %2 : i8
    %61 = llvm.select %60, %3, %57 fastmath<contract> : i1, f32
    %62 = llvm.select %59, %21, %61 fastmath<contract> : i1, f32
    %63 = llvm.icmp "eq" %58, %19 : i8
    %64 = llvm.select %63, %20, %62 fastmath<contract> : i1, f32
    %65 = llvm.and %28, %18 : i64
    %66 = llvm.icmp "eq" %65, %8 : i64
    %67 = llvm.xor %66, %10 : i1
    %68 = llvm.bitcast %64 : f32 to i32
    %69 = llvm.sext %68 : i32 to i64
    %70 = llvm.and %69, %22 : i64
    %71 = llvm.or %70, %23 : i64
    %72 = llvm.select %67, %71, %70 fastmath<contract> : i1, i64
    %73 = llvm.trunc %72 : i64 to i32
    %74 = llvm.bitcast %73 : i32 to f32
    llvm.return %74 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kFloat8ToF32_f8e4m3fnuz = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e4m3fnuz_to_f32(%arg0: i8 {llvm.noundef}) -> f32 {
    %0 = llvm.mlir.constant(3 : i8) : i8
    %1 = llvm.mlir.constant(15 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(20 : i64) : i64
    %5 = llvm.mlir.constant(119 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(3 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(7 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(-2147483649 : i64) : i64
    %20 = llvm.mlir.constant(2147483648 : i64) : i64
    %21 = llvm.mlir.constant(-128 : i8) : i8
    %22 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %23 = llvm.mlir.constant(127 : i8) : i8
    %24 = llvm.lshr %arg0, %0 : i8
    %25 = llvm.and %24, %1 : i8
    %26 = llvm.zext %25 : i8 to i64
    %27 = llvm.sext %arg0 : i8 to i64
    %28 = llvm.and %27, %13 : i64
    %29 = llvm.xor %28, %11 : i64
    %30 = llvm.icmp "slt" %28, %8 : i64
    %31 = llvm.select %30, %29, %28 fastmath<contract> : i1, i64
    %32 = llvm.call_intrinsic "llvm.ctlz"(%31, %9) : (i64, i1) -> i64
    %33 = llvm.sub %12, %32 : i64
    %34 = llvm.sub %7, %33 : i64
    %35 = llvm.add %34, %6 : i64
    %36 = llvm.shl %28, %35 : i64
    %37 = llvm.and %36, %13 : i64
    %38 = llvm.sub %26, %34 : i64
    %39 = llvm.icmp "eq" %28, %8 : i64
    %40 = llvm.xor %39, %10 : i1
    %41 = llvm.icmp "eq" %26, %8 : i64
    %42 = llvm.select %41, %40, %9 fastmath<contract> : i1, i1
    %43 = llvm.select %42, %38, %26 fastmath<contract> : i1, i64
    %44 = llvm.select %42, %37, %28 fastmath<contract> : i1, i64
    %45 = llvm.add %43, %5 : i64
    %46 = llvm.shl %44, %4 : i64
    %47 = llvm.shl %45, %15 : i64
    %48 = llvm.and %47, %14 : i64
    %49 = llvm.or %48, %8 : i64
    %50 = llvm.trunc %49 : i64 to i32
    %51 = llvm.sext %50 : i32 to i64
    %52 = llvm.and %51, %16 : i64
    %53 = llvm.and %46, %17 : i64
    %54 = llvm.or %52, %53 : i64
    %55 = llvm.trunc %54 : i64 to i32
    %56 = llvm.bitcast %55 : i32 to f32
    %57 = llvm.and %arg0, %23 : i8
    %58 = llvm.icmp "eq" %57, %2 : i8
    %59 = llvm.select %58, %3, %56 fastmath<contract> : i1, f32
    %60 = llvm.and %27, %18 : i64
    %61 = llvm.icmp "eq" %60, %8 : i64
    %62 = llvm.xor %61, %10 : i1
    %63 = llvm.bitcast %59 : f32 to i32
    %64 = llvm.sext %63 : i32 to i64
    %65 = llvm.and %64, %19 : i64
    %66 = llvm.or %65, %20 : i64
    %67 = llvm.select %62, %66, %65 fastmath<contract> : i1, i64
    %68 = llvm.trunc %67 : i64 to i32
    %69 = llvm.bitcast %68 : i32 to f32
    %70 = llvm.icmp "eq" %arg0, %21 : i8
    %71 = llvm.select %70, %22, %69 fastmath<contract> : i1, f32
    llvm.return %71 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kFloat8ToF32_f8e5m2fnuz = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e5m2fnuz_to_f32(%arg0: i8 {llvm.noundef}) -> f32 {
    %0 = llvm.mlir.constant(2 : i8) : i8
    %1 = llvm.mlir.constant(31 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(21 : i64) : i64
    %5 = llvm.mlir.constant(111 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(3 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(-2147483649 : i64) : i64
    %20 = llvm.mlir.constant(2147483648 : i64) : i64
    %21 = llvm.mlir.constant(-128 : i8) : i8
    %22 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %23 = llvm.mlir.constant(127 : i8) : i8
    %24 = llvm.lshr %arg0, %0 : i8
    %25 = llvm.and %24, %1 : i8
    %26 = llvm.zext %25 : i8 to i64
    %27 = llvm.sext %arg0 : i8 to i64
    %28 = llvm.and %27, %13 : i64
    %29 = llvm.xor %28, %11 : i64
    %30 = llvm.icmp "slt" %28, %8 : i64
    %31 = llvm.select %30, %29, %28 fastmath<contract> : i1, i64
    %32 = llvm.call_intrinsic "llvm.ctlz"(%31, %9) : (i64, i1) -> i64
    %33 = llvm.sub %12, %32 : i64
    %34 = llvm.sub %7, %33 : i64
    %35 = llvm.add %34, %6 : i64
    %36 = llvm.shl %28, %35 : i64
    %37 = llvm.and %36, %13 : i64
    %38 = llvm.sub %26, %34 : i64
    %39 = llvm.icmp "eq" %28, %8 : i64
    %40 = llvm.xor %39, %10 : i1
    %41 = llvm.icmp "eq" %26, %8 : i64
    %42 = llvm.select %41, %40, %9 fastmath<contract> : i1, i1
    %43 = llvm.select %42, %38, %26 fastmath<contract> : i1, i64
    %44 = llvm.select %42, %37, %28 fastmath<contract> : i1, i64
    %45 = llvm.add %43, %5 : i64
    %46 = llvm.shl %44, %4 : i64
    %47 = llvm.shl %45, %15 : i64
    %48 = llvm.and %47, %14 : i64
    %49 = llvm.or %48, %8 : i64
    %50 = llvm.trunc %49 : i64 to i32
    %51 = llvm.sext %50 : i32 to i64
    %52 = llvm.and %51, %16 : i64
    %53 = llvm.and %46, %17 : i64
    %54 = llvm.or %52, %53 : i64
    %55 = llvm.trunc %54 : i64 to i32
    %56 = llvm.bitcast %55 : i32 to f32
    %57 = llvm.and %arg0, %23 : i8
    %58 = llvm.icmp "eq" %57, %2 : i8
    %59 = llvm.select %58, %3, %56 fastmath<contract> : i1, f32
    %60 = llvm.and %27, %18 : i64
    %61 = llvm.icmp "eq" %60, %8 : i64
    %62 = llvm.xor %61, %10 : i1
    %63 = llvm.bitcast %59 : f32 to i32
    %64 = llvm.sext %63 : i32 to i64
    %65 = llvm.and %64, %19 : i64
    %66 = llvm.or %65, %20 : i64
    %67 = llvm.select %62, %66, %65 fastmath<contract> : i1, i64
    %68 = llvm.trunc %67 : i64 to i32
    %69 = llvm.bitcast %68 : i32 to f32
    %70 = llvm.icmp "eq" %arg0, %21 : i8
    %71 = llvm.select %70, %22, %69 fastmath<contract> : i1, f32
    llvm.return %71 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kF32ToFloat8_f8e4m3fn = R"MLIR8(module {
    llvm.func internal @__kgen_pop_cast_f32_to_f8e4m3fn(%arg0: f32) -> i8 {
      %0 = llvm.mlir.constant(0 : i64) : i64
      %1 = llvm.mlir.constant(2147483648 : i64) : i64
      %2 = llvm.mlir.constant(false) : i1
      %3 = llvm.mlir.constant(true) : i1
      %4 = llvm.mlir.constant(1 : i8) : i8
      %5 = llvm.mlir.constant(7 : i8) : i8
      %6 = llvm.mlir.constant(1 : i32) : i32
      %7 = llvm.mlir.constant(0 : i32) : i32
      %8 = llvm.mlir.constant(126 : i8) : i8
      %9 = llvm.mlir.constant(0 : i8) : i8
      %10 = llvm.mlir.constant(-128 : i8) : i8
      %11 = llvm.mlir.constant(255 : i32) : i32
      %12 = llvm.mlir.constant(23 : i32) : i32
      %13 = llvm.mlir.constant(8388607 : i64) : i64
      %14 = llvm.mlir.constant(3 : i32) : i32
      %15 = llvm.mlir.constant(516 : i32) : i32
      %16 = llvm.mlir.constant(127 : i32) : i32
      %17 = llvm.mlir.constant(0x7F : f8E4M3FN) : i8
      %18 = llvm.mlir.constant(4.480000e+02 : f32) : f32
      %19 = llvm.mlir.constant(-6 : i32) : i32
      %20 = llvm.mlir.constant(7 : i32) : i32
      %21 = llvm.mlir.constant(15 : i32) : i32
      %22 = llvm.mlir.constant(20 : i32) : i32
      %23 = llvm.mlir.constant(32 : i32) : i32
      %24 = llvm.mlir.constant(8388608 : i32) : i32
      %25 = llvm.mlir.constant(19 : i32) : i32
      %26 = llvm.mlir.constant(524287 : i32) : i32
      %27 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
      %28 = llvm.fcmp "oge" %27, %18 : f32
      %29 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %15) : (f32, i32) -> i1
      %30 = llvm.bitcast %arg0 : f32 to i32
      %31 = llvm.sext %30 : i32 to i64
      %32 = llvm.and %31, %1 : i64
      %33 = llvm.icmp "eq" %32, %0 : i64
      %34 = llvm.xor %33, %3 : i1
      %35 = llvm.select %34, %10, %9 : i1, i8
      %36 = llvm.or %35, %8 : i8
      %37 = llvm.lshr %30, %12 : i32
      %38 = llvm.and %37, %11 : i32
      %39 = llvm.sub %38, %16 : i32
      %40 = llvm.sub %19, %39 : i32
      %41 = llvm.shl %6, %40 : i32
      %42 = llvm.sub %41, %6 : i32
      %43 = llvm.icmp "slt" %40, %23 : i32
      %44 = llvm.add %39, %20 : i32
      %45 = llvm.and %44, %21 : i32
      %46 = llvm.shl %45, %14 : i32
      %47 = llvm.trunc %46 : i32 to i8
      %48 = llvm.icmp "sge" %39, %19 : i32
      %49 = llvm.and %31, %13 : i64
      %50 = llvm.trunc %49 : i64 to i32
      %51 = llvm.or %50, %24 : i32
      %52 = llvm.ashr %51, %40 : i32
      %53 = llvm.select %43, %52, %7 : i1, i32
      %54 = llvm.select %48, %50, %53 : i1, i32
      %55 = llvm.and %54, %26 : i32
      %56 = llvm.icmp "ne" %55, %7 : i32
      %57 = llvm.select %56, %6, %7 : i1, i32
      %58 = llvm.ashr %54, %25 : i32
      %59 = llvm.and %58, %6 : i32
      %60 = llvm.icmp "ne" %59, %7 : i32
      %61 = llvm.ashr %52, %22 : i32
      %62 = llvm.trunc %61 : i32 to i8
      %63 = llvm.and %62, %5 : i8
      %64 = llvm.select %43, %63, %9 : i1, i8
      %65 = llvm.and %51, %42 : i32
      %66 = llvm.icmp "ne" %65, %7 : i32
      %67 = llvm.select %66, %6, %7 : i1, i32
      %68 = llvm.select %43, %67, %7 : i1, i32
      %69 = llvm.select %48, %7, %68 : i1, i32
      %70 = llvm.or %69, %57 : i32
      %71 = llvm.select %60, %70, %59 : i1, i32
      %72 = llvm.icmp "ne" %71, %7 : i32
      %73 = llvm.ashr %50, %22 : i32
      %74 = llvm.trunc %73 : i32 to i8
      %75 = llvm.or %47, %74 : i8
      %76 = llvm.select %48, %75, %64 : i1, i8
      %77 = llvm.add %76, %4 : i8
      %78 = llvm.and %76, %4 : i8
      %79 = llvm.icmp "ne" %78, %9 : i8
      %80 = llvm.select %60, %79, %2 : i1, i1
      %81 = llvm.select %72, %3, %80 : i1, i1
      %82 = llvm.select %81, %77, %76 : i1, i8
      %83 = llvm.or %35, %82 : i8
      %84 = llvm.select %28, %36, %83 : i1, i8
      %85 = llvm.select %29, %36, %84 : i1, i8
      %86 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %14) : (f32, i32) -> i1
      %87 = llvm.select %86, %17, %85 : i1, i8
      llvm.return %87 : i8
    }
}
)MLIR8";

static constexpr llvm::StringLiteral kF32ToFloat8_f8e5m2 = R"MLIR8(module {
    llvm.func internal @__kgen_pop_cast_f32_to_f8e5m2(%arg0: f32) -> i8 {
      %0 = llvm.mlir.constant(0 : i64) : i64
      %1 = llvm.mlir.constant(2147483648 : i64) : i64
      %2 = llvm.mlir.constant(false) : i1
      %3 = llvm.mlir.constant(true) : i1
      %4 = llvm.mlir.constant(1 : i8) : i8
      %5 = llvm.mlir.constant(3 : i8) : i8
      %6 = llvm.mlir.constant(1 : i32) : i32
      %7 = llvm.mlir.constant(0 : i32) : i32
      %8 = llvm.mlir.constant(123 : i8) : i8
      %9 = llvm.mlir.constant(0 : i8) : i8
      %10 = llvm.mlir.constant(-128 : i8) : i8
      %11 = llvm.mlir.constant(255 : i32) : i32
      %12 = llvm.mlir.constant(23 : i32) : i32
      %13 = llvm.mlir.constant(8388607 : i64) : i64
      %14 = llvm.mlir.constant(3 : i32) : i32
      %15 = llvm.mlir.constant(516 : i32) : i32
      %16 = llvm.mlir.constant(127 : i32) : i32
      %17 = llvm.mlir.constant(0x7F : f8E5M2) : i8
      %18 = llvm.mlir.constant(5.734400e+04 : f32) : f32
      %19 = llvm.mlir.constant(-14 : i32) : i32
      %20 = llvm.mlir.constant(15 : i32) : i32
      %21 = llvm.mlir.constant(31 : i32) : i32
      %22 = llvm.mlir.constant(2 : i32) : i32
      %23 = llvm.mlir.constant(21 : i32) : i32
      %24 = llvm.mlir.constant(32 : i32) : i32
      %25 = llvm.mlir.constant(8388608 : i32) : i32
      %26 = llvm.mlir.constant(20 : i32) : i32
      %27 = llvm.mlir.constant(1048575 : i32) : i32
      %28 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
      %29 = llvm.fcmp "oge" %28, %18 : f32
      %30 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %15) : (f32, i32) -> i1
      %31 = llvm.bitcast %arg0 : f32 to i32
      %32 = llvm.sext %31 : i32 to i64
      %33 = llvm.and %32, %1 : i64
      %34 = llvm.icmp "eq" %33, %0 : i64
      %35 = llvm.xor %34, %3 : i1
      %36 = llvm.select %35, %10, %9 : i1, i8
      %37 = llvm.or %36, %8 : i8
      %38 = llvm.lshr %31, %12 : i32
      %39 = llvm.and %38, %11 : i32
      %40 = llvm.sub %39, %16 : i32
      %41 = llvm.sub %19, %40 : i32
      %42 = llvm.shl %6, %41 : i32
      %43 = llvm.sub %42, %6 : i32
      %44 = llvm.icmp "slt" %41, %24 : i32
      %45 = llvm.add %40, %20 : i32
      %46 = llvm.and %45, %21 : i32
      %47 = llvm.shl %46, %22 : i32
      %48 = llvm.trunc %47 : i32 to i8
      %49 = llvm.icmp "sge" %40, %19 : i32
      %50 = llvm.and %32, %13 : i64
      %51 = llvm.trunc %50 : i64 to i32
      %52 = llvm.or %51, %25 : i32
      %53 = llvm.ashr %52, %41 : i32
      %54 = llvm.select %44, %53, %7 : i1, i32
      %55 = llvm.select %49, %51, %54 : i1, i32
      %56 = llvm.and %55, %27 : i32
      %57 = llvm.icmp "ne" %56, %7 : i32
      %58 = llvm.select %57, %6, %7 : i1, i32
      %59 = llvm.ashr %55, %26 : i32
      %60 = llvm.and %59, %6 : i32
      %61 = llvm.icmp "ne" %60, %7 : i32
      %62 = llvm.ashr %53, %23 : i32
      %63 = llvm.trunc %62 : i32 to i8
      %64 = llvm.and %63, %5 : i8
      %65 = llvm.select %44, %64, %9 : i1, i8
      %66 = llvm.and %52, %43 : i32
      %67 = llvm.icmp "ne" %66, %7 : i32
      %68 = llvm.select %67, %6, %7 : i1, i32
      %69 = llvm.select %44, %68, %7 : i1, i32
      %70 = llvm.select %49, %7, %69 : i1, i32
      %71 = llvm.or %70, %58 : i32
      %72 = llvm.select %61, %71, %60 : i1, i32
      %73 = llvm.icmp "ne" %72, %7 : i32
      %74 = llvm.ashr %51, %23 : i32
      %75 = llvm.trunc %74 : i32 to i8
      %76 = llvm.or %48, %75 : i8
      %77 = llvm.select %49, %76, %65 : i1, i8
      %78 = llvm.add %77, %4 : i8
      %79 = llvm.and %77, %4 : i8
      %80 = llvm.icmp "ne" %79, %9 : i8
      %81 = llvm.select %61, %80, %2 : i1, i1
      %82 = llvm.select %73, %3, %81 : i1, i1
      %83 = llvm.select %82, %78, %77 : i1, i8
      %84 = llvm.or %36, %83 : i8
      %85 = llvm.select %29, %37, %84 : i1, i8
      %86 = llvm.select %30, %37, %85 : i1, i8
      %87 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %14) : (f32, i32) -> i1
      %88 = llvm.select %87, %17, %86 : i1, i8
      llvm.return %88 : i8
    }
}
)MLIR8";

static constexpr llvm::StringLiteral kF32ToFloat8_f8e4m3fnuz = R"MLIR8(module {
    llvm.func internal @__kgen_pop_cast_f32_to_f8e4m3fnuz(%arg0: f32) -> i8 {
      %0 = llvm.mlir.constant(0 : i64) : i64
      %1 = llvm.mlir.constant(0.000000e+00 : f8E4M3FNUZ) : i8
      %2 = llvm.mlir.constant(2147483648 : i64) : i64
      %3 = llvm.mlir.constant(false) : i1
      %4 = llvm.mlir.constant(true) : i1
      %5 = llvm.mlir.constant(1 : i8) : i8
      %6 = llvm.mlir.constant(7 : i8) : i8
      %7 = llvm.mlir.constant(1 : i32) : i32
      %8 = llvm.mlir.constant(0 : i32) : i32
      %9 = llvm.mlir.constant(127 : i8) : i8
      %10 = llvm.mlir.constant(0 : i8) : i8
      %11 = llvm.mlir.constant(-128 : i8) : i8
      %12 = llvm.mlir.constant(255 : i32) : i32
      %13 = llvm.mlir.constant(23 : i32) : i32
      %14 = llvm.mlir.constant(8388607 : i64) : i64
      %15 = llvm.mlir.constant(3 : i32) : i32
      %16 = llvm.mlir.constant(516 : i32) : i32
      %17 = llvm.mlir.constant(127 : i32) : i32
      %18 = llvm.mlir.constant(0x80 : f8E4M3FNUZ) : i8
      %19 = llvm.mlir.constant(2.400000e+02 : f32) : f32
      %20 = llvm.mlir.constant(-7 : i32) : i32
      %21 = llvm.mlir.constant(8 : i32) : i32
      %22 = llvm.mlir.constant(15 : i32) : i32
      %23 = llvm.mlir.constant(20 : i32) : i32
      %24 = llvm.mlir.constant(32 : i32) : i32
      %25 = llvm.mlir.constant(8388608 : i32) : i32
      %26 = llvm.mlir.constant(19 : i32) : i32
      %27 = llvm.mlir.constant(524287 : i32) : i32
      %28 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
      %29 = llvm.fcmp "oge" %28, %19 : f32
      %30 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %16) : (f32, i32) -> i1
      %31 = llvm.bitcast %arg0 : f32 to i32
      %32 = llvm.sext %31 : i32 to i64
      %33 = llvm.and %32, %2 : i64
      %34 = llvm.icmp "eq" %33, %0 : i64
      %35 = llvm.xor %34, %4 : i1
      %36 = llvm.select %35, %11, %10 : i1, i8
      %37 = llvm.or %36, %9 : i8
      %38 = llvm.lshr %31, %13 : i32
      %39 = llvm.and %38, %12 : i32
      %40 = llvm.sub %39, %17 : i32
      %41 = llvm.sub %20, %40 : i32
      %42 = llvm.shl %7, %41 : i32
      %43 = llvm.sub %42, %7 : i32
      %44 = llvm.icmp "slt" %41, %24 : i32
      %45 = llvm.add %40, %21 : i32
      %46 = llvm.and %45, %22 : i32
      %47 = llvm.shl %46, %15 : i32
      %48 = llvm.trunc %47 : i32 to i8
      %49 = llvm.icmp "sge" %40, %20 : i32
      %50 = llvm.and %32, %14 : i64
      %51 = llvm.trunc %50 : i64 to i32
      %52 = llvm.or %51, %25 : i32
      %53 = llvm.ashr %52, %41 : i32
      %54 = llvm.select %44, %53, %8 : i1, i32
      %55 = llvm.select %49, %51, %54 : i1, i32
      %56 = llvm.and %55, %27 : i32
      %57 = llvm.icmp "ne" %56, %8 : i32
      %58 = llvm.select %57, %7, %8 : i1, i32
      %59 = llvm.ashr %55, %26 : i32
      %60 = llvm.and %59, %7 : i32
      %61 = llvm.icmp "ne" %60, %8 : i32
      %62 = llvm.ashr %53, %23 : i32
      %63 = llvm.trunc %62 : i32 to i8
      %64 = llvm.and %63, %6 : i8
      %65 = llvm.select %44, %64, %10 : i1, i8
      %66 = llvm.and %52, %43 : i32
      %67 = llvm.icmp "ne" %66, %8 : i32
      %68 = llvm.select %67, %7, %8 : i1, i32
      %69 = llvm.select %44, %68, %8 : i1, i32
      %70 = llvm.select %49, %8, %69 : i1, i32
      %71 = llvm.or %70, %58 : i32
      %72 = llvm.select %61, %71, %60 : i1, i32
      %73 = llvm.icmp "ne" %72, %8 : i32
      %74 = llvm.ashr %51, %23 : i32
      %75 = llvm.trunc %74 : i32 to i8
      %76 = llvm.or %48, %75 : i8
      %77 = llvm.select %49, %76, %65 : i1, i8
      %78 = llvm.add %77, %5 : i8
      %79 = llvm.and %77, %5 : i8
      %80 = llvm.icmp "ne" %79, %10 : i8
      %81 = llvm.select %61, %80, %3 : i1, i1
      %82 = llvm.select %73, %4, %81 : i1, i1
      %83 = llvm.select %82, %78, %77 : i1, i8
      %84 = llvm.icmp "eq" %83, %10 : i8
      %85 = llvm.or %36, %83 : i8
      %86 = llvm.select %84, %1, %85 : i1, i8
      %87 = llvm.select %29, %37, %86 : i1, i8
      %88 = llvm.select %30, %37, %87 : i1, i8
      %89 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %15) : (f32, i32) -> i1
      %90 = llvm.select %89, %18, %88 : i1, i8
      llvm.return %90 : i8
    }
}
)MLIR8";

static constexpr llvm::StringLiteral kF32ToFloat8_f8e5m2fnuz = R"MLIR8(module {
    llvm.func internal @__kgen_pop_cast_f32_to_f8e5m2fnuz(%arg0: f32) -> i8 {
      %0 = llvm.mlir.constant(0 : i64) : i64
      %1 = llvm.mlir.constant(0.000000e+00 : f8E5M2FNUZ) : i8
      %2 = llvm.mlir.constant(2147483648 : i64) : i64
      %3 = llvm.mlir.constant(false) : i1
      %4 = llvm.mlir.constant(true) : i1
      %5 = llvm.mlir.constant(1 : i8) : i8
      %6 = llvm.mlir.constant(3 : i8) : i8
      %7 = llvm.mlir.constant(1 : i32) : i32
      %8 = llvm.mlir.constant(0 : i32) : i32
      %9 = llvm.mlir.constant(127 : i8) : i8
      %10 = llvm.mlir.constant(0 : i8) : i8
      %11 = llvm.mlir.constant(-128 : i8) : i8
      %12 = llvm.mlir.constant(255 : i32) : i32
      %13 = llvm.mlir.constant(23 : i32) : i32
      %14 = llvm.mlir.constant(8388607 : i64) : i64
      %15 = llvm.mlir.constant(3 : i32) : i32
      %16 = llvm.mlir.constant(516 : i32) : i32
      %17 = llvm.mlir.constant(127 : i32) : i32
      %18 = llvm.mlir.constant(0x80 : f8E5M2FNUZ) : i8
      %19 = llvm.mlir.constant(5.734400e+04 : f32) : f32
      %20 = llvm.mlir.constant(-15 : i32) : i32
      %21 = llvm.mlir.constant(16 : i32) : i32
      %22 = llvm.mlir.constant(31 : i32) : i32
      %23 = llvm.mlir.constant(2 : i32) : i32
      %24 = llvm.mlir.constant(21 : i32) : i32
      %25 = llvm.mlir.constant(32 : i32) : i32
      %26 = llvm.mlir.constant(8388608 : i32) : i32
      %27 = llvm.mlir.constant(20 : i32) : i32
      %28 = llvm.mlir.constant(1048575 : i32) : i32
      %29 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
      %30 = llvm.fcmp "oge" %29, %19 : f32
      %31 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %16) : (f32, i32) -> i1
      %32 = llvm.bitcast %arg0 : f32 to i32
      %33 = llvm.sext %32 : i32 to i64
      %34 = llvm.and %33, %2 : i64
      %35 = llvm.icmp "eq" %34, %0 : i64
      %36 = llvm.xor %35, %4 : i1
      %37 = llvm.select %36, %11, %10 : i1, i8
      %38 = llvm.or %37, %9 : i8
      %39 = llvm.lshr %32, %13 : i32
      %40 = llvm.and %39, %12 : i32
      %41 = llvm.sub %40, %17 : i32
      %42 = llvm.sub %20, %41 : i32
      %43 = llvm.shl %7, %42 : i32
      %44 = llvm.sub %43, %7 : i32
      %45 = llvm.icmp "slt" %42, %25 : i32
      %46 = llvm.add %41, %21 : i32
      %47 = llvm.and %46, %22 : i32
      %48 = llvm.shl %47, %23 : i32
      %49 = llvm.trunc %48 : i32 to i8
      %50 = llvm.icmp "sge" %41, %20 : i32
      %51 = llvm.and %33, %14 : i64
      %52 = llvm.trunc %51 : i64 to i32
      %53 = llvm.or %52, %26 : i32
      %54 = llvm.ashr %53, %42 : i32
      %55 = llvm.select %45, %54, %8 : i1, i32
      %56 = llvm.select %50, %52, %55 : i1, i32
      %57 = llvm.and %56, %28 : i32
      %58 = llvm.icmp "ne" %57, %8 : i32
      %59 = llvm.select %58, %7, %8 : i1, i32
      %60 = llvm.ashr %56, %27 : i32
      %61 = llvm.and %60, %7 : i32
      %62 = llvm.icmp "ne" %61, %8 : i32
      %63 = llvm.ashr %54, %24 : i32
      %64 = llvm.trunc %63 : i32 to i8
      %65 = llvm.and %64, %6 : i8
      %66 = llvm.select %45, %65, %10 : i1, i8
      %67 = llvm.and %53, %44 : i32
      %68 = llvm.icmp "ne" %67, %8 : i32
      %69 = llvm.select %68, %7, %8 : i1, i32
      %70 = llvm.select %45, %69, %8 : i1, i32
      %71 = llvm.select %50, %8, %70 : i1, i32
      %72 = llvm.or %71, %59 : i32
      %73 = llvm.select %62, %72, %61 : i1, i32
      %74 = llvm.icmp "ne" %73, %8 : i32
      %75 = llvm.ashr %52, %24 : i32
      %76 = llvm.trunc %75 : i32 to i8
      %77 = llvm.or %49, %76 : i8
      %78 = llvm.select %50, %77, %66 : i1, i8
      %79 = llvm.add %78, %5 : i8
      %80 = llvm.and %78, %5 : i8
      %81 = llvm.icmp "ne" %80, %10 : i8
      %82 = llvm.select %62, %81, %3 : i1, i1
      %83 = llvm.select %74, %4, %82 : i1, i1
      %84 = llvm.select %83, %79, %78 : i1, i8
      %85 = llvm.icmp "eq" %84, %10 : i8
      %86 = llvm.or %37, %84 : i8
      %87 = llvm.select %85, %1, %86 : i1, i8
      %88 = llvm.select %30, %38, %87 : i1, i8
      %89 = llvm.select %31, %38, %88 : i1, i8
      %90 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %15) : (f32, i32) -> i1
      %91 = llvm.select %90, %18, %89 : i1, i8
      llvm.return %91 : i8
    }
}
)MLIR8";

static constexpr llvm::StringLiteral kF32ToFloat8_f8e8m0fnu = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f32_to_f8e8m0fnu(%arg0: f32 {llvm.noundef}) -> i8 {
    %0 = llvm.mlir.constant(0xFF : f8E8M0FNU) : i8
    %1 = llvm.mlir.constant(516 : i32) : i32
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(23 : i32) : i32
    %5 = llvm.mlir.constant(255 : i32) : i32
    %6 = llvm.mlir.constant(8388607 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(-2 : i8) : i8
    %9 = llvm.mlir.constant(0 : i8) : i8
    %10 = llvm.mlir.constant(4194304 : i32) : i32
    %11 = llvm.mlir.constant(1 : i8) : i8
    %12 = llvm.mlir.constant(3 : i32) : i32
    %13 = llvm.bitcast %arg0 : f32 to i32
    %14 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %1) : (f32, i32) -> i1
    %15 = llvm.lshr %13, %4 : i32
    %16 = llvm.and %15, %5 : i32
    %17 = llvm.trunc %16 : i32 to i8
    %18 = llvm.and %13, %6 : i32
    %19 = llvm.icmp "ugt" %18, %7 : i32
    %20 = llvm.icmp "eq" %17, %8 : i8
    %21 = llvm.xor %20, %3 : i1
    %22 = llvm.select %19, %21, %2 fastmath<contract> : i1, i1
    %23 = llvm.icmp "eq" %17, %9 : i8
    %24 = llvm.icmp "ule" %18, %10 : i32
    %25 = llvm.select %23, %24, %2 fastmath<contract> : i1, i1
    %26 = llvm.xor %25, %3 : i1
    %27 = llvm.select %22, %26, %2 fastmath<contract> : i1, i1
    %28 = llvm.add %17, %11 : i8
    %29 = llvm.select %27, %28, %17 fastmath<contract> : i1, i8
    %30 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %12) : (f32, i32) -> i1
    %31 = llvm.select %30, %3, %14 fastmath<contract> : i1, i1
    %32 = llvm.select %31, %0, %29 fastmath<contract> : i1, i8
    llvm.return %32 : i8
  }
}
)MLIR8";

static constexpr ConversionSnippet kFloatConversionSnippets[] = {
    {DType::f8e4m3fn, DType::f32, "__kgen_pop_cast_f8e4m3fn_to_f32",
     kFloat8ToF32_f8e4m3fn},
    {DType::f8e5m2, DType::f32, "__kgen_pop_cast_f8e5m2_to_f32",
     kFloat8ToF32_f8e5m2},
    {DType::f8e4m3fnuz, DType::f32, "__kgen_pop_cast_f8e4m3fnuz_to_f32",
     kFloat8ToF32_f8e4m3fnuz},
    {DType::f8e5m2fnuz, DType::f32, "__kgen_pop_cast_f8e5m2fnuz_to_f32",
     kFloat8ToF32_f8e5m2fnuz},
    {DType::f32, DType::f8e4m3fn, "__kgen_pop_cast_f32_to_f8e4m3fn",
     kF32ToFloat8_f8e4m3fn},
    {DType::f32, DType::f8e5m2, "__kgen_pop_cast_f32_to_f8e5m2",
     kF32ToFloat8_f8e5m2},
    {DType::f32, DType::f8e4m3fnuz, "__kgen_pop_cast_f32_to_f8e4m3fnuz",
     kF32ToFloat8_f8e4m3fnuz},
    {DType::f32, DType::f8e5m2fnuz, "__kgen_pop_cast_f32_to_f8e5m2fnuz",
     kF32ToFloat8_f8e5m2fnuz},
    {DType::f32, DType::f8e8m0fnu, "__kgen_pop_cast_f32_to_f8e8m0fnu",
     kF32ToFloat8_f8e8m0fnu},
};

static constexpr llvm::StringLiteral kBranchingFloat8ToF32_f8e4m3fn = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e4m3fn_to_f32_branching(%arg0: i8) -> f32 {
    %0 = llvm.mlir.constant(3 : i8) : i8
    %1 = llvm.mlir.constant(15 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(20 : i64) : i64
    %5 = llvm.mlir.constant(120 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(3 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(7 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(127 : i8) : i8
    %20 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %21 = llvm.mlir.constant(-2147483649 : i64) : i64
    %22 = llvm.mlir.constant(2147483648 : i64) : i64
    %23 = llvm.lshr %arg0, %0 : i8
    %24 = llvm.and %23, %1 : i8
    %25 = llvm.zext %24 : i8 to i64
    %26 = llvm.sext %arg0 : i8 to i64
    %27 = llvm.and %26, %13 : i64
    %28 = llvm.xor %27, %11 : i64
    %29 = llvm.icmp "slt" %27, %8 : i64
    %30 = llvm.select %29, %28, %27 : i1, i64
    %31 = llvm.icmp "eq" %27, %8 : i64
    %32 = llvm.xor %31, %10 : i1
    %33 = llvm.icmp "eq" %25, %8 : i64
    %34 = llvm.select %33, %32, %9 : i1, i1
    llvm.cond_br %34, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %35 = llvm.call_intrinsic "llvm.ctlz"(%30, %9) : (i64, i1) -> i64
    %36 = llvm.sub %12, %35 : i64
    %37 = llvm.sub %7, %36 : i64
    %38 = llvm.sub %25, %37 : i64
    %39 = llvm.add %37, %6 : i64
    %40 = llvm.shl %27, %39 : i64
    %41 = llvm.and %40, %13 : i64
    llvm.br ^bb3(%38, %41 : i64, i64)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%25, %27 : i64, i64)
  ^bb3(%42: i64, %43: i64):  // 2 preds: ^bb1, ^bb2
    %44 = llvm.add %42, %5 : i64
    %45 = llvm.shl %43, %4 : i64
    %46 = llvm.shl %44, %15 : i64
    %47 = llvm.and %46, %14 : i64
    %48 = llvm.or %47, %8 : i64
    %49 = llvm.trunc %48 : i64 to i32
    %50 = llvm.sext %49 : i32 to i64
    %51 = llvm.and %50, %16 : i64
    %52 = llvm.and %45, %17 : i64
    %53 = llvm.or %51, %52 : i64
    %54 = llvm.trunc %53 : i64 to i32
    %55 = llvm.bitcast %54 : i32 to f32
    %56 = llvm.and %arg0, %19 : i8
    %57 = llvm.icmp "eq" %56, %2 : i8
    %58 = llvm.select %57, %3, %55 : i1, f32
    %59 = llvm.icmp "eq" %56, %19 : i8
    %60 = llvm.select %59, %20, %58 : i1, f32
    %61 = llvm.and %26, %18 : i64
    %62 = llvm.icmp "eq" %61, %8 : i64
    %63 = llvm.xor %62, %10 : i1
    %64 = llvm.bitcast %60 : f32 to i32
    %65 = llvm.sext %64 : i32 to i64
    %66 = llvm.and %65, %21 : i64
    %67 = llvm.or %66, %22 : i64
    %68 = llvm.select %63, %67, %66 : i1, i64
    %69 = llvm.trunc %68 : i64 to i32
    %70 = llvm.bitcast %69 : i32 to f32
    llvm.return %70 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingFloat8ToF32_f8e5m2 = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e5m2_to_f32_branching(%arg0: i8) -> f32 {
    %0 = llvm.mlir.constant(2 : i8) : i8
    %1 = llvm.mlir.constant(31 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(21 : i64) : i64
    %5 = llvm.mlir.constant(112 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(3 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(124 : i8) : i8
    %20 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %21 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %22 = llvm.mlir.constant(-2147483649 : i64) : i64
    %23 = llvm.mlir.constant(2147483648 : i64) : i64
    %24 = llvm.mlir.constant(127 : i8) : i8
    %25 = llvm.lshr %arg0, %0 : i8
    %26 = llvm.and %25, %1 : i8
    %27 = llvm.zext %26 : i8 to i64
    %28 = llvm.sext %arg0 : i8 to i64
    %29 = llvm.and %28, %13 : i64
    %30 = llvm.xor %29, %11 : i64
    %31 = llvm.icmp "slt" %29, %8 : i64
    %32 = llvm.select %31, %30, %29 : i1, i64
    %33 = llvm.icmp "eq" %29, %8 : i64
    %34 = llvm.xor %33, %10 : i1
    %35 = llvm.icmp "eq" %27, %8 : i64
    %36 = llvm.select %35, %34, %9 : i1, i1
    llvm.cond_br %36, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %37 = llvm.call_intrinsic "llvm.ctlz"(%32, %9) : (i64, i1) -> i64
    %38 = llvm.sub %12, %37 : i64
    %39 = llvm.sub %7, %38 : i64
    %40 = llvm.sub %27, %39 : i64
    %41 = llvm.add %39, %6 : i64
    %42 = llvm.shl %29, %41 : i64
    %43 = llvm.and %42, %13 : i64
    llvm.br ^bb3(%40, %43 : i64, i64)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%27, %29 : i64, i64)
  ^bb3(%44: i64, %45: i64):  // 2 preds: ^bb1, ^bb2
    %46 = llvm.add %44, %5 : i64
    %47 = llvm.shl %45, %4 : i64
    %48 = llvm.shl %46, %15 : i64
    %49 = llvm.and %48, %14 : i64
    %50 = llvm.or %49, %8 : i64
    %51 = llvm.trunc %50 : i64 to i32
    %52 = llvm.sext %51 : i32 to i64
    %53 = llvm.and %52, %16 : i64
    %54 = llvm.and %47, %17 : i64
    %55 = llvm.or %53, %54 : i64
    %56 = llvm.trunc %55 : i64 to i32
    %57 = llvm.bitcast %56 : i32 to f32
    %58 = llvm.and %arg0, %24 : i8
    %59 = llvm.icmp "ugt" %58, %19 : i8
    %60 = llvm.icmp "eq" %58, %2 : i8
    %61 = llvm.select %60, %3, %57 : i1, f32
    %62 = llvm.icmp "eq" %58, %19 : i8
    llvm.cond_br %62, ^bb4, ^bb6
  ^bb4:  // pred: ^bb3
    llvm.br ^bb8(%20 : f32)
  ^bb5(%63: f32):  // pred: ^bb6
    llvm.br ^bb8(%63 : f32)
  ^bb6:  // pred: ^bb3
    llvm.cond_br %59, ^bb7(%61 : f32), ^bb5(%61 : f32)
  ^bb7(%64: f32):  // pred: ^bb6
    llvm.br ^bb8(%21 : f32)
  ^bb8(%65: f32):  // 3 preds: ^bb4, ^bb5, ^bb7
    %66 = llvm.and %28, %18 : i64
    %67 = llvm.icmp "eq" %66, %8 : i64
    %68 = llvm.xor %67, %10 : i1
    %69 = llvm.bitcast %65 : f32 to i32
    %70 = llvm.sext %69 : i32 to i64
    %71 = llvm.and %70, %22 : i64
    %72 = llvm.or %71, %23 : i64
    %73 = llvm.select %68, %72, %71 : i1, i64
    %74 = llvm.trunc %73 : i64 to i32
    %75 = llvm.bitcast %74 : i32 to f32
    llvm.return %75 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingFloat8ToF32_f8e4m3fnuz = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e4m3fnuz_to_f32_branching(%arg0: i8) -> f32 {
    %0 = llvm.mlir.constant(3 : i8) : i8
    %1 = llvm.mlir.constant(15 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(20 : i64) : i64
    %5 = llvm.mlir.constant(119 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(3 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(7 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(-2147483649 : i64) : i64
    %20 = llvm.mlir.constant(2147483648 : i64) : i64
    %21 = llvm.mlir.constant(-128 : i8) : i8
    %22 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %23 = llvm.mlir.constant(127 : i8) : i8
    %24 = llvm.lshr %arg0, %0 : i8
    %25 = llvm.and %24, %1 : i8
    %26 = llvm.zext %25 : i8 to i64
    %27 = llvm.sext %arg0 : i8 to i64
    %28 = llvm.and %27, %13 : i64
    %29 = llvm.xor %28, %11 : i64
    %30 = llvm.icmp "slt" %28, %8 : i64
    %31 = llvm.select %30, %29, %28 : i1, i64
    %32 = llvm.icmp "eq" %28, %8 : i64
    %33 = llvm.xor %32, %10 : i1
    %34 = llvm.icmp "eq" %26, %8 : i64
    %35 = llvm.select %34, %33, %9 : i1, i1
    llvm.cond_br %35, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %36 = llvm.call_intrinsic "llvm.ctlz"(%31, %9) : (i64, i1) -> i64
    %37 = llvm.sub %12, %36 : i64
    %38 = llvm.sub %7, %37 : i64
    %39 = llvm.sub %26, %38 : i64
    %40 = llvm.add %38, %6 : i64
    %41 = llvm.shl %28, %40 : i64
    %42 = llvm.and %41, %13 : i64
    llvm.br ^bb3(%39, %42 : i64, i64)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%26, %28 : i64, i64)
  ^bb3(%43: i64, %44: i64):  // 2 preds: ^bb1, ^bb2
    %45 = llvm.add %43, %5 : i64
    %46 = llvm.shl %44, %4 : i64
    %47 = llvm.shl %45, %15 : i64
    %48 = llvm.and %47, %14 : i64
    %49 = llvm.or %48, %8 : i64
    %50 = llvm.trunc %49 : i64 to i32
    %51 = llvm.sext %50 : i32 to i64
    %52 = llvm.and %51, %16 : i64
    %53 = llvm.and %46, %17 : i64
    %54 = llvm.or %52, %53 : i64
    %55 = llvm.trunc %54 : i64 to i32
    %56 = llvm.bitcast %55 : i32 to f32
    %57 = llvm.and %arg0, %23 : i8
    %58 = llvm.icmp "eq" %57, %2 : i8
    %59 = llvm.select %58, %3, %56 : i1, f32
    %60 = llvm.and %27, %18 : i64
    %61 = llvm.icmp "eq" %60, %8 : i64
    %62 = llvm.xor %61, %10 : i1
    %63 = llvm.bitcast %59 : f32 to i32
    %64 = llvm.sext %63 : i32 to i64
    %65 = llvm.and %64, %19 : i64
    %66 = llvm.or %65, %20 : i64
    %67 = llvm.select %62, %66, %65 : i1, i64
    %68 = llvm.trunc %67 : i64 to i32
    %69 = llvm.bitcast %68 : i32 to f32
    %70 = llvm.icmp "eq" %arg0, %21 : i8
    %71 = llvm.select %70, %22, %69 : i1, f32
    llvm.return %71 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingFloat8ToF32_f8e5m2fnuz = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f8e5m2fnuz_to_f32_branching(%arg0: i8) -> f32 {
    %0 = llvm.mlir.constant(2 : i8) : i8
    %1 = llvm.mlir.constant(31 : i8) : i8
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(21 : i64) : i64
    %5 = llvm.mlir.constant(111 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.constant(-1 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(3 : i64) : i64
    %14 = llvm.mlir.constant(-2155872256 : i64) : i64
    %15 = llvm.mlir.constant(23 : i64) : i64
    %16 = llvm.mlir.constant(-8388608 : i64) : i64
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(128 : i64) : i64
    %19 = llvm.mlir.constant(-2147483649 : i64) : i64
    %20 = llvm.mlir.constant(2147483648 : i64) : i64
    %21 = llvm.mlir.constant(-128 : i8) : i8
    %22 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %23 = llvm.mlir.constant(127 : i8) : i8
    %24 = llvm.lshr %arg0, %0 : i8
    %25 = llvm.and %24, %1 : i8
    %26 = llvm.zext %25 : i8 to i64
    %27 = llvm.sext %arg0 : i8 to i64
    %28 = llvm.and %27, %13 : i64
    %29 = llvm.xor %28, %11 : i64
    %30 = llvm.icmp "slt" %28, %8 : i64
    %31 = llvm.select %30, %29, %28 : i1, i64
    %32 = llvm.icmp "eq" %28, %8 : i64
    %33 = llvm.xor %32, %10 : i1
    %34 = llvm.icmp "eq" %26, %8 : i64
    %35 = llvm.select %34, %33, %9 : i1, i1
    llvm.cond_br %35, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %36 = llvm.call_intrinsic "llvm.ctlz"(%31, %9) : (i64, i1) -> i64
    %37 = llvm.sub %12, %36 : i64
    %38 = llvm.sub %7, %37 : i64
    %39 = llvm.sub %26, %38 : i64
    %40 = llvm.add %38, %6 : i64
    %41 = llvm.shl %28, %40 : i64
    %42 = llvm.and %41, %13 : i64
    llvm.br ^bb3(%39, %42 : i64, i64)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%26, %28 : i64, i64)
  ^bb3(%43: i64, %44: i64):  // 2 preds: ^bb1, ^bb2
    %45 = llvm.add %43, %5 : i64
    %46 = llvm.shl %44, %4 : i64
    %47 = llvm.shl %45, %15 : i64
    %48 = llvm.and %47, %14 : i64
    %49 = llvm.or %48, %8 : i64
    %50 = llvm.trunc %49 : i64 to i32
    %51 = llvm.sext %50 : i32 to i64
    %52 = llvm.and %51, %16 : i64
    %53 = llvm.and %46, %17 : i64
    %54 = llvm.or %52, %53 : i64
    %55 = llvm.trunc %54 : i64 to i32
    %56 = llvm.bitcast %55 : i32 to f32
    %57 = llvm.and %arg0, %23 : i8
    %58 = llvm.icmp "eq" %57, %2 : i8
    %59 = llvm.select %58, %3, %56 : i1, f32
    %60 = llvm.and %27, %18 : i64
    %61 = llvm.icmp "eq" %60, %8 : i64
    %62 = llvm.xor %61, %10 : i1
    %63 = llvm.bitcast %59 : f32 to i32
    %64 = llvm.sext %63 : i32 to i64
    %65 = llvm.and %64, %19 : i64
    %66 = llvm.or %65, %20 : i64
    %67 = llvm.select %62, %66, %65 : i1, i64
    %68 = llvm.trunc %67 : i64 to i32
    %69 = llvm.bitcast %68 : i32 to f32
    %70 = llvm.icmp "eq" %arg0, %21 : i8
    %71 = llvm.select %70, %22, %69 : i1, f32
    llvm.return %71 : f32
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingF32ToFloat8_f8e4m3fn = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f32_to_f8e4m3fn_branching(%arg0: f32) -> i8 {
    %0 = llvm.mlir.constant(524287 : i32) : i32
    %1 = llvm.mlir.constant(19 : i32) : i32
    %2 = llvm.mlir.constant(15 : i32) : i32
    %3 = llvm.mlir.constant(7 : i32) : i32
    %4 = llvm.mlir.constant(20 : i32) : i32
    %5 = llvm.mlir.constant(8388608 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(-6 : i32) : i32
    %8 = llvm.mlir.constant(4.480000e+02 : f32) : f32
    %9 = llvm.mlir.constant(0x7F : f8E4M3FN) : i8
    %10 = llvm.mlir.constant(127 : i32) : i32
    %11 = llvm.mlir.constant(516 : i32) : i32
    %12 = llvm.mlir.constant(3 : i32) : i32
    %13 = llvm.mlir.constant(8388607 : i64) : i64
    %14 = llvm.mlir.constant(23 : i32) : i32
    %15 = llvm.mlir.constant(255 : i32) : i32
    %16 = llvm.mlir.constant(-128 : i8) : i8
    %17 = llvm.mlir.constant(0 : i8) : i8
    %18 = llvm.mlir.constant(126 : i8) : i8
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(1 : i32) : i32
    %21 = llvm.mlir.constant(7 : i8) : i8
    %22 = llvm.mlir.undef : i8
    %23 = llvm.mlir.constant(1 : i8) : i8
    %24 = llvm.mlir.constant(true) : i1
    %25 = llvm.mlir.constant(false) : i1
    %26 = llvm.mlir.constant(2147483648 : i64) : i64
    %27 = llvm.mlir.constant(0 : i64) : i64
    %28 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
    %29 = llvm.fcmp "oge" %28, %8 : f32
    %30 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %11) : (f32, i32) -> i1
    %31 = llvm.bitcast %arg0 : f32 to i32
    %32 = llvm.sext %31 : i32 to i64
    %33 = llvm.and %32, %26 : i64
    %34 = llvm.icmp "eq" %33, %27 : i64
    %35 = llvm.xor %34, %24 : i1
    %36 = llvm.select %35, %16, %17 : i1, i8
    %37 = llvm.or %36, %18 : i8
    %38 = llvm.lshr %31, %14 : i32
    %39 = llvm.and %38, %15 : i32
    %40 = llvm.sub %39, %10 : i32
    %41 = llvm.icmp "sge" %40, %7 : i32
    %42 = llvm.and %32, %13 : i64
    %43 = llvm.trunc %42 : i64 to i32
    %44 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %12) : (f32, i32) -> i1
    llvm.cond_br %44, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb11(%9 : i8)
  ^bb2:  // pred: ^bb0
    llvm.cond_br %30, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb10(%37 : i8)
  ^bb4:  // pred: ^bb2
    llvm.cond_br %29, ^bb5, ^bb7
  ^bb5:  // pred: ^bb4
    llvm.return %37 : i8
  ^bb6(%45: i8, %46: i32, %47: i32, %48: i32):  // pred: ^bb7
    %49 = llvm.or %47, %5 : i32
    %50 = llvm.sub %7, %46 : i32
    %51 = llvm.ashr %49, %50 : i32
    %52 = llvm.ashr %51, %4 : i32
    %53 = llvm.trunc %52 : i32 to i8
    %54 = llvm.and %53, %21 : i8
    %55 = llvm.shl %20, %50 : i32
    %56 = llvm.sub %55, %20 : i32
    %57 = llvm.and %49, %56 : i32
    %58 = llvm.icmp "ne" %57, %19 : i32
    %59 = llvm.select %58, %20, %19 : i1, i32
    %60 = llvm.icmp "slt" %50, %6 : i32
    %61 = llvm.select %60, %54, %17 : i1, i8
    %62 = llvm.select %60, %51, %19 : i1, i32
    %63 = llvm.select %60, %59, %48 : i1, i32
    llvm.br ^bb9(%61, %62, %63 : i8, i32, i32)
  ^bb7:  // pred: ^bb4
    llvm.cond_br %41, ^bb8(%22, %40, %43, %19 : i8, i32, i32, i32), ^bb6(%22, %40, %43, %19 : i8, i32, i32, i32)
  ^bb8(%64: i8, %65: i32, %66: i32, %67: i32):  // pred: ^bb7
    %68 = llvm.add %65, %3 : i32
    %69 = llvm.and %68, %2 : i32
    %70 = llvm.shl %69, %12 : i32
    %71 = llvm.trunc %70 : i32 to i8
    %72 = llvm.ashr %66, %4 : i32
    %73 = llvm.trunc %72 : i32 to i8
    %74 = llvm.or %71, %73 : i8
    llvm.br ^bb9(%74, %66, %67 : i8, i32, i32)
  ^bb9(%75: i8, %76: i32, %77: i32):  // 2 preds: ^bb6, ^bb8
    %78 = llvm.and %75, %23 : i8
    %79 = llvm.icmp "ne" %78, %17 : i8
    %80 = llvm.ashr %76, %1 : i32
    %81 = llvm.and %80, %20 : i32
    %82 = llvm.and %76, %0 : i32
    %83 = llvm.icmp "ne" %82, %19 : i32
    %84 = llvm.select %83, %20, %19 : i1, i32
    %85 = llvm.or %77, %84 : i32
    %86 = llvm.icmp "ne" %81, %19 : i32
    %87 = llvm.select %86, %85, %81 : i1, i32
    %88 = llvm.icmp "ne" %87, %19 : i32
    %89 = llvm.select %86, %79, %25 : i1, i1
    %90 = llvm.select %88, %24, %89 : i1, i1
    %91 = llvm.add %75, %23 : i8
    %92 = llvm.select %90, %91, %75 : i1, i8
    %93 = llvm.or %36, %92 : i8
    llvm.br ^bb10(%93 : i8)
  ^bb10(%94: i8):  // 2 preds: ^bb3, ^bb9
    llvm.br ^bb11(%94 : i8)
  ^bb11(%95: i8):  // 2 preds: ^bb1, ^bb10
    llvm.return %95 : i8
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingF32ToFloat8_f8e5m2 = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f32_to_f8e5m2_branching(%arg0: f32) -> i8 {
    %0 = llvm.mlir.constant(1048575 : i32) : i32
    %1 = llvm.mlir.constant(20 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(31 : i32) : i32
    %4 = llvm.mlir.constant(15 : i32) : i32
    %5 = llvm.mlir.constant(21 : i32) : i32
    %6 = llvm.mlir.constant(8388608 : i32) : i32
    %7 = llvm.mlir.constant(32 : i32) : i32
    %8 = llvm.mlir.constant(-14 : i32) : i32
    %9 = llvm.mlir.constant(5.734400e+04 : f32) : f32
    %10 = llvm.mlir.constant(0x7F : f8E5M2) : i8
    %11 = llvm.mlir.constant(127 : i32) : i32
    %12 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %13 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %14 = llvm.mlir.constant(-2147483649 : i64) : i64
    %15 = llvm.mlir.constant(516 : i32) : i32
    %16 = llvm.mlir.constant(3 : i32) : i32
    %17 = llvm.mlir.constant(8388607 : i64) : i64
    %18 = llvm.mlir.constant(23 : i32) : i32
    %19 = llvm.mlir.constant(255 : i32) : i32
    %20 = llvm.mlir.constant(-128 : i8) : i8
    %21 = llvm.mlir.constant(0 : i8) : i8
    %22 = llvm.mlir.constant(123 : i8) : i8
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(1 : i32) : i32
    %25 = llvm.mlir.constant(3 : i8) : i8
    %26 = llvm.mlir.undef : i8
    %27 = llvm.mlir.constant(1 : i8) : i8
    %28 = llvm.mlir.constant(true) : i1
    %29 = llvm.mlir.constant(false) : i1
    %30 = llvm.mlir.constant(2147483648 : i64) : i64
    %31 = llvm.mlir.constant(0 : i64) : i64
    %32 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
    %33 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %15) : (f32, i32) -> i1
    %34 = llvm.bitcast %arg0 : f32 to i32
    %35 = llvm.sext %34 : i32 to i64
    %36 = llvm.and %35, %30 : i64
    %37 = llvm.icmp "eq" %36, %31 : i64
    %38 = llvm.xor %37, %28 : i1
    %39 = llvm.select %38, %20, %21 : i1, i8
    %40 = llvm.or %39, %22 : i8
    %41 = llvm.lshr %34, %18 : i32
    %42 = llvm.and %41, %19 : i32
    %43 = llvm.sub %42, %11 : i32
    %44 = llvm.icmp "sge" %43, %8 : i32
    %45 = llvm.and %35, %17 : i64
    %46 = llvm.trunc %45 : i64 to i32
    %47 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %16) : (f32, i32) -> i1
    llvm.cond_br %47, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb16(%10 : i8)
  ^bb2:  // pred: ^bb0
    llvm.cond_br %33, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb15(%40 : i8)
  ^bb4:  // pred: ^bb2
    llvm.cond_br %29, ^bb5, ^bb7
  ^bb5:  // pred: ^bb4
    llvm.br ^bb9(%12 : f32)
  ^bb6(%48: f32):  // pred: ^bb7
    llvm.br ^bb9(%48 : f32)
  ^bb7:  // pred: ^bb4
    llvm.cond_br %29, ^bb8(%9 : f32), ^bb6(%9 : f32)
  ^bb8(%49: f32):  // pred: ^bb7
    llvm.br ^bb9(%13 : f32)
  ^bb9(%50: f32):  // 3 preds: ^bb5, ^bb6, ^bb8
    %51 = llvm.bitcast %50 : f32 to i32
    %52 = llvm.sext %51 : i32 to i64
    %53 = llvm.and %52, %14 : i64
    %54 = llvm.trunc %53 : i64 to i32
    %55 = llvm.bitcast %54 : i32 to f32
    %56 = llvm.fcmp "oge" %32, %55 : f32
    llvm.cond_br %56, ^bb10, ^bb12
  ^bb10:  // pred: ^bb9
    llvm.return %40 : i8
  ^bb11(%57: i8, %58: i32, %59: i32, %60: i32):  // pred: ^bb12
    %61 = llvm.or %59, %6 : i32
    %62 = llvm.sub %8, %58 : i32
    %63 = llvm.ashr %61, %62 : i32
    %64 = llvm.ashr %63, %5 : i32
    %65 = llvm.trunc %64 : i32 to i8
    %66 = llvm.and %65, %25 : i8
    %67 = llvm.shl %24, %62 : i32
    %68 = llvm.sub %67, %24 : i32
    %69 = llvm.and %61, %68 : i32
    %70 = llvm.icmp "ne" %69, %23 : i32
    %71 = llvm.select %70, %24, %23 : i1, i32
    %72 = llvm.icmp "slt" %62, %7 : i32
    %73 = llvm.select %72, %66, %21 : i1, i8
    %74 = llvm.select %72, %63, %23 : i1, i32
    %75 = llvm.select %72, %71, %60 : i1, i32
    llvm.br ^bb14(%73, %74, %75 : i8, i32, i32)
  ^bb12:  // pred: ^bb9
    llvm.cond_br %44, ^bb13(%26, %43, %46, %23 : i8, i32, i32, i32), ^bb11(%26, %43, %46, %23 : i8, i32, i32, i32)
  ^bb13(%76: i8, %77: i32, %78: i32, %79: i32):  // pred: ^bb12
    %80 = llvm.add %77, %4 : i32
    %81 = llvm.and %80, %3 : i32
    %82 = llvm.shl %81, %2 : i32
    %83 = llvm.trunc %82 : i32 to i8
    %84 = llvm.ashr %78, %5 : i32
    %85 = llvm.trunc %84 : i32 to i8
    %86 = llvm.or %83, %85 : i8
    llvm.br ^bb14(%86, %78, %79 : i8, i32, i32)
  ^bb14(%87: i8, %88: i32, %89: i32):  // 2 preds: ^bb11, ^bb13
    %90 = llvm.and %87, %27 : i8
    %91 = llvm.icmp "ne" %90, %21 : i8
    %92 = llvm.ashr %88, %1 : i32
    %93 = llvm.and %92, %24 : i32
    %94 = llvm.and %88, %0 : i32
    %95 = llvm.icmp "ne" %94, %23 : i32
    %96 = llvm.select %95, %24, %23 : i1, i32
    %97 = llvm.or %89, %96 : i32
    %98 = llvm.icmp "ne" %93, %23 : i32
    %99 = llvm.select %98, %97, %93 : i1, i32
    %100 = llvm.icmp "ne" %99, %23 : i32
    %101 = llvm.select %98, %91, %29 : i1, i1
    %102 = llvm.select %100, %28, %101 : i1, i1
    %103 = llvm.add %87, %27 : i8
    %104 = llvm.select %102, %103, %87 : i1, i8
    %105 = llvm.or %39, %104 : i8
    llvm.br ^bb15(%105 : i8)
  ^bb15(%106: i8):  // 2 preds: ^bb3, ^bb14
    llvm.br ^bb16(%106 : i8)
  ^bb16(%107: i8):  // 2 preds: ^bb1, ^bb15
    llvm.return %107 : i8
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingF32ToFloat8_f8e4m3fnuz = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f32_to_f8e4m3fnuz_branching(%arg0: f32) -> i8 {
    %0 = llvm.mlir.constant(0.000000e+00 : f8E4M3FNUZ) : i8
    %1 = llvm.mlir.constant(524287 : i32) : i32
    %2 = llvm.mlir.constant(19 : i32) : i32
    %3 = llvm.mlir.constant(15 : i32) : i32
    %4 = llvm.mlir.constant(8 : i32) : i32
    %5 = llvm.mlir.constant(20 : i32) : i32
    %6 = llvm.mlir.constant(8388608 : i32) : i32
    %7 = llvm.mlir.constant(32 : i32) : i32
    %8 = llvm.mlir.constant(-7 : i32) : i32
    %9 = llvm.mlir.constant(2.400000e+02 : f32) : f32
    %10 = llvm.mlir.constant(0x80 : f8E4M3FNUZ) : i8
    %11 = llvm.mlir.constant(127 : i32) : i32
    %12 = llvm.mlir.constant(516 : i32) : i32
    %13 = llvm.mlir.constant(3 : i32) : i32
    %14 = llvm.mlir.constant(8388607 : i64) : i64
    %15 = llvm.mlir.constant(23 : i32) : i32
    %16 = llvm.mlir.constant(255 : i32) : i32
    %17 = llvm.mlir.constant(-128 : i8) : i8
    %18 = llvm.mlir.constant(0 : i8) : i8
    %19 = llvm.mlir.constant(127 : i8) : i8
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(7 : i8) : i8
    %23 = llvm.mlir.undef : i8
    %24 = llvm.mlir.constant(1 : i8) : i8
    %25 = llvm.mlir.constant(true) : i1
    %26 = llvm.mlir.constant(false) : i1
    %27 = llvm.mlir.constant(2147483648 : i64) : i64
    %28 = llvm.mlir.constant(0 : i64) : i64
    %29 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
    %30 = llvm.fcmp "oge" %29, %9 : f32
    %31 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %12) : (f32, i32) -> i1
    %32 = llvm.bitcast %arg0 : f32 to i32
    %33 = llvm.sext %32 : i32 to i64
    %34 = llvm.and %33, %27 : i64
    %35 = llvm.icmp "eq" %34, %28 : i64
    %36 = llvm.xor %35, %25 : i1
    %37 = llvm.select %36, %17, %18 : i1, i8
    %38 = llvm.or %37, %19 : i8
    %39 = llvm.lshr %32, %15 : i32
    %40 = llvm.and %39, %16 : i32
    %41 = llvm.sub %40, %11 : i32
    %42 = llvm.icmp "sge" %41, %8 : i32
    %43 = llvm.and %33, %14 : i64
    %44 = llvm.trunc %43 : i64 to i32
    %45 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %13) : (f32, i32) -> i1
    llvm.cond_br %45, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb11(%10 : i8)
  ^bb2:  // pred: ^bb0
    llvm.cond_br %31, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb10(%38 : i8)
  ^bb4:  // pred: ^bb2
    llvm.cond_br %30, ^bb5, ^bb7
  ^bb5:  // pred: ^bb4
    llvm.return %38 : i8
  ^bb6(%46: i8, %47: i32, %48: i32, %49: i32):  // pred: ^bb7
    %50 = llvm.or %48, %6 : i32
    %51 = llvm.sub %8, %47 : i32
    %52 = llvm.ashr %50, %51 : i32
    %53 = llvm.ashr %52, %5 : i32
    %54 = llvm.trunc %53 : i32 to i8
    %55 = llvm.and %54, %22 : i8
    %56 = llvm.shl %21, %51 : i32
    %57 = llvm.sub %56, %21 : i32
    %58 = llvm.and %50, %57 : i32
    %59 = llvm.icmp "ne" %58, %20 : i32
    %60 = llvm.select %59, %21, %20 : i1, i32
    %61 = llvm.icmp "slt" %51, %7 : i32
    %62 = llvm.select %61, %55, %18 : i1, i8
    %63 = llvm.select %61, %52, %20 : i1, i32
    %64 = llvm.select %61, %60, %49 : i1, i32
    llvm.br ^bb9(%62, %63, %64 : i8, i32, i32)
  ^bb7:  // pred: ^bb4
    llvm.cond_br %42, ^bb8(%23, %41, %44, %20 : i8, i32, i32, i32), ^bb6(%23, %41, %44, %20 : i8, i32, i32, i32)
  ^bb8(%65: i8, %66: i32, %67: i32, %68: i32):  // pred: ^bb7
    %69 = llvm.add %66, %4 : i32
    %70 = llvm.and %69, %3 : i32
    %71 = llvm.shl %70, %13 : i32
    %72 = llvm.trunc %71 : i32 to i8
    %73 = llvm.ashr %67, %5 : i32
    %74 = llvm.trunc %73 : i32 to i8
    %75 = llvm.or %72, %74 : i8
    llvm.br ^bb9(%75, %67, %68 : i8, i32, i32)
  ^bb9(%76: i8, %77: i32, %78: i32):  // 2 preds: ^bb6, ^bb8
    %79 = llvm.and %76, %24 : i8
    %80 = llvm.icmp "ne" %79, %18 : i8
    %81 = llvm.ashr %77, %2 : i32
    %82 = llvm.and %81, %21 : i32
    %83 = llvm.and %77, %1 : i32
    %84 = llvm.icmp "ne" %83, %20 : i32
    %85 = llvm.select %84, %21, %20 : i1, i32
    %86 = llvm.or %78, %85 : i32
    %87 = llvm.icmp "ne" %82, %20 : i32
    %88 = llvm.select %87, %86, %82 : i1, i32
    %89 = llvm.icmp "ne" %88, %20 : i32
    %90 = llvm.select %87, %80, %26 : i1, i1
    %91 = llvm.select %89, %25, %90 : i1, i1
    %92 = llvm.add %76, %24 : i8
    %93 = llvm.select %91, %92, %76 : i1, i8
    %94 = llvm.or %37, %93 : i8
    %95 = llvm.icmp "eq" %93, %18 : i8
    %96 = llvm.select %95, %0, %94 : i1, i8
    llvm.br ^bb10(%96 : i8)
  ^bb10(%97: i8):  // 2 preds: ^bb3, ^bb9
    llvm.br ^bb11(%97 : i8)
  ^bb11(%98: i8):  // 2 preds: ^bb1, ^bb10
    llvm.return %98 : i8
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingF32ToFloat8_f8e5m2fnuz = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f32_to_f8e5m2fnuz_branching(%arg0: f32) -> i8 {
    %0 = llvm.mlir.constant(0.000000e+00 : f8E5M2FNUZ) : i8
    %1 = llvm.mlir.constant(1048575 : i32) : i32
    %2 = llvm.mlir.constant(20 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(31 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(21 : i32) : i32
    %7 = llvm.mlir.constant(8388608 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mlir.constant(-15 : i32) : i32
    %10 = llvm.mlir.constant(5.734400e+04 : f32) : f32
    %11 = llvm.mlir.constant(0x80 : f8E5M2FNUZ) : i8
    %12 = llvm.mlir.constant(127 : i32) : i32
    %13 = llvm.mlir.constant(516 : i32) : i32
    %14 = llvm.mlir.constant(3 : i32) : i32
    %15 = llvm.mlir.constant(8388607 : i64) : i64
    %16 = llvm.mlir.constant(23 : i32) : i32
    %17 = llvm.mlir.constant(255 : i32) : i32
    %18 = llvm.mlir.constant(-128 : i8) : i8
    %19 = llvm.mlir.constant(0 : i8) : i8
    %20 = llvm.mlir.constant(127 : i8) : i8
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.mlir.constant(3 : i8) : i8
    %24 = llvm.mlir.undef : i8
    %25 = llvm.mlir.constant(1 : i8) : i8
    %26 = llvm.mlir.constant(true) : i1
    %27 = llvm.mlir.constant(false) : i1
    %28 = llvm.mlir.constant(2147483648 : i64) : i64
    %29 = llvm.mlir.constant(0 : i64) : i64
    %30 = llvm.call_intrinsic "llvm.fabs"(%arg0) : (f32) -> f32
    %31 = llvm.fcmp "oge" %30, %10 : f32
    %32 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %13) : (f32, i32) -> i1
    %33 = llvm.bitcast %arg0 : f32 to i32
    %34 = llvm.sext %33 : i32 to i64
    %35 = llvm.and %34, %28 : i64
    %36 = llvm.icmp "eq" %35, %29 : i64
    %37 = llvm.xor %36, %26 : i1
    %38 = llvm.select %37, %18, %19 : i1, i8
    %39 = llvm.or %38, %20 : i8
    %40 = llvm.lshr %33, %16 : i32
    %41 = llvm.and %40, %17 : i32
    %42 = llvm.sub %41, %12 : i32
    %43 = llvm.icmp "sge" %42, %9 : i32
    %44 = llvm.and %34, %15 : i64
    %45 = llvm.trunc %44 : i64 to i32
    %46 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %14) : (f32, i32) -> i1
    llvm.cond_br %46, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb11(%11 : i8)
  ^bb2:  // pred: ^bb0
    llvm.cond_br %32, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb10(%39 : i8)
  ^bb4:  // pred: ^bb2
    llvm.cond_br %31, ^bb5, ^bb7
  ^bb5:  // pred: ^bb4
    llvm.return %39 : i8
  ^bb6(%47: i8, %48: i32, %49: i32, %50: i32):  // pred: ^bb7
    %51 = llvm.or %49, %7 : i32
    %52 = llvm.sub %9, %48 : i32
    %53 = llvm.ashr %51, %52 : i32
    %54 = llvm.ashr %53, %6 : i32
    %55 = llvm.trunc %54 : i32 to i8
    %56 = llvm.and %55, %23 : i8
    %57 = llvm.shl %22, %52 : i32
    %58 = llvm.sub %57, %22 : i32
    %59 = llvm.and %51, %58 : i32
    %60 = llvm.icmp "ne" %59, %21 : i32
    %61 = llvm.select %60, %22, %21 : i1, i32
    %62 = llvm.icmp "slt" %52, %8 : i32
    %63 = llvm.select %62, %56, %19 : i1, i8
    %64 = llvm.select %62, %53, %21 : i1, i32
    %65 = llvm.select %62, %61, %50 : i1, i32
    llvm.br ^bb9(%63, %64, %65 : i8, i32, i32)
  ^bb7:  // pred: ^bb4
    llvm.cond_br %43, ^bb8(%24, %42, %45, %21 : i8, i32, i32, i32), ^bb6(%24, %42, %45, %21 : i8, i32, i32, i32)
  ^bb8(%66: i8, %67: i32, %68: i32, %69: i32):  // pred: ^bb7
    %70 = llvm.add %67, %5 : i32
    %71 = llvm.and %70, %4 : i32
    %72 = llvm.shl %71, %3 : i32
    %73 = llvm.trunc %72 : i32 to i8
    %74 = llvm.ashr %68, %6 : i32
    %75 = llvm.trunc %74 : i32 to i8
    %76 = llvm.or %73, %75 : i8
    llvm.br ^bb9(%76, %68, %69 : i8, i32, i32)
  ^bb9(%77: i8, %78: i32, %79: i32):  // 2 preds: ^bb6, ^bb8
    %80 = llvm.and %77, %25 : i8
    %81 = llvm.icmp "ne" %80, %19 : i8
    %82 = llvm.ashr %78, %2 : i32
    %83 = llvm.and %82, %22 : i32
    %84 = llvm.and %78, %1 : i32
    %85 = llvm.icmp "ne" %84, %21 : i32
    %86 = llvm.select %85, %22, %21 : i1, i32
    %87 = llvm.or %79, %86 : i32
    %88 = llvm.icmp "ne" %83, %21 : i32
    %89 = llvm.select %88, %87, %83 : i1, i32
    %90 = llvm.icmp "ne" %89, %21 : i32
    %91 = llvm.select %88, %81, %27 : i1, i1
    %92 = llvm.select %90, %26, %91 : i1, i1
    %93 = llvm.add %77, %25 : i8
    %94 = llvm.select %92, %93, %77 : i1, i8
    %95 = llvm.or %38, %94 : i8
    %96 = llvm.icmp "eq" %94, %19 : i8
    %97 = llvm.select %96, %0, %95 : i1, i8
    llvm.br ^bb10(%97 : i8)
  ^bb10(%98: i8):  // 2 preds: ^bb3, ^bb9
    llvm.br ^bb11(%98 : i8)
  ^bb11(%99: i8):  // 2 preds: ^bb1, ^bb10
    llvm.return %99 : i8
  }
}
)MLIR8";

static constexpr llvm::StringLiteral kBranchingF32ToFloat8_f8e8m0fnu = R"MLIR8(module {
  llvm.func internal @__kgen_pop_cast_f32_to_f8e8m0fnu_branching(%arg0: f32) -> i8 {
    %0 = llvm.mlir.constant(0xFF : f8E8M0FNU) : i8
    %1 = llvm.mlir.constant(516 : i32) : i32
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(23 : i32) : i32
    %5 = llvm.mlir.constant(255 : i32) : i32
    %6 = llvm.mlir.constant(8388607 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(-2 : i8) : i8
    %9 = llvm.mlir.constant(0 : i8) : i8
    %10 = llvm.mlir.constant(4194304 : i32) : i32
    %11 = llvm.mlir.constant(1 : i8) : i8
    %12 = llvm.mlir.constant(3 : i32) : i32
    %13 = llvm.bitcast %arg0 : f32 to i32
    %14 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %1) : (f32, i32) -> i1
    %15 = llvm.lshr %13, %4 : i32
    %16 = llvm.and %15, %5 : i32
    %17 = llvm.trunc %16 : i32 to i8
    %18 = llvm.and %13, %6 : i32
    %19 = llvm.icmp "ugt" %18, %7 : i32
    %20 = llvm.icmp "eq" %17, %8 : i8
    %21 = llvm.xor %20, %3 : i1
    %22 = llvm.icmp "eq" %17, %9 : i8
    %23 = llvm.icmp "ule" %18, %10 : i32
    %24 = llvm.select %22, %23, %2 : i1, i1
    %25 = llvm.xor %24, %3 : i1
    %26 = llvm.add %17, %11 : i8
    %27 = llvm.call_intrinsic "llvm.is.fpclass"(%arg0, %12) : (f32, i32) -> i1
    %28 = llvm.select %27, %3, %14 : i1, i1
    llvm.cond_br %28, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb3(%0 : i8)
  ^bb2:  // pred: ^bb0
    %29 = llvm.select %19, %21, %2 : i1, i1
    %30 = llvm.select %29, %25, %2 : i1, i1
    %31 = llvm.select %30, %26, %17 : i1, i8
    llvm.br ^bb3(%31 : i8)
  ^bb3(%32: i8):  // 2 preds: ^bb1, ^bb2
    llvm.return %32 : i8
  }
}
)MLIR8";

static constexpr ConversionSnippet kBranchingFloatConversionSnippets[] = {
    {DType::f8e4m3fn, DType::f32, "__kgen_pop_cast_f8e4m3fn_to_f32_branching",
     kBranchingFloat8ToF32_f8e4m3fn},
    {DType::f8e5m2, DType::f32, "__kgen_pop_cast_f8e5m2_to_f32_branching",
     kBranchingFloat8ToF32_f8e5m2},
    {DType::f8e4m3fnuz, DType::f32, "__kgen_pop_cast_f8e4m3fnuz_to_f32_branching",
     kBranchingFloat8ToF32_f8e4m3fnuz},
    {DType::f8e5m2fnuz, DType::f32, "__kgen_pop_cast_f8e5m2fnuz_to_f32_branching",
     kBranchingFloat8ToF32_f8e5m2fnuz},
    {DType::f32, DType::f8e4m3fn, "__kgen_pop_cast_f32_to_f8e4m3fn_branching",
     kBranchingF32ToFloat8_f8e4m3fn},
    {DType::f32, DType::f8e5m2, "__kgen_pop_cast_f32_to_f8e5m2_branching",
     kBranchingF32ToFloat8_f8e5m2},
    {DType::f32, DType::f8e4m3fnuz, "__kgen_pop_cast_f32_to_f8e4m3fnuz_branching",
     kBranchingF32ToFloat8_f8e4m3fnuz},
    {DType::f32, DType::f8e5m2fnuz, "__kgen_pop_cast_f32_to_f8e5m2fnuz_branching",
     kBranchingF32ToFloat8_f8e5m2fnuz},
    {DType::f32, DType::f8e8m0fnu, "__kgen_pop_cast_f32_to_f8e8m0fnu_branching",
     kBranchingF32ToFloat8_f8e8m0fnu},
};

} // namespace
