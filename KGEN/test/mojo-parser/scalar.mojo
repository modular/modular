# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s | FileCheck %s


# REFERENCE
#   kgen.generator.interface @erf_scalar<DT: dtype>(%in: !pop.scalar<DT>) -> !pop.scalar<DT>
#
#   lit.func @erf_scalar_taylor<DT: dtype>(%x: !pop.scalar<DT>) -> !pop.scalar<DT>
#     constraints <[in(:dtype DT, [float32, float64]), "incorrect element type"]> implements @erf_scalar {
#     // Compute erf(x) = (2.0*x)/Sqrt(Pi) - (2*x^3)/(3.0*Sqrt(Pi)) in Horner form as
#     // = x * (- 0.37612638903183752463 * x^2 + 1.1283791670955125739)
#     // = x * fma(x^2, -0.37612638903183752463, 1.1283791670955125739)
#     %c0 = pop.constant(1.1283791670955125739) : !pop.scalar<DT>
#     %c1 = pop.constant(-0.37612638903183752463) : !pop.scalar<DT>
#     %x2 = pop.mul %x, %x : !pop.scalar<DT>
#     %t0 = pop.fma %x2, %c1, %c0 : !pop.scalar<DT>
#     %t1 = pop.mul %t0, %x : !pop.scalar<DT>
#     lit.return %t1 : !pop.scalar<DT>
#   }


@register_passable
struct Scalar[type: DType]:
    fn __copyinit__(self) -> Self:
        return Self {}


fn fma[
    type: DType
](x: Scalar[type], y: Scalar[type], z: Scalar[type]) -> Scalar[type]:
    # use lower level library here, the impl depend on the type of DType, i.e, Float32, float64...
    return x


fn erf_scalar_taylor[type: DType](x: Scalar[type]) -> Scalar[type]:
    #  TODO: return x * fma[type](x*x, -0.37612638903183752463, 1.1283791670955125739)
    return x


# CHECK-LABEL: lit.func @"fma_float32
fn fma_float32(x: Float32, y: Float32, z: Float32) -> Float32:
    # CHECK: %0 = lit.call {{.*}}__mul__{{.*}}(%x, %y)
    # CHECK: %1 = lit.call {{.*}}__add__{{.*}}(%0, %z)
    # CHECK: lit.return %1
    return x * y + z


# CHECK-LABEL: lit.func @"erf_scalar_taylor_float32
fn erf_scalar_taylor_float32(x: Float32) -> Float32:
    # CHECK: %[[CST:.*]] = kgen.param.constant: {{.*}}FloatLiteral = <{{.*}}"-0.3761{{.*}}>
    # CHECK: lit.call {{.*}}__init__({{.*}}$float_literal::FloatLiteral){{.*}}(%[[CST]])
    return x * fma_float32(
        x * x, -0.37612638903183752463, 1.1283791670955125739
    )


##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"erf_taylor_vector
fn erf_taylor_vector[
    size: Int, type: __mlir_type.`!kgen.dtype`
](x: SIMD[type, size]) -> SIMD[type, size]:
    # CHECK: = kgen.param.constant: {{.*}}FloatLiteral = <{{.*}}"-0.37612638903183754"}>>
    return x * (x * x).fma(-0.37612638903183752463, 1.1283791670955125739)
