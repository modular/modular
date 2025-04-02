# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort
from pathlib import Path
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle

from gpu.host._nvidia_cuda import CUstream
from memory import UnsafePointer
from collections.string import StaticString
from utils import StaticTuple

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CURAND_LIBRARY_PATH = "/usr/local/cuda/lib64/libcurand.so"

alias CUDA_CURAND_LIBRARY = _Global[
    "CUDA_CURAND_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    if not Path(CUDA_CURAND_LIBRARY_PATH).exists():
        return abort[_OwnedDLHandle](
            "the CUDA cuRand library was not found at "
            + CUDA_CURAND_LIBRARY_PATH
        )
    return _OwnedDLHandle(CUDA_CURAND_LIBRARY_PATH)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CURAND_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

alias curandDistributionShift_t = UnsafePointer[curandDistributionShift_st]


@register_passable("trivial")
struct curandDistributionShift_st:
    var probability: curandDistribution_t
    var host_probability: curandDistribution_t
    var shift: UInt32
    var length: UInt32
    var host_gen: UInt32


@register_passable("trivial")
struct curandDistributionM2Shift_st:
    var histogram: curandHistogramM2_t
    var host_histogram: curandHistogramM2_t
    var shift: UInt32
    var length: UInt32
    var host_gen: UInt32


@register_passable("trivial")
struct curandHistogramM2_st:
    var V: curandHistogramM2V_t
    var host_V: curandHistogramM2V_t
    var K: curandHistogramM2V_t
    var host_K: curandHistogramM2V_t
    var host_gen: UInt32


alias curandDistribution_t = UnsafePointer[Float64]


@register_passable("trivial")
struct curandDiscreteDistribution_st:
    var self_host_ptr: curandDiscreteDistribution_t
    var M2: curandDistributionM2Shift_t
    var host_M2: curandDistributionM2Shift_t
    var stddev: Float64
    var mean: Float64
    var method: curandMethod_t
    var host_gen: UInt32


fn curandGeneratePoissonMethod(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Int16],
    n: Int,
    func: Float64,
    method: curandMethod,
) -> curandStatus:
    return _get_dylib_function[
        "curandGeneratePoissonMethod",
        fn (
            curandGenerator_t, UnsafePointer[Int16], Int, Float64, curandMethod
        ) -> curandStatus,
    ]()(generator, output_ptr, n, func, method)


fn curandGenerateLongLong(
    generator: curandGenerator_t, output_ptr: UnsafePointer[Int64], num: Int
) -> curandStatus:
    """
    \\brief Generate 64-bit quasirandom numbers.

    Use generator to generate num 64-bit results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 64-bit values with every bit random.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param num - Number of random 64-bit values to generate

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
        a previous kernel launch \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_TYPE_ERROR if the generator is not a 64 bit quasirandom generator\\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateLongLong",
        fn (curandGenerator_t, UnsafePointer[Int64], Int) -> curandStatus,
    ]()(generator, output_ptr, num)


@value
struct libraryPropertyType_t:
    var _value: Int32
    alias MAJOR_VERSION = Self(0)
    alias MINOR_VERSION = Self(1)
    alias PATCH_LEVEL = Self(2)


fn curandGetProperty(
    type: libraryPropertyType_t, value: UnsafePointer[Int16]
) -> curandStatus:
    """
     \\brief Return the value of the curand property.

     Return in *value the number for the property described by type of the
     dynamically linked CURAND library.

     \\param type - CUDA library property
     \\param value - integer value for the requested property

     \\return
     - CURAND_STATUS_SUCCESS if the property value was successfully returned \\n
     - CURAND_STATUS_OUT_OF_RANGE if the property type is not recognized \\n
    ."""
    return _get_dylib_function[
        "curandGetProperty",
        fn (libraryPropertyType_t, UnsafePointer[Int16]) -> curandStatus,
    ]()(type, value)


@value
@register_passable("trivial")
struct curandRngType(Writable):
    """
    CURAND generator types
    ."""

    var _value: Int8
    alias CURAND_RNG_TEST = Self(0)
    alias CURAND_RNG_PSEUDO_DEFAULT = Self(1)
    alias CURAND_RNG_PSEUDO_XORWOW = Self(2)
    alias CURAND_RNG_PSEUDO_MRG32K3A = Self(3)
    alias CURAND_RNG_PSEUDO_MTGP32 = Self(4)
    alias CURAND_RNG_PSEUDO_MT19937 = Self(5)
    alias CURAND_RNG_PSEUDO_PHILOX4_32_10 = Self(6)
    alias CURAND_RNG_QUASI_DEFAULT = Self(7)
    alias CURAND_RNG_QUASI_SOBOL32 = Self(8)
    alias CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = Self(9)
    alias CURAND_RNG_QUASI_SOBOL64 = Self(10)
    alias CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = Self(11)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CURAND_RNG_TEST:
            return writer.write("CURAND_RNG_TEST")
        if self is Self.CURAND_RNG_PSEUDO_DEFAULT:
            return writer.write("CURAND_RNG_PSEUDO_DEFAULT")
        if self is Self.CURAND_RNG_PSEUDO_XORWOW:
            return writer.write("CURAND_RNG_PSEUDO_XORWOW")
        if self is Self.CURAND_RNG_PSEUDO_MRG32K3A:
            return writer.write("CURAND_RNG_PSEUDO_MRG32K3A")
        if self is Self.CURAND_RNG_PSEUDO_MTGP32:
            return writer.write("CURAND_RNG_PSEUDO_MTGP32")
        if self is Self.CURAND_RNG_PSEUDO_MT19937:
            return writer.write("CURAND_RNG_PSEUDO_MT19937")
        if self is Self.CURAND_RNG_PSEUDO_PHILOX4_32_10:
            return writer.write("CURAND_RNG_PSEUDO_PHILOX4_32_10")
        if self is Self.CURAND_RNG_QUASI_DEFAULT:
            return writer.write("CURAND_RNG_QUASI_DEFAULT")
        if self is Self.CURAND_RNG_QUASI_SOBOL32:
            return writer.write("CURAND_RNG_QUASI_SOBOL32")
        if self is Self.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32:
            return writer.write("CURAND_RNG_QUASI_SCRAMBLED_SOBOL32")
        if self is Self.CURAND_RNG_QUASI_SOBOL64:
            return writer.write("CURAND_RNG_QUASI_SOBOL64")
        if self is Self.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64:
            return writer.write("CURAND_RNG_QUASI_SCRAMBLED_SOBOL64")
        abort("invalid curandRngType entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("curandRngType(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias curandHistogramM2K_t = UnsafePointer[Int16]


fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus:
    """
    \\brief Destroy an existing generator.

    Destroy an existing generator and free all memory associated with its state.

    \\param generator - Generator to destroy

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_SUCCESS if generator was destroyed successfully \\n
    ."""
    return _get_dylib_function[
        "curandDestroyGenerator", fn (curandGenerator_t) -> curandStatus
    ]()(generator)


fn curandGetScrambleConstants64(
    constants: UnsafePointer[UnsafePointer[Int64]],
) -> curandStatus:
    """
    \\brief Get scramble constants for 64-bit scrambled Sobol' .

    Get a pointer to an array of scramble constants that can be used
    for quasirandom number generation.  The resulting pointer will
    reference an array of unsinged long longs in host memory.

    The array contains constants for many dimensions.  Each dimension
    has a single unsigned long long constant.

    \\param constants - Address of pointer in which to return scramble constants

    \\return
    - CURAND_STATUS_SUCCESS if the pointer was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandGetScrambleConstants64",
        fn (UnsafePointer[UnsafePointer[Int64]]) -> curandStatus,
    ]()(constants)


alias curandHistogramM2V_t = UnsafePointer[Float64]

alias curandHistogramM2V_st = curandDistribution_st

alias curandDiscreteDistribution_t = UnsafePointer[
    curandDiscreteDistribution_st
]


fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus:
    """
    \\brief Setup starting states.

    Generate the starting state of the generator.  This function is
    automatically called by generation functions such as
    ::curandGenerate() and ::curandGenerateUniform().
    It can be called manually for performance testing reasons to separate
    timings for starting state generation and random number generation.

    \\param generator - Generator to update

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
        a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_SUCCESS if the seeds were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateSeeds", fn (curandGenerator_t) -> curandStatus
    ]()(generator)


fn curandGenerateBinomial(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Int16],
    num: Int,
    n: Int16,
    p: Float64,
) -> curandStatus:
    return _get_dylib_function[
        "curandGenerateBinomial",
        fn (
            curandGenerator_t, UnsafePointer[Int16], Int, Int16, Float64
        ) -> curandStatus,
    ]()(generator, output_ptr, num, n, p)


fn curandGenerateLogNormalDouble(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Float64],
    n: Int,
    mean: Float64,
    stddev: Float64,
) -> curandStatus:
    """
    \\brief Generate log-normally distributed doubles.

    Use generator to generate n double results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 64-bit floating point values with log-normal distribution based on
    an associated normal distribution with mean mean and standard deviation stddev.

    Normally distributed results are generated from pseudorandom generators
    with a Box-Muller transform, and so require n to be even.
    Quasirandom generators use an inverse cumulative distribution
    function to preserve dimensionality.
    The normally distributed results are transformed into log-normal distribution.

    There may be slight numerical differences between results generated
    on the GPU with generators created with ::curandCreateGenerator()
    and results calculated on the CPU with generators created with
    ::curandCreateGeneratorHost().  These differences arise because of
    differences in results for transcendental functions.  In addition,
    future versions of CURAND may use newer versions of the CUDA math
    library, so different versions of CURAND may give slightly different
    numerical values.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param n - Number of doubles to generate
    \\param mean - Mean of normal distribution
    \\param stddev - Standard deviation of normal distribution

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension, or is not a multiple
       of two for pseudorandom generators \\n
    - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateLogNormalDouble",
        fn (
            curandGenerator_t, UnsafePointer[Float64], Int, Float64, Float64
        ) -> curandStatus,
    ]()(generator, output_ptr, n, mean, stddev)


fn curandGenerateNormal(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Float32],
    n: Int,
    mean: Float32,
    stddev: Float32,
) -> curandStatus:
    """
    \\brief Generate normally distributed doubles.

    Use generator to generate n float results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 32-bit floating point values with mean mean and standard
    deviation stddev.

    Normally distributed results are generated from pseudorandom generators
    with a Box-Muller transform, and so require n to be even.
    Quasirandom generators use an inverse cumulative distribution
    function to preserve dimensionality.

    There may be slight numerical differences between results generated
    on the GPU with generators created with ::curandCreateGenerator()
    and results calculated on the CPU with generators created with
    ::curandCreateGeneratorHost().  These differences arise because of
    differences in results for transcendental functions.  In addition,
    future versions of CURAND may use newer versions of the CUDA math
    library, so different versions of CURAND may give slightly different
    numerical values.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param n - Number of floats to generate
    \\param mean - Mean of normal distribution
    \\param stddev - Standard deviation of normal distribution

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension, or is not a multiple
       of two for pseudorandom generators \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateNormal",
        fn (
            curandGenerator_t, UnsafePointer[Float32], Int, Float32, Float32
        ) -> curandStatus,
    ]()(generator, output_ptr, n, mean, stddev)


fn curandGenerateLogNormal(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Float32],
    n: Int,
    mean: Float32,
    stddev: Float32,
) -> curandStatus:
    """
    \\brief Generate log-normally distributed floats.

    Use generator to generate n float results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 32-bit floating point values with log-normal distribution based on
    an associated normal distribution with mean mean and standard deviation stddev.

    Normally distributed results are generated from pseudorandom generators
    with a Box-Muller transform, and so require n to be even.
    Quasirandom generators use an inverse cumulative distribution
    function to preserve dimensionality.
    The normally distributed results are transformed into log-normal distribution.

    There may be slight numerical differences between results generated
    on the GPU with generators created with ::curandCreateGenerator()
    and results calculated on the CPU with generators created with
    ::curandCreateGeneratorHost().  These differences arise because of
    differences in results for transcendental functions.  In addition,
    future versions of CURAND may use newer versions of the CUDA math
    library, so different versions of CURAND may give slightly different
    numerical values.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param n - Number of floats to generate
    \\param mean - Mean of associated normal distribution
    \\param stddev - Standard deviation of associated normal distribution

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension, or is not a multiple
       of two for pseudorandom generators \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateLogNormal",
        fn (
            curandGenerator_t, UnsafePointer[Float32], Int, Float32, Float32
        ) -> curandStatus,
    ]()(generator, output_ptr, n, mean, stddev)


#
#  CURAND generator
#
#  \\cond UNHIDE_TYPEDEFS .
alias curandGenerator_st = NoneType
alias curandGenerator_t = UnsafePointer[curandGenerator_st]


@value
@register_passable("trivial")
struct curandMethod(Writable):
    """\\cond UNHIDE_ENUMS ."""

    var _value: Int8
    alias CURAND_CHOOSE_BEST = Self(0)
    alias CURAND_ITR = Self(1)
    alias CURAND_KNUTH = Self(2)
    alias CURAND_HITR = Self(3)
    alias CURAND_M1 = Self(4)
    alias CURAND_M2 = Self(5)
    alias CURAND_BINARY_SEARCH = Self(6)
    alias CURAND_DISCRETE_GAUSS = Self(7)
    alias CURAND_REJECTION = Self(8)
    alias CURAND_DEVICE_API = Self(9)
    alias CURAND_FAST_REJECTION = Self(10)
    alias CURAND_3RD = Self(11)
    alias CURAND_DEFINITION = Self(12)
    alias CURAND_POISSON = Self(13)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CURAND_CHOOSE_BEST:
            return writer.write("CURAND_CHOOSE_BEST")
        if self is Self.CURAND_ITR:
            return writer.write("CURAND_ITR")
        if self is Self.CURAND_KNUTH:
            return writer.write("CURAND_KNUTH")
        if self is Self.CURAND_HITR:
            return writer.write("CURAND_HITR")
        if self is Self.CURAND_M1:
            return writer.write("CURAND_M1")
        if self is Self.CURAND_M2:
            return writer.write("CURAND_M2")
        if self is Self.CURAND_BINARY_SEARCH:
            return writer.write("CURAND_BINARY_SEARCH")
        if self is Self.CURAND_DISCRETE_GAUSS:
            return writer.write("CURAND_DISCRETE_GAUSS")
        if self is Self.CURAND_REJECTION:
            return writer.write("CURAND_REJECTION")
        if self is Self.CURAND_DEVICE_API:
            return writer.write("CURAND_DEVICE_API")
        if self is Self.CURAND_FAST_REJECTION:
            return writer.write("CURAND_FAST_REJECTION")
        if self is Self.CURAND_3RD:
            return writer.write("CURAND_3RD")
        if self is Self.CURAND_DEFINITION:
            return writer.write("CURAND_DEFINITION")
        if self is Self.CURAND_POISSON:
            return writer.write("CURAND_POISSON")
        abort("invalid curandMethod entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("curandMethod(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn curandSetGeneratorOffset(
    generator: curandGenerator_t, offset: Int64
) -> curandStatus:
    """
    \\brief Set the absolute offset of the pseudo or quasirandom number generator.

    Set the absolute offset of the pseudo or quasirandom number generator.

    All values of offset are valid.  The offset position is absolute, not
    relative to the current position in the sequence.

    \\param generator - Generator to modify
    \\param offset - Absolute offset position

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_SUCCESS if generator offset was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandSetGeneratorOffset",
        fn (curandGenerator_t, Int64) -> curandStatus,
    ]()(generator, offset)


fn curandSetQuasiRandomGeneratorDimensions(
    generator: curandGenerator_t, num_dimensions: Int16
) -> curandStatus:
    """
    \\brief Set the number of dimensions.

    Set the number of dimensions to be generated by the quasirandom number
    generator.

    Legal values for num_dimensions are 1 to 20000.

    \\param generator - Generator to modify
    \\param num_dimensions - Number of dimensions

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_OUT_OF_RANGE if num_dimensions is not valid \\n
    - CURAND_STATUS_TYPE_ERROR if the generator is not a quasirandom number generator \\n
    - CURAND_STATUS_SUCCESS if generator ordering was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandSetQuasiRandomGeneratorDimensions",
        fn (curandGenerator_t, Int16) -> curandStatus,
    ]()(generator, num_dimensions)


# \\endcond
#
# CURAND distribution M2
#
# \\cond UNHIDE_TYPEDEFS .
alias curandDistributionM2Shift_t = UnsafePointer[curandDistributionM2Shift_st]


fn curandGetVersion(version: UnsafePointer[Int16]) -> curandStatus:
    """
    \\brief Return the version number of the library.

    Return in *version the version number of the dynamically linked CURAND
    library.  The format is the same as CUDART_VERSION from the CUDA Runtime.
    The only supported configuration is CURAND version equal to CUDA Runtime
    version.

    \\param version - CURAND library version

    \\return
    - CURAND_STATUS_SUCCESS if the version number was successfully returned \\n
    ."""
    return _get_dylib_function[
        "curandGetVersion", fn (UnsafePointer[Int16]) -> curandStatus
    ]()(version)


alias curandHistogramM2K_st = Int16

alias curandMethod_t = curandMethod


@value
@register_passable("trivial")
struct curandStatus(Writable):
    """
    CURAND function call status types
    ."""

    var _value: Int8
    alias CURAND_STATUS_SUCCESS = Self(0)
    alias CURAND_STATUS_VERSION_MISMATCH = Self(1)
    alias CURAND_STATUS_NOT_INITIALIZED = Self(2)
    alias CURAND_STATUS_ALLOCATION_FAILED = Self(3)
    alias CURAND_STATUS_TYPE_ERROR = Self(4)
    alias CURAND_STATUS_OUT_OF_RANGE = Self(5)
    alias CURAND_STATUS_LENGTH_NOT_MULTIPLE = Self(6)
    alias CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = Self(7)
    alias CURAND_STATUS_LAUNCH_FAILURE = Self(8)
    alias CURAND_STATUS_PREEXISTING_FAILURE = Self(9)
    alias CURAND_STATUS_INITIALIZATION_FAILED = Self(10)
    alias CURAND_STATUS_ARCH_MISMATCH = Self(11)
    alias CURAND_STATUS_INTERNAL_ERROR = Self(12)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CURAND_STATUS_SUCCESS:
            return writer.write("CURAND_STATUS_SUCCESS")
        if self is Self.CURAND_STATUS_VERSION_MISMATCH:
            return writer.write("CURAND_STATUS_VERSION_MISMATCH")
        if self is Self.CURAND_STATUS_NOT_INITIALIZED:
            return writer.write("CURAND_STATUS_NOT_INITIALIZED")
        if self is Self.CURAND_STATUS_ALLOCATION_FAILED:
            return writer.write("CURAND_STATUS_ALLOCATION_FAILED")
        if self is Self.CURAND_STATUS_TYPE_ERROR:
            return writer.write("CURAND_STATUS_TYPE_ERROR")
        if self is Self.CURAND_STATUS_OUT_OF_RANGE:
            return writer.write("CURAND_STATUS_OUT_OF_RANGE")
        if self is Self.CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return writer.write("CURAND_STATUS_LENGTH_NOT_MULTIPLE")
        if self is Self.CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return writer.write("CURAND_STATUS_DOUBLE_PRECISION_REQUIRED")
        if self is Self.CURAND_STATUS_LAUNCH_FAILURE:
            return writer.write("CURAND_STATUS_LAUNCH_FAILURE")
        if self is Self.CURAND_STATUS_PREEXISTING_FAILURE:
            return writer.write("CURAND_STATUS_PREEXISTING_FAILURE")
        if self is Self.CURAND_STATUS_INITIALIZATION_FAILED:
            return writer.write("CURAND_STATUS_INITIALIZATION_FAILED")
        if self is Self.CURAND_STATUS_ARCH_MISMATCH:
            return writer.write("CURAND_STATUS_ARCH_MISMATCH")
        if self is Self.CURAND_STATUS_INTERNAL_ERROR:
            return writer.write("CURAND_STATUS_INTERNAL_ERROR")
        abort("invalid curandStatus entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("curandStatus(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct curandDirectionVectorSet(Writable):
    """
    CURAND choice of direction vector set
    ."""

    var _value: Int8
    alias CURAND_DIRECTION_VECTORS_32_JOEKUO6 = Self(0)
    alias CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = Self(1)
    alias CURAND_DIRECTION_VECTORS_64_JOEKUO6 = Self(2)
    alias CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = Self(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CURAND_DIRECTION_VECTORS_32_JOEKUO6:
            return writer.write("CURAND_DIRECTION_VECTORS_32_JOEKUO6")
        if self is Self.CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            return writer.write("CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6")
        if self is Self.CURAND_DIRECTION_VECTORS_64_JOEKUO6:
            return writer.write("CURAND_DIRECTION_VECTORS_64_JOEKUO6")
        if self is Self.CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            return writer.write("CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6")
        abort("invalid curandDirectionVectorSet entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("curandDirectionVectorSet(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn curandGenerateUniform(
    generator: curandGenerator_t, output_ptr: UnsafePointer[Float32], num: Int
) -> curandStatus:
    """
    \\brief Generate uniformly distributed floats.

    Use generator to generate num float results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 32-bit floating point values between 0.0f and 1.0f,
    excluding 0.0f and including 1.0f.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param num - Number of floats to generate

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateUniform",
        fn (curandGenerator_t, UnsafePointer[Float32], Int) -> curandStatus,
    ]()(generator, output_ptr, num)


fn curandGenerateBinomialMethod(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Int16],
    num: Int,
    n: Int16,
    p: Float64,
    method: curandMethod,
) -> curandStatus:
    return _get_dylib_function[
        "curandGenerateBinomialMethod",
        fn (
            curandGenerator_t,
            UnsafePointer[Int16],
            Int,
            Int16,
            Float64,
            curandMethod,
        ) -> curandStatus,
    ]()(generator, output_ptr, num, n, p, method)


fn curandCreatePoissonDistribution(
    func: Float64,
    discrete_distribution: UnsafePointer[curandDiscreteDistribution_t],
) -> curandStatus:
    """
    \\brief Construct the histogram array for a Poisson distribution.

    Construct the histogram array for the Poisson distribution with func func.
    For func greater than 2000, an approximation with a normal distribution is used.

    \\param func - func for the Poisson distribution


    \\param discrete_distribution - pointer to the histogram in device memory

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \\n
    - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \\n
    - CURAND_STATUS_NOT_INITIALIZED if the distribution pointer was null \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_OUT_OF_RANGE if func is non-positive or greater than 400,000 \\n
    - CURAND_STATUS_SUCCESS if the histogram was generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandCreatePoissonDistribution",
        fn (
            Float64, UnsafePointer[curandDiscreteDistribution_t]
        ) -> curandStatus,
    ]()(func, discrete_distribution)


# \\cond UNHIDE_TYPEDEFS .
alias curandDirectionVectorSet_t = curandDirectionVectorSet


fn curandCreateGenerator(
    generator: UnsafePointer[curandGenerator_t], rng_type: curandRngType
) -> curandStatus:
    """
    \\brief Create new random number generator.

    Creates a new random number generator of type rng_type
    and returns it in *generator.

    Legal values for rng_type are:
    - CURAND_RNG_PSEUDO_DEFAULT
    - CURAND_RNG_PSEUDO_XORWOW
    - CURAND_RNG_PSEUDO_MRG32K3A
    - CURAND_RNG_PSEUDO_MTGP32
    - CURAND_RNG_PSEUDO_MT19937
    - CURAND_RNG_PSEUDO_PHILOX4_32_10
    - CURAND_RNG_QUASI_DEFAULT
    - CURAND_RNG_QUASI_SOBOL32
    - CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
    - CURAND_RNG_QUASI_SOBOL64
    - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

    When rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen
    is CURAND_RNG_PSEUDO_XORWOW.  \\n
    When rng_type is CURAND_RNG_QUASI_DEFAULT,
    the type chosen is CURAND_RNG_QUASI_SOBOL32.

    The default values for rng_type = CURAND_RNG_PSEUDO_XORWOW are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_PSEUDO_MRG32K3A are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_PSEUDO_MTGP32 are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_PSEUDO_MT19937 are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    * The default values for rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10 are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SOBOL32 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SOBOL64 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SCRAMBBLED_SOBOL32 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    \\param generator - Pointer to generator
    \\param rng_type - Type of generator to create

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated \\n
    - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \\n
    - CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the
      dynamically linked library version \\n
    - CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid \\n
    - CURAND_STATUS_SUCCESS if generator was created successfully \\n

    ."""
    return _get_dylib_function[
        "curandCreateGenerator",
        fn (UnsafePointer[curandGenerator_t], curandRngType) -> curandStatus,
    ]()(generator, rng_type)


fn curandSetGeneratorOrdering(
    generator: curandGenerator_t, order: curandOrdering
) -> curandStatus:
    """
    \\brief Set the ordering of results of the pseudo or quasirandom number generator.

    Set the ordering of results of the pseudo or quasirandom number generator.

    Legal values of order for pseudorandom generators are:
    - CURAND_ORDERING_PSEUDO_DEFAULT
    - CURAND_ORDERING_PSEUDO_BEST
    - CURAND_ORDERING_PSEUDO_SEEDED
    - CURAND_ORDERING_PSEUDO_LEGACY

    Legal values of order for quasirandom generators are:
    - CURAND_ORDERING_QUASI_DEFAULT

    \\param generator - Generator to modify
    \\param order - Ordering of results

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_OUT_OF_RANGE if the ordering is not valid \\n
    - CURAND_STATUS_SUCCESS if generator ordering was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandSetGeneratorOrdering",
        fn (curandGenerator_t, curandOrdering) -> curandStatus,
    ]()(generator, order)


# \\cond UNHIDE_TYPEDEFS .
alias curandStatus_t = curandStatus

#
#  CURAND distribution
#
#  \\cond UNHIDE_TYPEDEFS .
alias curandDistribution_st = Float64

# \\cond UNHIDE_TYPEDEFS .
alias curandRngType_t = curandRngType


fn curandGenerateUniformDouble(
    generator: curandGenerator_t, output_ptr: UnsafePointer[Float64], num: Int
) -> curandStatus:
    """
    \\brief Generate uniformly distributed doubles.

    Use generator to generate num double results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 64-bit double precision floating point values between
    0.0 and 1.0, excluding 0.0 and including 1.0.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param num - Number of doubles to generate

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension \\n
    - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateUniformDouble",
        fn (curandGenerator_t, UnsafePointer[Float64], Int) -> curandStatus,
    ]()(generator, output_ptr, num)


fn curandGenerateNormalDouble(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Float64],
    n: Int,
    mean: Float64,
    stddev: Float64,
) -> curandStatus:
    """
    \\brief Generate normally distributed doubles.

    Use generator to generate n double results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 64-bit floating point values with mean mean and standard
    deviation stddev.

    Normally distributed results are generated from pseudorandom generators
    with a Box-Muller transform, and so require n to be even.
    Quasirandom generators use an inverse cumulative distribution
    function to preserve dimensionality.

    There may be slight numerical differences between results generated
    on the GPU with generators created with ::curandCreateGenerator()
    and results calculated on the CPU with generators created with
    ::curandCreateGeneratorHost().  These differences arise because of
    differences in results for transcendental functions.  In addition,
    future versions of CURAND may use newer versions of the CUDA math
    library, so different versions of CURAND may give slightly different
    numerical values.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param n - Number of doubles to generate
    \\param mean - Mean of normal distribution
    \\param stddev - Standard deviation of normal distribution

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension, or is not a multiple
       of two for pseudorandom generators \\n
    - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerateNormalDouble",
        fn (
            curandGenerator_t, UnsafePointer[Float64], Int, Float64, Float64
        ) -> curandStatus,
    ]()(generator, output_ptr, n, mean, stddev)


fn curandGetDirectionVectors32(
    vectors: UnsafePointer[NoneType], set: curandDirectionVectorSet
) -> curandStatus:
    """
    \\brief Get direction vectors for 32-bit quasirandom number generation.

    Get a pointer to an array of direction vectors that can be used
    for quasirandom number generation.  The resulting pointer will
    reference an array of direction vectors in host memory.

    The array contains vectors for many dimensions.  Each dimension
    has 32 vectors.  Each individual vector is an unsigned int.

    Legal values for set are:
    - CURAND_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
    - CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)

    \\param vectors - Address of pointer in which to return direction vectors
    \\param set - Which set of direction vectors to use

    \\return
    - CURAND_STATUS_OUT_OF_RANGE if the choice of set is invalid \\n
    - CURAND_STATUS_SUCCESS if the pointer was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandGetDirectionVectors32",
        fn (UnsafePointer[NoneType], curandDirectionVectorSet) -> curandStatus,
    ]()(vectors, set)


fn curandDestroyDistribution(
    discrete_distribution: curandDiscreteDistribution_t,
) -> curandStatus:
    """
    \\brief Destroy the histogram array for a discrete distribution (e.g. Poisson).

    Destroy the histogram array for a discrete distribution created by curandCreatePoissonDistribution.

    \\param discrete_distribution - pointer to device memory where the histogram is stored

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the histogram was never created \\n
    - CURAND_STATUS_SUCCESS if the histogram was destroyed successfully \\n
    ."""
    return _get_dylib_function[
        "curandDestroyDistribution",
        fn (curandDiscreteDistribution_t) -> curandStatus,
    ]()(discrete_distribution)


fn curandGenerate(
    generator: curandGenerator_t, output_ptr: UnsafePointer[Int16], num: Int
) -> curandStatus:
    """
    \\brief Generate 32-bit pseudo or quasirandom numbers.

    Use generator to generate num 32-bit results into the device memory at
    outputPtr.  The device memory must have been previously allocated and be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 32-bit values with every bit random.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param num - Number of random 32-bit values to generate

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
        a previous kernel launch \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_TYPE_ERROR if the generator is a 64 bit quasirandom generator.
    (use ::curandGenerateLongLong() with 64 bit quasirandom generators)
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGenerate",
        fn (curandGenerator_t, UnsafePointer[Int16], Int) -> curandStatus,
    ]()(generator, output_ptr, num)


alias curandHistogramM2_t = UnsafePointer[curandHistogramM2_st]

#
#  CURAND array of 64-bit direction vectors
#
#  \\cond UNHIDE_TYPEDEFS .
alias curandDirectionVectors64_t = StaticTuple[UInt64, 64]


@value
@register_passable("trivial")
struct curandOrdering(Writable):
    """
    CURAND ordering of results in memory
    ."""

    var _value: Int8
    alias CURAND_ORDERING_PSEUDO_BEST = Self(0)
    alias CURAND_ORDERING_PSEUDO_DEFAULT = Self(1)
    alias CURAND_ORDERING_PSEUDO_SEEDED = Self(2)
    alias CURAND_ORDERING_PSEUDO_LEGACY = Self(3)
    alias CURAND_ORDERING_PSEUDO_DYNAMIC = Self(4)
    alias CURAND_ORDERING_QUASI_DEFAULT = Self(5)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CURAND_ORDERING_PSEUDO_BEST:
            return writer.write("CURAND_ORDERING_PSEUDO_BEST")
        if self is Self.CURAND_ORDERING_PSEUDO_DEFAULT:
            return writer.write("CURAND_ORDERING_PSEUDO_DEFAULT")
        if self is Self.CURAND_ORDERING_PSEUDO_SEEDED:
            return writer.write("CURAND_ORDERING_PSEUDO_SEEDED")
        if self is Self.CURAND_ORDERING_PSEUDO_LEGACY:
            return writer.write("CURAND_ORDERING_PSEUDO_LEGACY")
        if self is Self.CURAND_ORDERING_PSEUDO_DYNAMIC:
            return writer.write("CURAND_ORDERING_PSEUDO_DYNAMIC")
        if self is Self.CURAND_ORDERING_QUASI_DEFAULT:
            return writer.write("CURAND_ORDERING_QUASI_DEFAULT")
        abort("invalid curandOrdering entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("curandOrdering(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn curandSetPseudoRandomGeneratorSeed(
    generator: curandGenerator_t, seed: Int64
) -> curandStatus:
    """
    \\brief Set the seed value of the pseudo-random number generator.

    Set the seed value of the pseudorandom number generator.
    All values of seed are valid.  Different seeds will produce different sequences.
    Different seeds will often not be statistically correlated with each other,
    but some pairs of seed values may generate sequences which are statistically correlated.

    \\param generator - Generator to modify
    \\param seed - Seed value

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_TYPE_ERROR if the generator is not a pseudorandom number generator \\n
    - CURAND_STATUS_SUCCESS if generator seed was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandSetPseudoRandomGeneratorSeed",
        fn (curandGenerator_t, Int64) -> curandStatus,
    ]()(generator, seed)


fn curandSetStream(
    generator: curandGenerator_t, stream: CUstream
) -> curandStatus:
    """
    \\brief Set the current stream for CURAND kernel launches.

    Set the current stream for CURAND kernel launches.  All library functions
    will use this stream until set again.

    \\param generator - Generator to modify
    \\param stream - CUstream to use or ::NULL for null stream

    \\return
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_SUCCESS if stream was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandSetStream", fn (curandGenerator_t, CUstream) -> curandStatus
    ]()(generator, stream)


fn curandCreateGeneratorHost(
    generator: UnsafePointer[curandGenerator_t], rng_type: curandRngType
) -> curandStatus:
    """
    \\brief Create new host CPU random number generator.

    Creates a new host CPU random number generator of type rng_type
    and returns it in *generator.

    Legal values for rng_type are:
    - CURAND_RNG_PSEUDO_DEFAULT
    - CURAND_RNG_PSEUDO_XORWOW
    - CURAND_RNG_PSEUDO_MRG32K3A
    - CURAND_RNG_PSEUDO_MTGP32
    - CURAND_RNG_PSEUDO_MT19937
    - CURAND_RNG_PSEUDO_PHILOX4_32_10
    - CURAND_RNG_QUASI_DEFAULT
    - CURAND_RNG_QUASI_SOBOL32
    - CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
    - CURAND_RNG_QUASI_SOBOL64
    - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

    When rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen
    is CURAND_RNG_PSEUDO_XORWOW.  \\n
    When rng_type is CURAND_RNG_QUASI_DEFAULT,
    the type chosen is CURAND_RNG_QUASI_SOBOL32.

    The default values for rng_type = CURAND_RNG_PSEUDO_XORWOW are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_PSEUDO_MRG32K3A are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_PSEUDO_MTGP32 are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_PSEUDO_MT19937 are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    * The default values for rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10 are:
    - seed = 0
    - offset = 0
    - ordering = CURAND_ORDERING_PSEUDO_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SOBOL32 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SOBOL64 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    The default values for rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
    - dimensions = 1
    - offset = 0
    - ordering = CURAND_ORDERING_QUASI_DEFAULT

    \\param generator - Pointer to generator
    \\param rng_type - Type of generator to create

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \\n
    - CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the
      dynamically linked library version \\n
    - CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid \\n
    - CURAND_STATUS_SUCCESS if generator was created successfully \\n
    ."""
    return _get_dylib_function[
        "curandCreateGeneratorHost",
        fn (UnsafePointer[curandGenerator_t], curandRngType) -> curandStatus,
    ]()(generator, rng_type)


# \\cond UNHIDE_TYPEDEFS .
alias curandOrdering_t = curandOrdering


fn curandGeneratePoisson(
    generator: curandGenerator_t,
    output_ptr: UnsafePointer[Int16],
    n: Int,
    func: Float64,
) -> curandStatus:
    """
    \\brief Generate Poisson-distributed unsigned ints.

    Use generator to generate n unsigned int results into device memory at
    outputPtr.  The device memory must have been previously allocated and must be
    large enough to hold all the results.  Launches are done with the stream
    set using ::curandSetStream(), or the null stream if no stream has been set.

    Results are 32-bit unsigned int point values with Poisson distribution, with func func.

    \\param generator - Generator to use
    \\param outputPtr - Pointer to device memory to store CUDA-generated results, or
                    Pointer to host memory to store CPU-generated results
    \\param n - Number of unsigned ints to generate
    \\param func - func for the Poisson distribution

    \\return
    - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \\n
    - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \\n
    - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
       a previous kernel launch \\n
    - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \\n
    - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
       not a multiple of the quasirandom dimension\\n
    - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU or sm does not support double precision \\n
    - CURAND_STATUS_OUT_OF_RANGE if func is non-positive or greater than 400,000 \\n
    - CURAND_STATUS_SUCCESS if the results were generated successfully \\n
    ."""
    return _get_dylib_function[
        "curandGeneratePoisson",
        fn (
            curandGenerator_t, UnsafePointer[Int16], Int, Float64
        ) -> curandStatus,
    ]()(generator, output_ptr, n, func)


#
#  CURAND array of 32-bit direction vectors
#
#  \\cond UNHIDE_TYPEDEFS .
alias curandDirectionVectors32_t = StaticTuple[UInt32, 32]


fn curandGetScrambleConstants32(
    constants: UnsafePointer[UnsafePointer[Int16]],
) -> curandStatus:
    """
    \\brief Get scramble constants for 32-bit scrambled Sobol' .

    Get a pointer to an array of scramble constants that can be used
    for quasirandom number generation.  The resulting pointer will
    reference an array of unsinged ints in host memory.

    The array contains constants for many dimensions.  Each dimension
    has a single unsigned int constant.

    \\param constants - Address of pointer in which to return scramble constants

    \\return
    - CURAND_STATUS_SUCCESS if the pointer was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandGetScrambleConstants32",
        fn (UnsafePointer[UnsafePointer[Int16]]) -> curandStatus,
    ]()(constants)


fn curandGetDirectionVectors64(
    vectors: UnsafePointer[NoneType], set: curandDirectionVectorSet
) -> curandStatus:
    """
    \\brief Get direction vectors for 64-bit quasirandom number generation.

    Get a pointer to an array of direction vectors that can be used
    for quasirandom number generation.  The resulting pointer will
    reference an array of direction vectors in host memory.

    The array contains vectors for many dimensions.  Each dimension
    has 64 vectors.  Each individual vector is an unsigned long long.

    Legal values for set are:
    - CURAND_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
    - CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)

    \\param vectors - Address of pointer in which to return direction vectors
    \\param set - Which set of direction vectors to use

    \\return
    - CURAND_STATUS_OUT_OF_RANGE if the choice of set is invalid \\n
    - CURAND_STATUS_SUCCESS if the pointer was set successfully \\n
    ."""
    return _get_dylib_function[
        "curandGetDirectionVectors64",
        fn (UnsafePointer[NoneType], curandDirectionVectorSet) -> curandStatus,
    ]()(vectors, set)
