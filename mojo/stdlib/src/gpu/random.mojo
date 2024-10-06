# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Implements a basic RNG using the Philox algorithm.
"""

from memory import bitcast

from .intrinsics import mulwide


fn _mulhilow(a: UInt32, b: UInt32) -> SIMD[DType.uint32, 2]:
    var res = mulwide(a, b)
    return bitcast[DType.uint32, 2](res)


struct Random[rounds: Int = 6]:
    var _key: SIMD[DType.uint32, 2]
    var _counter: SIMD[DType.uint32, 4]

    fn __init__(
        inout self,
        *,
        seed: UInt64 = 0,
        subsequence: UInt64 = 0,
        offset: UInt64 = 0,
    ):
        self._key = bitcast[DType.uint32, 2](seed)
        self._counter = bitcast[DType.uint32, 4](
            SIMD[DType.uint64, 2](offset, subsequence)
        )

    @always_inline
    fn step(inout self) -> SIMD[DType.uint32, 4]:
        alias K_PHILOX_10 = SIMD[DType.uint32, 2](0x9E3779B9, 0xBB67AE85)

        @parameter
        for i in range(rounds):
            self._counter = self._single_round(self._counter, self._key)
            self._key += K_PHILOX_10
        return self._single_round(self._counter, self._key)

    @always_inline
    fn step_uniform(inout self) -> SIMD[DType.float32, 4]:
        # The inverse of 2^32
        alias INV_2_32 = 2.3283064e-10
        return self.step().cast[DType.float32]() * INV_2_32

    @always_inline
    fn _incrn(inout self, n: Int64):
        var hilo = bitcast[DType.uint32, 2](n)
        var hi = hilo[0]
        var lo = hilo[1]

        self._counter[0] += lo
        if self._counter[0] < lo:
            hi += 1
        self._counter[1] += hi
        if hi <= self._counter[1]:
            return
        self._counter[2] += 1
        if self._counter[2]:
            return
        self._counter[3] += 1

    @always_inline
    fn _single_round(
        self, counter: SIMD[DType.uint32, 4], key: SIMD[DType.uint32, 2]
    ) -> SIMD[DType.uint32, 4]:
        alias K_PHILOX_SA = 0xD2511F53
        alias K_PHILOX_SB = 0xCD9E8D57

        var res0 = _mulhilow(K_PHILOX_SA, counter[0])
        var res1 = _mulhilow(K_PHILOX_SB, counter[3])
        return SIMD[DType.uint32, 4](
            res1[1] ^ counter[1] ^ key[0],
            res1[0],
            res0[1] ^ counter[3] ^ key[1],
            res0[0],
        )
