# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import env_get_int

import compiler
from logger import Logger
from max.tensor import ManagedTensorSlice, foreach, OutputTensor, InputTensor
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList

alias logger = Logger()


@compiler.register("use_splitk_reduction_scheme")
struct UseSplitkReductionScheme:
    @staticmethod
    fn execute(
        out: OutputTensor[type = DType.int32, rank=1],
    ):
        alias split_k_reduction_scheme = env_get_int[
            "SPLITK_REDUCTION_SCHEME", 2
        ]()
        out[0] = split_k_reduction_scheme


@compiler.register("use_logger")
struct UseLogger:
    @staticmethod
    fn execute(
        out: OutputTensor[type = DType.int32, rank=1],
    ):
        logger.error("I'm a custom Mojo function!")
        out[0] = Int(logger.level._value)


@compiler.register("add_one_custom")
struct AddOneCustom:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        fn add_one[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return x.load[width](idx) + 1

        foreach[add_one, target=target](out, ctx)

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"
