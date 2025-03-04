# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum

class DType(enum.Enum):
    bool = 1

    int8 = 135

    int16 = 137

    int32 = 139

    int64 = 141

    uint8 = 134

    uint16 = 136

    uint32 = 138

    uint64 = 140

    float16 = 70

    float32 = 72

    float64 = 73

    bfloat16 = 71

    float8_e4m3 = 65

    float8_e4m3fn = 66

    float8_e4m3fnuz = 67

    float8_e5m2 = 68

    float8_e5m2fnuz = 69

    unknown = 0

    @property
    def size_in_bytes(self) -> int: ...

bfloat16: DType = DType.bfloat16

bool: DType = DType.bool

float16: DType = DType.float16

float32: DType = DType.float32

float64: DType = DType.float64

float8_e4m3: DType = DType.float8_e4m3

float8_e4m3fn: DType = DType.float8_e4m3fn

float8_e4m3fnuz: DType = DType.float8_e4m3fnuz

float8_e5m2: DType = DType.float8_e5m2

float8_e5m2fnuz: DType = DType.float8_e5m2fnuz

int16: DType = DType.int16

int32: DType = DType.int32

int64: DType = DType.int64

int8: DType = DType.int8

uint16: DType = DType.uint16

uint32: DType = DType.uint32

uint64: DType = DType.uint64

uint8: DType = DType.uint8

unknown: DType = DType.unknown
