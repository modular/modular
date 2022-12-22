# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Bool import Bool
from Int import Int
from SIMD import SIMD


struct Buffer[size: __mlir_type.index, type: __mlir_type.`!kgen.dtype`]:
    var pointer: __mlir_type.`!pop.pointer<#lit.placeholder: !kgen.mlirtype>`[
        __mlir_type.`!pop.scalar<#lit.placeholder : !kgen.dtype>>`[type]
    ]
    var dynamic_size: Int

    fn __new__(
        ptr: __mlir_type.`!pop.pointer<#lit.placeholder: !kgen.mlirtype>`[
            __mlir_type.`!pop.scalar<#lit.placeholder : !kgen.dtype>>`[type]
        ],
    ) -> Buffer[size, type]:
        # Construct a Buffer type with statically known size
        __mlir_op.`kgen.param.assert`[
            _type:[],
            cond : __mlir_attr.`#kgen.param.expr<xor, true, #lit.placeholder : i1> : i1`[
                __mlir_attr.`#kgen.param.expr<eq, #kgen.unknown : index, #lit.placeholder : index> : i1`[
                    size
                ]
            ],
            message:"must be a known size",
        ]()
        return __mlir_op.`kgen.struct.create`[_type : Buffer[size, type]](
            ptr, Int(size)
        )

    fn __new__(
        ptr: __mlir_type.`!pop.pointer<#lit.placeholder: !kgen.mlirtype>`[
            __mlir_type.`!pop.scalar<#lit.placeholder : !kgen.dtype>>`[type]
        ],
        in_size: Int,
    ) -> Buffer[size, type]:
        # Construct a Buffer type with dynamic size
        __mlir_op.`kgen.param.assert`[
            _type:[],
            cond : __mlir_attr.`#kgen.param.expr<eq, #kgen.unknown : index, #lit.placeholder : index> : i1`[
                size
            ],
            message:"must be a dynamic size",
        ]()
        return __mlir_op.`kgen.struct.create`[_type : Buffer[size, type]](
            ptr, in_size
        )

    fn _is_dynamic(self) -> Bool:
        # Returns true if the size is not statically known.
        return __mlir_attr.`#kgen.param.expr<eq, #kgen.unknown : index, #lit.placeholder : index> : i1`[
            size
        ]

    fn __len__(self) -> Int:
        # Returns the dynamic size if the buffer is not statically known,
        # otherwise returns the statically known size parameter.
        if self._is_dynamic():
            return self.dynamic_size
        return size

    fn __getitem__(self, idx: Int) -> SIMD[1, type]:
        # Loads a single element (SIMD of size 1) from the buffer at the
        # specified index
        return self.simd_load[1](idx)

    fn simd_load[width: __mlir_type.index](self, idx: Int) -> SIMD[width, type]:
        # Loads a simd value from the buffer at the specified index
        var offset = __mlir_op.`pop.offset`(self.pointer, idx.__as_mlir_index())
        var ptr = __mlir_op.`pop.pointer.bitcast`[
            _type : __mlir_type.`!pop.pointer<#lit.placeholder: !kgen.mlirtype>`[
                __mlir_type.`!pop.simd<#lit.placeholder<0> : index, #lit.placeholder<1> : !kgen.dtype>>`[
                    width, type
                ]
            ]
        ](offset)
        var result = __mlir_op.`pop.load`[
            alignment : __mlir_attr.`1: index`,
            _type : __mlir_type.`!pop.simd<#lit.placeholder : index, #lit.placeholder : !kgen.dtype>>`[
                width, type
            ],
        ](ptr)
        return result

    fn __setitem__(self, idx: Int, val: SIMD[1, type]):
        # Stores a single value into the buffer at the specified index
        self.simd_store[1](idx, val)

    fn simd_store[
        width: __mlir_type.index
    ](self, idx: Int, val: SIMD[width, type]):
        # Stores a simd value into the buffer at the specified index
        var offset = __mlir_op.`pop.offset`(self.pointer, idx.__as_mlir_index())
        var ptr = __mlir_op.`pop.pointer.bitcast`[
            _type : __mlir_type.`!pop.pointer<#lit.placeholder: !kgen.mlirtype>`[
                __mlir_type.`!pop.simd<#lit.placeholder<0> : index, #lit.placeholder<1> : !kgen.dtype>>`[
                    width, type
                ]
            ]
        ](offset)
        __mlir_op.`pop.store`[
            alignment : __mlir_attr.`1: index`,
            _type:[],
        ](val.value, ptr)
