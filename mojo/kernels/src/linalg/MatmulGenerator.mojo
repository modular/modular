# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import div_ceil, max, min
from sys.info import simdbytewidth, simdwidthof

from algorithm import tile, tile_and_unswitch, vectorize
from buffer import NDBuffer
from buffer.list import Dim, DimList
from Matmul import (
    GemmShape,
    MatmulConfig,
    MatmulDataType,
    MatmulInnerLoopBPacked,
    MatmulOperandLayout,
    PackMatrixCols,
    PackMatrixRows,
    calculate_tile_n_k,
)
from MatmulUtils import get_pack_data_size
from memory import stack_allocation
from memory.unsafe import DTypePointer

from utils.index import Index, StaticIntTuple


@register_passable("trivial")
struct GemmIdentifiers:
    """A helper struct collecting the unified indices for gemm dimensions and
    gemm operand/results. This is used to index a list of dimensions or a list
    of buffers.

    TODO: This struct is a temporary solution for initial tile generator, should
    further clean up into separate enums.
    """

    # Represents the M dimension on the gemm space.
    alias DimM = 0

    # Represents the N dimension on the gemm space.
    alias DimN = 1

    # Represents the K dimension on the gemm space.
    alias DimK = 2

    # Represents the output matrix, i.e C matrix.
    alias ResultC = 0

    # Represents the first input matrix, i.e A matrix.
    alias OperandA = 1

    # Represents the second input matrix, i.e B matrix.
    alias OperandB = 2


@register_passable("trivial")
struct StaticGemmShape:
    """A helper implementation of gemm shape for static state tracking. Each of
    entries has a unknown state besides integer values.
    """

    var M: Dim
    var N: Dim
    var K: Dim

    fn __getitem__(self, idx: Int) -> Dim:
        if idx == GemmIdentifiers.DimM:
            return self.M
        if idx == GemmIdentifiers.DimN:
            return self.N
        if idx == GemmIdentifiers.DimK:
            return self.K
        return 0

    fn __setitem__(inout self, idx: Int, value: Dim):
        if idx == GemmIdentifiers.DimM:
            self.M = value
            return
        if idx == GemmIdentifiers.DimN:
            self.N = value
            return
        if idx == GemmIdentifiers.DimK:
            self.K = value
            return

    @staticmethod
    fn all_unknown() -> StaticGemmShape:
        """A utility builder that builds a GemmShape representation with all the
        dimensions unknown.

        Returns(GemmShape):
            The constructed gemm shape struct.
        """

        return StaticGemmShape {
            M: Dim(),
            N: Dim(),
            K: Dim(),
        }

    fn _as_gemm_shape(self) -> GemmShape:
        """Test and debug utility. Returns the currently held
        value as a GemmShape, converting unknowns to -1.
        """

        @always_inline
        @parameter
        fn _convert_unknown(value: Dim) -> Int:
            if value:
                return value.get()
            return -1

        return GemmShape(
            _convert_unknown(self.M),
            _convert_unknown(self.N),
            _convert_unknown(self.K),
        )


@register_passable("trivial")
struct GemmSwitch:
    """A helper struct recording if each of the dimensions in the gemm space,
    i.e. M,N,K has been un-switched. See also unswitch in the functional
    patterns.

    TODO: Would benefit from having a `StaticBoolTuple`.
    """

    # Indicates if the M dimensions is unswitched.
    var M: Bool

    # Indicates if the N dimensions is unswitched.
    var N: Bool

    # Indicates if the K dimensions is unswitched.
    var K: Bool

    @staticmethod
    @always_inline
    fn all_false() -> Self:
        """Factory utility returning a all-false instance of this struct.

        Returns (GemmSwitch):
            The all-false instance constructed.
        """
        return Self {M: False, N: False, K: False}

    fn __setitem__(inout self, idx: Int, value: Bool):
        """Setter utility based on indices from `GemmIdentifiers.

        Args:
            idx: The identifier for the dimension to set.
            value: The value to set to on the given dimension.

        """
        if idx == GemmIdentifiers.DimM:
            self.M = value
        elif idx == GemmIdentifiers.DimN:
            self.N = value
        elif idx == GemmIdentifiers.DimK:
            self.K = value


@register_passable("trivial")
struct MatmulDynamicState[data_type: MatmulDataType]:
    """Dynamic state space for the matmul tiling generator. This state space
    keeps track of all the runtime-known values including:
        - Runtime pointers to the input, output buffers and stack allocations.
        - Input and output dynamic shapes.
        - Dynamic tile bounds for dynamic tiling.
    """

    # TODO: use some buffer type to represent these.

    # Pointer to the buffer holding the first input matrix, i.e. matrix A.
    var a: DTypePointer[data_type.value_type]

    # Pointer to the buffer holding the second input matrix, i.e. matrix B.
    var b: DTypePointer[data_type.value_type]

    # Pointer to the buffer holding the first output matrix, i.e. matrix C.
    var c: DTypePointer[data_type.accum_type]

    # Buffer holding the packed data for b.
    var packed_b: DTypePointer[data_type.value_type]

    # Placeholder, To be enabled in other tiling variants.
    var packed_a: DTypePointer[data_type.value_type]
    var packed_c: DTypePointer[data_type.value_type]

    # TODO: generalize and pipe through stride logic.
    # Global problem size for the matmul space.
    var global_gemm_size: GemmShape

    # Global offset on the gemm problem of the current call.
    var global_offset: GemmShape

    # Amount of valid data on each dimension.
    var valid_tile_bound: GemmShape

    fn __init__() -> Self:
        return Self {
            a: DTypePointer[data_type.value_type](),
            b: DTypePointer[data_type.value_type](),
            c: DTypePointer[data_type.accum_type](),
            packed_b: DTypePointer[data_type.value_type](),
            packed_a: DTypePointer[data_type.value_type](),
            packed_c: DTypePointer[data_type.value_type](),
            global_gemm_size: GemmShape(0, 0, 0),
            global_offset: GemmShape(0, 0, 0),
            valid_tile_bound: GemmShape(0, 0, 0),
        }

    fn apply_offset(
        self,
        dimension_idx: Int,
        offset: Int,
    ) -> Self:
        """Applies an offset to the specified dimension, updating both base
        offset and dynamic bound.

        Args:
            dimension_idx: Dimension identifiers for the dimension to be
                updated, expected values are GemmIdentifiers.{DimM, DimN, DimK}.
            offset: The offset value to apply.

        Returns (Self):
            The updated dynamic state.
        """

        var updated = self
        # New global offset simply the sum of local and global offset.
        updated.global_offset[dimension_idx] = (
            self.global_offset[dimension_idx] + offset
        )
        # Since the starting offset advanced, the available data bound ahead
        #  should be reduced.
        updated.valid_tile_bound[dimension_idx] = (
            self.valid_tile_bound[dimension_idx] - offset
        )
        return updated

    fn apply_new_bound(
        self,
        dimension_idx: Int,
        bound: Int,
    ) -> Self:
        """Updates the dynamic bound to the new value. This utility is used in
        dynamic tiling, where the matmul space is dynamically partitioned into
        smaller matmuls.

        Args:
            dimension_idx: Dimension identifiers for the dimension to be
                updated, expected values are GemmIdentifiers.{DimM, DimN, DimK}.
            bound: The bound value to use.

        Returns (Self):
            The updated dynamic state.
        """
        var updated = self
        # New tile bound is minimum of the tile size and the original bound.
        updated.valid_tile_bound[dimension_idx] = min(
            self.valid_tile_bound[dimension_idx], bound
        )
        return updated

    @staticmethod
    @always_inline
    fn get[
        matmul_config: MatmulConfig, data_layout: MatmulOperandLayout
    ](
        c: NDBuffer[data_type.accum_type, 2, matmul_config.c_shape],
        a: NDBuffer[data_type.value_type, 2, matmul_config.a_shape],
        b: NDBuffer[data_type.value_type, 2, matmul_config.b_shape],
        global_offset: GemmShape,
        valid_tile_bound: GemmShape,
    ) -> Self:
        """Construct a dynamic state from original gemm inputs.

        Parameters:
            matmul_config: Static config of the matmul algorithm.
            data_layout: Data layout of the operand matrices.

        Args:
            c: Buffer allocated for result matrix C.
            a: Buffer allocated for input matrix A.
            b: Buffer allocated for input matrix B.
            global_offset: Offset on the gemm space for this call.
            valid_tile_bound: Size of the tile on the gemm space that this call
                should process.
        """

        # Read out the global gemm space size.
        var global_gemm_size = GemmShape.get[
            matmul_config,
            data_layout,
            data_type,
        ](c, a, b)

        var dynamic_state = Self {
            # Record data pointers for the operand and result space.
            a: a.data,
            b: b.data,
            c: c.data,
            packed_b: DTypePointer[data_type.value_type](),
            packed_a: DTypePointer[data_type.value_type](),
            packed_c: DTypePointer[data_type.value_type](),
            global_gemm_size: global_gemm_size,
            # Read out dynamic tile offset and tile bound.
            global_offset: global_offset,
            valid_tile_bound: valid_tile_bound,
        }

        # Allocate stack buffers for packing.
        dynamic_state._allocate_buffers()
        return dynamic_state

    @always_inline
    fn _allocate_buffers(inout self):
        """Allocate space for packing and maybe other intermediate data space.
        """
        # TODO: read these data out from matmul config.
        self.packed_b = stack_allocation[
            get_pack_data_size[data_type.value_type](),  # Count.
            data_type.value_type,  # Data type.
            simdbytewidth(),  # Alignment.
        ]()

    # TODO:
    #  All three utilities below should be grouped outside of the dynamic state
    #  itself.
    @always_inline
    fn get_global_operand_buffer[
        operand_id: Int, transposed: Bool
    ](self) -> NDBuffer[data_type.value_type, 2]:
        """Utility to get an NDBuffer handle to the global space holding the
        operands.
            Args:
                operand_id (Int): Indicates which operand to get, expect
            GemmIdentifier.OperandA or GemmIdentifier.OperandB.
                transposed (Bool): Indicates whether the data is in transposed
            layout.
        """
        var buffer_pointer = self.a
        var buffer_shape = Index(0, 0)

        @parameter
        if operand_id == GemmIdentifiers.OperandA:
            if transposed:
                buffer_shape = Index(
                    self.global_gemm_size.K, self.global_gemm_size.M
                )

            else:
                buffer_shape = Index(
                    self.global_gemm_size.M, self.global_gemm_size.K
                )
            buffer_pointer = self.a

        @parameter
        if operand_id == GemmIdentifiers.OperandB:
            if transposed:
                buffer_shape = Index(
                    self.global_gemm_size.N, self.global_gemm_size.K
                )

            else:
                buffer_shape = Index(
                    self.global_gemm_size.K, self.global_gemm_size.N
                )
            buffer_pointer = self.b

        return NDBuffer[data_type.value_type, 2](buffer_pointer, buffer_shape)

    @always_inline
    fn get_global_result_buffer(
        self,
    ) -> NDBuffer[data_type.accum_type, 2]:
        """Utility to get an NDBuffer handle to the global space holding the
        result i.e. matrix C buffer.
        """
        return NDBuffer[data_type.accum_type, 2](
            self.c, Index(self.global_gemm_size.M, self.global_gemm_size.N)
        )

    @always_inline
    fn get_packed_operand_buffer[
        operand_id: Int, inner_size: Int
    ](self, tile_dimension: StaticIntTuple[2]) -> NDBuffer[
        data_type.value_type, 3
    ]:
        """Utility to get an NDBuffer handle to the local space holding the
        packed operands when applicable.

        Creates buffer layout from [M][N] to [M/inner_size][N][inner_size]

        Parameters:
            operand_id: Indicates which operand to get, expect
                GemmIdentifier.OperandA or GemmIdentifier.OperandB.
            inner_size: The inner dimension size of the packed layout.

        Args:
            tile_dimension: The size of the packed tile in [M,N] as described
                above.
        """
        var buffer_shape = Index(
            div_ceil(tile_dimension[0], inner_size),
            tile_dimension[1],
            inner_size,
        )

        var buffer_pointer = self.packed_a

        # Assign the underlying data pointer.
        @parameter
        if operand_id == GemmIdentifiers.OperandA:
            buffer_pointer = self.packed_a

        @parameter
        if operand_id == GemmIdentifiers.OperandB:
            buffer_pointer = self.packed_b

        return NDBuffer[data_type.value_type, 3](buffer_pointer, buffer_shape)


@register_passable("trivial")
struct MatmulStaticState:
    """Compile-time state space for the matmul tiling generator. This state
    space holds all the information that is compile-time known, including:
        - Statically known buffer or tile sizes.
        - Statically known unswitch variables on each dimension.
        - Statically known buffer data layouts, i.e. transposed, packed etc.
        - Target simd size.
    """

    # Indicates if each of the gemm dimensions has been unswitched.
    var static_gemm_switch: GemmSwitch

    # Represents the statically known shapes of a subtile or a buffer that
    #  a component may be working on.
    var static_gemm_shape: StaticGemmShape

    # TODO: need a reasonable implementation of a bool vector.
    # Indicates the data layout on the original buffer.
    var static_data_layout: MatmulOperandLayout

    # Target info: TODO wrap this in another struct like target info etc.
    var simd_size: Int

    # Indicates which buffer to read for the current tile level.
    var c_packed: Bool
    var a_packed: Bool
    var b_packed: Bool

    @staticmethod
    fn initialize[
        data_type: MatmulDataType, data_layout: MatmulOperandLayout
    ]() -> Self:
        """Factory utility defining the default values for the static states."""

        return Self {
            static_gemm_switch: GemmSwitch.all_false(),
            static_gemm_shape: StaticGemmShape.all_unknown(),
            static_data_layout: data_layout,
            simd_size: simdwidthof[data_type.value_type](),
            a_packed: False,
            b_packed: False,
            c_packed: False,
        }

    fn set_tile_size(self, dimension_idx: Int, tile_size: Int) -> Self:
        """Updates the tile size to the new value. This utility is used in
        tiling, where the matmul space is statically partitioned into
        smaller matmuls.

        Args:
            dimension_idx: Dimension identifiers for the dimension to be
                updated, expected values are GemmIdentifiers.{DimM, DimN, DimK}.
            tile_size: The tile size to use.

        Returns (Self):
            The updated static state.
        """
        var updated_state = self
        updated_state.static_gemm_shape[dimension_idx] = Dim(tile_size)
        return updated_state

    fn set_static_switch(self, dimension_idx: Int, switch: Bool) -> Self:
        """Updates the static switch to the new value.

        Args:
            dimension_idx: Dimension identifiers for the dimension to be
                updated, expected values are GemmIdentifiers.{DimM, DimN, DimK}.
            switch: The new switch value to use.

        Returns (Self):
            The updated static state.
        """
        var updated_state = self
        updated_state.static_gemm_switch[dimension_idx] = switch
        return updated_state

    fn set_packed(self, idx: Int) -> Self:
        """Set the packed status of the given operand.

        Args:
            idx: Operand identifiers for the dimension to be updated, expected
                values are GemmIdentifier.{ResultC, OperandA, OperandB}.

        Returns (Self):
            The updated static state.
        """
        var updated = self
        if idx == GemmIdentifiers.ResultC:
            updated.c_packed = True
        elif idx == GemmIdentifiers.OperandA:
            updated.a_packed = True
        elif idx == GemmIdentifiers.OperandB:
            updated.b_packed = True
        return updated


@register_passable("trivial")
struct MatmulActionKind:
    """Enum struct representing an "op code" that a matmul generator can
    perform."""

    # The underlying representation.
    var value: Int

    # NoOp: placeholder op used as default value.
    alias NoOp = MatmulActionKind(0)

    # TileStatic: tile a dimension statically.
    alias TileStatic = MatmulActionKind(1)

    # TileDynamic: tile a dimension dynamically.
    alias TileDynamic = MatmulActionKind(2)

    # TileAndUnswitch: tile a dimension statically and peel the remainder
    #   iterations.
    alias TileAndUnswitch = MatmulActionKind(3)

    # Placeholder for micro kernel call.
    alias MicroKernel = MatmulActionKind(4)

    # Placeholder for matrix packing call.
    alias PackB = MatmulActionKind(5)

    # PrintTileState: Test and debug utility, prints the current static and
    #  dynamic state.
    alias PrintTileState = MatmulActionKind(6)

    fn __init__(value: Int) -> Self:
        return Self {value: value}

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value


@register_passable("trivial")
struct MatmulAction:
    """Represents a single "operation" performed by matmul generator."""

    # Kind of operation to perform, i.e. the "op code".
    var kind: MatmulActionKind

    # Indicates the dimension the current operation acts on when applicable.
    var tiled_dimension: Int

    # Indicates the tile sizes the current operation uses when applicable.
    var tile_sizes: VariadicList[Int]

    # Indicates if matmul epilog is performed at this level.
    var do_epilog: Bool


struct PackInterface[
    static_state: MatmulStaticState,
    data_type: MatmulDataType,
    packing_action: MatmulAction,
]:
    """Interface wrapper for sub-matrix packing routine implementations. This
    struct provides a unified interface to call the packing routines. Additional
    packing implementations, e.g. amx transpose etc can be added and dispatched
    in this struct.
    """

    @staticmethod
    @always_inline
    fn pack(dynamic_state: MatmulDynamicState[data_type]):
        """Interface API. This is the method that the matmul generator calls and
        this method is responsible for statically dispatching to implementations
        based on the action parameters.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
        """

        @parameter
        if packing_action.kind == MatmulActionKind.PackB:
            Self._pack_b(dynamic_state)

    @always_inline
    @staticmethod
    fn _pack_b(dynamic_state: MatmulDynamicState[data_type]):
        """
        Implementation of pack_b using PackMatrixRows and PackMatrixCols
            routines.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
        """
        # Parameter checks on supported configuration so far.

        # Construct global buffer for matrix B.
        var b_buffer = dynamic_state.get_global_operand_buffer[
            GemmIdentifiers.OperandB,
            static_state.static_data_layout.transpose_b,
        ]()

        # inner dimension of packed layout.
        alias inner_size = static_state.static_data_layout.pack_b_inner_size
        var tile_N = max(dynamic_state.valid_tile_bound.N, inner_size)

        # Construct local buffer for matrix packed B.
        var b_packed_buffer = dynamic_state.get_packed_operand_buffer[
            GemmIdentifiers.OperandB, inner_size
        ](
            # Target amount of data to pack.
            Index(
                tile_N,
                dynamic_state.valid_tile_bound.K,
            )
        )

        if static_state.static_data_layout.transpose_b:
            PackMatrixRows[
                # original buffer size.
                DimList.create_unknown[2](),
                # packed buffer size.
                DimList.create_unknown[3](),
                # data type.
                data_type.value_type,
                # simd size.
                static_state.simd_size,
                # pack inner size, in number of elements.
                inner_size,
            ].run(
                b_packed_buffer,
                b_buffer,
                # Input is [N, K]:
                # Starting global offset for packing.
                Index(
                    dynamic_state.global_offset.N,
                    dynamic_state.global_offset.K,
                ),
                # Targeted tile size to pack.
                Index(
                    tile_N,
                    dynamic_state.valid_tile_bound.K,
                ),
                # Valid amount of input from the starting offset.
                Index(
                    dynamic_state.valid_tile_bound.N,
                    dynamic_state.valid_tile_bound.K,
                ),
            )
        else:
            PackMatrixCols[
                # original buffer size.
                DimList.create_unknown[2](),
                # packed buffer size.
                DimList.create_unknown[3](),
                # data type.
                data_type.value_type,
                # simd size.
                static_state.simd_size,
                # pack inner size, in number of elements.
                inner_size,
                # use VNNI
                False,
                # use I8MM,
                False,
            ].run(
                # Input is [K, N]:
                b_packed_buffer,
                b_buffer,
                # Starting global offset for packing.
                Index(
                    dynamic_state.global_offset.K,
                    dynamic_state.global_offset.N,
                ),
                # Targeted tile size to pack.
                Index(dynamic_state.valid_tile_bound.K, tile_N),
                # Valid amount of input from the starting offset.
                Index(
                    dynamic_state.valid_tile_bound.K,
                    dynamic_state.valid_tile_bound.N,
                ),
            )


struct MicroKernelInterface[
    static_state: MatmulStaticState,
    data_type: MatmulDataType,
    micro_kernel_action: MatmulAction,
]:
    """Interface wrapper for inner micro-kernel routine implementations. This
    struct provides a unified interface to call the micro kernels routines.
    Additional micro kernel implementations, e.g. amx, neon, can be added and
    dispatched in this struct.
    """

    @staticmethod
    @always_inline
    fn run(dynamic_state: MatmulDynamicState[data_type]):
        """Interface API.This is the method that the matmul generator calls and
        this method is responsible for statically dispatching to implementations
        based on the action parameters.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
        """
        # TODO:
        #  Statically dispatch micro kernels, once we have other kernels.
        Self._b_packed_micro_kernel(dynamic_state)

    @staticmethod
    @always_inline
    fn _b_packed_micro_kernel(dynamic_state: MatmulDynamicState[data_type]):
        """Microkernel implementation with packed B and unpacked A. Best used
        on AVX targets, but AVX not required.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
        """
        # This micro-kernel needs b to be packed, and a, c not packed.
        constrained[static_state.b_packed]()
        constrained[static_state.a_packed == False]()
        constrained[static_state.c_packed == False]()

        # This micro-kernel requires constant tile size on the M and N dimension
        #  for register blocking.
        constrained[static_state.static_gemm_shape.M.has_value()]()

        # Calculate inner size on the packed layout.
        alias inner_size = static_state.static_data_layout.pack_b_inner_size
        var tile_N = max(dynamic_state.valid_tile_bound.N, inner_size)

        # Create buffer handles to pass to inner kernel.
        var c_buffer = dynamic_state.get_global_result_buffer()
        var a_buffer = dynamic_state.get_global_operand_buffer[
            GemmIdentifiers.OperandA,
            static_state.static_data_layout.transpose_a,
        ]()
        var b_packed_buffer = dynamic_state.get_packed_operand_buffer[
            GemmIdentifiers.OperandB, inner_size
        ](
            # Target amount of data to pack.
            Index(
                tile_N,
                dynamic_state.valid_tile_bound.K,
            )
        )

        # Launch the actual inner kernel.
        # TODO: make the parameters depend on matmul config.
        #  The prefetch distance should go in a target info struct.
        alias prefetch_b_distance_k = 4
        MatmulInnerLoopBPacked[
            # shape a
            DimList.create_unknown[2](),
            # shape c
            DimList.create_unknown[2](),
            # shape packed_b
            DimList.create_unknown[3](),
            data_type.value_type,
            data_type.value_type,
            data_type.accum_type,
            static_state.simd_size,
            static_state.static_gemm_shape.M.get(),
            inner_size,
            static_state.static_gemm_switch.N,
            prefetch_b_distance_k,
            saturated_vnni=True,
        ].run(
            c_buffer,
            a_buffer,
            b_packed_buffer,
            dynamic_state.global_offset.as_index(),
            # TODO:
            #  Shouldn't really need this. This is problem within
            #    MatmulInnerLoopBPacked
            #  Need to propagate this change into the micro-kernel itself.
            dynamic_state.valid_tile_bound.as_index()
            + dynamic_state.global_offset.as_index(),
            Index(
                tile_N,
                dynamic_state.valid_tile_bound.K,
            ),
        )


struct MatmulGenerator[
    data_type: MatmulDataType,
    epilog_function: fn (MatmulDynamicState[data_type]) capturing -> None,
]:
    """A matmul generator that realizes matmul algorithms given a set of tiling
    actions.
    """

    @staticmethod
    @always_inline
    fn generate[
        static_state: MatmulStaticState, actions: VariadicList[MatmulAction]
    ](
        dynamic_state: MatmulDynamicState[data_type],
        action_tile_shapes: VariadicList[TileShape],
    ):
        """Main API of the matmul generator. Realizes matmul operation given the
        initial states and configurations.

        Parameters:
            static_state: The initial static state of the matmul space.
            actions: The set of tiling actions for realizing the matmul
                configuration.

        Args:
            dynamic_state: The runtime arguments to the matmul operator.
            action_tile_shapes: The list of tile shapes needed by matmul actions
                that use a dynamic tile size. The size of this list needs to be
                equal to the number of matmul actions that use a dynamic tile
                size. The order of the list must exactly match the `actions`
                parameter.
        """

        debug_assert(
            len(actions) == len(action_tile_shapes),
            "Number of static actions must equal number of dynamic tile sizes",
        )

        Self._generate[static_state, 0, actions](
            dynamic_state, action_tile_shapes
        )

    @staticmethod
    @always_inline
    fn _generate[
        static_state: MatmulStaticState,
        current_action_idx: Int,
        actions: VariadicList[MatmulAction],
    ](
        dynamic_state: MatmulDynamicState[data_type],
        action_tile_shapes: VariadicList[TileShape],
    ):
        """Dispatching utility for each of the static tiling actions.

        Parameters:
            static_state: The current static state before applying the current
                action.
            current_action_idx: Index of the current action to apply on the
                given list of actions.
            actions: The list of actions of the matmul generator.
        """

        @parameter
        if current_action_idx == len(actions):
            return
        # Read the current action.
        alias current_action = actions[current_action_idx]

        @parameter
        if current_action.kind == MatmulActionKind.TileStatic:
            # Realize static tile.
            Self._static_tile[static_state, current_action_idx, actions](
                dynamic_state, action_tile_shapes
            )
            Self._process_epilog[current_action, static_state](dynamic_state)

        @parameter
        if current_action.kind == MatmulActionKind.TileDynamic:
            # Realize dynamic tile.
            Self._dynamic_tile[static_state, current_action_idx, actions](
                dynamic_state, action_tile_shapes
            )
            Self._process_epilog[current_action, static_state](dynamic_state)

        @parameter
        if current_action.kind == MatmulActionKind.TileAndUnswitch:
            # Realize tile and unswitch.
            Self._tile_and_unswitch[static_state, current_action_idx, actions](
                dynamic_state, action_tile_shapes
            )
            Self._process_epilog[current_action, static_state](dynamic_state)

        @parameter
        if current_action.kind == MatmulActionKind.PrintTileState:
            Self._print_tile_state[static_state](dynamic_state)

        @parameter
        if current_action.kind == MatmulActionKind.PackB:
            # Pack the B subtile.
            PackInterface[static_state, data_type, current_action].pack(
                dynamic_state
            )

            # Generate the next loop level.
            Self._generate[
                static_state.set_packed(GemmIdentifiers.OperandB),
                current_action_idx + 1,
                actions,
            ](dynamic_state, action_tile_shapes)

            Self._process_epilog[current_action, static_state](dynamic_state)

        @parameter
        if current_action.kind == MatmulActionKind.MicroKernel:
            # Call the microkernel on the current tile.
            MicroKernelInterface[static_state, data_type, current_action].run(
                dynamic_state
            )
            # Note: fusing epilog into microkernel needs to be implemented inside
            # the microkernel.

    @staticmethod
    @always_inline
    fn _process_epilog[
        action: MatmulAction, static_state: MatmulStaticState
    ](dynamic_state: MatmulDynamicState[data_type]):
        """A general step that should be added in each of the tiling actions
        to enable support for configurable epilog fusion.

        Example usage: suppose the tile configuration generates the following
        loops:
            for x in Tile0:
                for y in Tile1:
                    for z in Tile2:
                        matmul_fma's
                    // (Tile2.do_epilog==True) will realize the epilog here.
                // (Tile1.do_epilog==True) will realize the epilog here.
            // (Tile0.do_epilog==True) will realize the epilog here.

        Parameters:
            action: The current tiling action to process.
            static_state: The static tile state.

        Args:
            dynamic_state: The dynamic tile state, containing the dynamic bounds
                of the current tile.
        """

        @parameter
        if action.do_epilog:
            epilog_function(dynamic_state)

    @staticmethod
    @always_inline
    fn _static_tile[
        static_state: MatmulStaticState,
        current_action_idx: Int,
        actions: VariadicList[MatmulAction],
    ](
        dynamic_state: MatmulDynamicState[data_type],
        action_tile_shapes: VariadicList[TileShape],
    ):
        """Implementation of the static tile action.

        Parameters:
            static_state: The current static state before applying the current
                action.
            current_action_idx: Index of the current action to apply on the
                given list of actions.
            actions: The list of actions of the matmul generator.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
            action_tile_shapes: The list of tile shapes needed by matmul actions
                that use a dynamic tile size. The size of this list needs to be
                equal to the number of matmul actions that use a dynamic tile
                size. The order of the list must exactly match the `actions`
                parameter.
        """
        alias current_action = actions[current_action_idx]

        @always_inline
        @parameter
        fn static_tiled_func[tile_size: Int](local_offset: Int):
            Self._generate[
                # static tile action puts a static tile size and an unswitch
                #  on the given dimension.
                static_state.set_tile_size(
                    current_action.tiled_dimension, tile_size
                ).set_static_switch(current_action.tiled_dimension, True),
                current_action_idx + 1,
                actions,
            ](
                # note that the return value of apply_offset and apply_new_bound
                # is a COPY of the dynamic_state, with the offset and new bound
                # adjusted appropriately
                dynamic_state.apply_offset(
                    current_action.tiled_dimension, local_offset
                ).apply_new_bound(current_action.tiled_dimension, tile_size),
                action_tile_shapes,
            )

        tile[static_tiled_func, current_action.tile_sizes](
            0,  # local start offset
            dynamic_state.valid_tile_bound[current_action.tiled_dimension],
        )

    @staticmethod
    @always_inline
    fn _dynamic_tile[
        static_state: MatmulStaticState,
        current_action_idx: Int,
        actions: VariadicList[MatmulAction],
    ](
        dynamic_state: MatmulDynamicState[data_type],
        action_tile_shapes: VariadicList[TileShape],
    ):
        """Implementation of the dynamic tile action.

        Parameters:
            static_state: The current static state before applying the current
                action.
            current_action_idx: Index of the current action to apply on the
                given list of actions.
            actions: The list of actions of the matmul generator.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
            action_tile_shapes: The list of tile shapes needed by matmul actions
                that use a dynamic tile size. The size of this list needs to be
                equal to the number of matmul actions that use a dynamic tile
                size. The order of the list must exactly match the `actions`
                parameter.
        """
        alias current_action = actions[current_action_idx]

        # Dynamic tile shouldn't be applied to a dimension that's already
        #  static-shaped. Currently they do not yet compose.
        # Could add support if needed.
        constrained[
            not static_state.static_gemm_shape[
                current_action.tiled_dimension
            ].has_value()
        ]()

        @always_inline
        @parameter
        fn dynamic_tiled_func(local_offset: Int, tile_size: Int):
            # dynamic tile only updates the dynamic offset and bound.
            Self._generate[static_state, current_action_idx + 1, actions,](
                # note that the return value of apply_offset and apply_new_bound
                # is a COPY of the dynamic_state, with the offset and new bound
                # adjusted appropriately
                dynamic_state.apply_offset(
                    current_action.tiled_dimension, local_offset
                ).apply_new_bound(current_action.tiled_dimension, tile_size),
                action_tile_shapes,
            )

        # TODO: extend to add support for real dynamic tiles. i.e. implement a
        #  "heuristic" action that saves the targeted tile size on the dynamic
        #  state space.
        tile[dynamic_tiled_func](
            0,  # local start offset
            dynamic_state.valid_tile_bound[current_action.tiled_dimension],
            action_tile_shapes[current_action_idx],
        )

    @staticmethod
    @always_inline
    fn _tile_and_unswitch[
        static_state: MatmulStaticState,
        current_action_idx: Int,
        actions: VariadicList[MatmulAction],
    ](
        dynamic_state: MatmulDynamicState[data_type],
        action_tile_shapes: VariadicList[TileShape],
    ):
        """Implementation of the tile and unswitch action.

        Parameters:
            static_state: The current static state before applying the current
                action.
            current_action_idx: Index of the current action to apply on the
                given list of actions.
            actions: The list of actions of the matmul generator.

        Args:
            dynamic_state: The current dynamic state before applying the current
                action.
            action_tile_shapes: The list of tile shapes needed by matmul actions
                that use a dynamic tile size. The size of this list needs to be
                equal to the number of matmul actions that use a dynamic tile
                size. The order of the list must exactly match the `actions`
                parameter.
        """
        alias current_action = actions[current_action_idx]

        @always_inline
        @parameter
        fn dynamic_switched_func[
            static_switch: Bool
        ](local_offset: Int, upperbound: Int, tile_size: Int):
            # Set the tile size and unswitch flag accordingly.
            Self._generate[
                static_state.set_static_switch(
                    current_action.tiled_dimension, static_switch
                ),
                current_action_idx + 1,
                actions,
            ](
                dynamic_state.apply_offset(
                    current_action.tiled_dimension, local_offset
                ).apply_new_bound(
                    current_action.tiled_dimension, min(tile_size, upperbound)
                ),
                action_tile_shapes,
            )

        tile_and_unswitch[dynamic_switched_func](
            0,  # local start offset
            dynamic_state.valid_tile_bound[current_action.tiled_dimension],
            action_tile_shapes[current_action_idx],
        )

    @staticmethod
    fn _print_tile_state[
        static_state: MatmulStaticState
    ](dynamic_state: MatmulDynamicState[data_type]):
        """Implementation of the print tile state utility.

        Parameters:
            static_state: The current static state.

        Args:
            dynamic_state: the current dynamic state.
        """
        print("global offset:", dynamic_state.global_offset.as_index())
        print("dynamic tile bound:", dynamic_state.valid_tile_bound.as_index())
        print(
            "static tile bound:",
            static_state.static_gemm_shape._as_gemm_shape().as_index(),
        )
        print(
            "transpose a and b:",
            static_state.static_data_layout.transpose_a,
            static_state.static_data_layout.transpose_b,
        )
        print(
            "packed a,b,c:",
            static_state.a_packed,
            static_state.b_packed,
            static_state.c_packed,
        )
        print(
            "unswitched m,n,k:",
            static_state.static_gemm_switch.M,
            static_state.static_gemm_switch.N,
            static_state.static_gemm_switch.K,
        )
        print("---------------------------\n")


# TODO: In the future, this could be encapsulated in a MatmulActionDynamicData object
alias TileShape = VariadicList[Int]


struct TiledMatmulGenerated[
    config: MatmulConfig,
    data_type: MatmulDataType,
    data_layout: MatmulOperandLayout,
]:
    """A single-thread matmul routine backed by the matmul generator."""

    @staticmethod
    fn run(
        c: NDBuffer[data_type.accum_type, 2, config.c_shape],
        a: NDBuffer[data_type.value_type, 2, config.a_shape],
        b: NDBuffer[data_type.value_type, 2, config.b_shape],
        global_tile_offset: GemmShape,
        global_tile_shape: GemmShape,
    ):
        """Interface function to run tiled matmul on a given sub-tile.

        Args:
            c: Pre-allocated buffer space for result.
            a: Operand A of the matmul.
            b: Operand B of the matmul.
            global_tile_offset: Tile offset on the original buffer.
            global_tile_shape: Tile shape this call will process.
        """
        # TODO: (once we have more) wrap these generators behind a common
        #  interface.
        # MatmulGenerator does not support i8mm or vnni yet so set the factor to 1
        alias factor = 1
        var tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size, 1
        ](global_tile_shape)

        # Define an no-op epilog function.
        @always_inline
        @parameter
        fn epilog_no_op(dynamic_state: MatmulDynamicState[data_type]):
            return

        MatmulGenerator[data_type, epilog_no_op].generate[
            MatmulStaticState.initialize[data_type, data_layout](),
            VariadicList[MatmulAction](
                # Tile on K Dimension.
                MatmulAction {
                    kind: MatmulActionKind.TileDynamic,
                    tiled_dimension: GemmIdentifiers.DimK,
                    tile_sizes: VariadicList[Int](),  # unused
                    do_epilog: False,
                },
                # Tile on N Dimension.
                MatmulAction {
                    kind: MatmulActionKind.TileAndUnswitch,
                    tile_sizes: VariadicList[Int](),  # unused
                    tiled_dimension: GemmIdentifiers.DimN,
                    do_epilog: False,
                },
                # Pack B on the current tile.
                MatmulAction {
                    kind: MatmulActionKind.PackB,
                    tile_sizes: VariadicList[Int](),  # unused
                    tiled_dimension: 0,  # unused
                    do_epilog: False,
                },
                # Tile on M Dimension.
                MatmulAction {
                    kind: MatmulActionKind.TileStatic,
                    tile_sizes: VariadicList[Int](
                        config.a_row_size, 4, 3, 2, 1
                    ),
                    tiled_dimension: GemmIdentifiers.DimM,
                    do_epilog: False,
                },
                # Call the micro kernel.
                #  TODO: add micro kernel selector.
                MatmulAction {
                    kind: MatmulActionKind.MicroKernel,
                    tile_sizes: VariadicList[Int](),  # unused
                    tiled_dimension: 0,  # unused
                    do_epilog: False,
                },
            ),
        ](
            MatmulDynamicState[data_type].get[
                # matmul config
                config,
                # matmul data layout
                data_layout,
            ](
                c,
                a,
                b,
                # global offset
                global_tile_offset,
                # dynamic tile bound
                global_tile_shape,
            ),
            VariadicList[TileShape](
                TileShape(tile_n_k[1]),  # K
                # the N tile_sizes can't be less than pack_inner_size
                # because the inner dim of the tile is fixed to be pack_inner_size
                TileShape(tile_n_k[0], config.pack_inner_size),  # N
                TileShape(),
                TileShape(),
                TileShape(),
            ),
        )


struct TiledMatmulBiasGenerated[
    config: MatmulConfig,
    data_type: MatmulDataType,
    data_layout: MatmulOperandLayout,
]:
    """A single-thread matmul routine backed by the matmul generator."""

    @staticmethod
    fn run(
        c: NDBuffer[data_type.accum_type, 2, config.c_shape],
        a: NDBuffer[data_type.value_type, 2, config.a_shape],
        b: NDBuffer[data_type.value_type, 2, config.b_shape],
        bias: NDBuffer[data_type.accum_type, 1, config.shape_bias],
        global_tile_offset: GemmShape,
        global_tile_shape: GemmShape,
    ):
        """Interface function to run tiled matmul on a given sub-tile.

        Args:
            c: Pre-allocated buffer space for result.
            a: Operand A of the matmul.
            b: Operand B of the matmul.
            bias: The bias value.
            global_tile_offset: Tile offset on the original buffer.
            global_tile_shape: Tile shape this call will process.
        """
        # TODO: (once we have more) wrap these generators behind a common
        #  interface.
        # MatmulGenerator does not support i8mm or vnni yet so set the factor to 1
        var tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size, 1
        ](global_tile_shape)

        # Define a bias epilog function that can apply the bias vector at
        #  selected tile level.
        # TODO:
        #  1. Would be nice to be able to pass in static states as well.
        #  2. Would be nice not to have to define this epilog inplace, current
        #  bottleneck is `data_type`.
        @always_inline
        @parameter
        fn epilog_bias(dynamic_state: MatmulDynamicState[data_type]):
            # This check ensures that we only add the bias once
            if not (
                dynamic_state.global_offset.K + dynamic_state.valid_tile_bound.K
                == dynamic_state.global_gemm_size.K
            ):
                return
            # Get global c buffer.
            var c_buffer = dynamic_state.get_global_result_buffer()

            # Loop over the current tile.
            @always_inline
            @__copy_capture(c_buffer)
            @parameter
            fn bias_col_chunk[col_chunk_size: Int](idx_n: Int):
                var n_coord = idx_n + dynamic_state.global_offset.N
                var bias_val = bias.load[width=col_chunk_size](n_coord)
                for idx_m in range(dynamic_state.valid_tile_bound.M):
                    var m_coord = idx_m + dynamic_state.global_offset.M
                    var c_coord = Index(m_coord, n_coord)
                    var c_val = c_buffer.load[width=col_chunk_size](c_coord)
                    c_buffer.store[width=col_chunk_size](
                        c_coord, c_val + bias_val
                    )

            # TODO: Search tile factor
            vectorize[bias_col_chunk, config.simd_size, unroll_factor=4](
                dynamic_state.valid_tile_bound.N
            )

        MatmulGenerator[data_type, epilog_bias].generate[
            MatmulStaticState.initialize[data_type, data_layout](),
            VariadicList[MatmulAction](
                # Tile on K Dimension.
                MatmulAction {
                    kind: MatmulActionKind.TileDynamic,
                    tiled_dimension: GemmIdentifiers.DimK,
                    tile_sizes: VariadicList[Int](),  # unused
                    do_epilog: False,
                },
                # Tile on N Dimension.
                MatmulAction {
                    kind: MatmulActionKind.TileAndUnswitch,
                    tile_sizes: VariadicList[Int](
                        # Preferred tile size.
                        config.pack_inner_size,
                        # Secondary tile sizes.
                        config.simd_size * 2,
                        config.simd_size,
                    ),
                    tiled_dimension: GemmIdentifiers.DimN,
                    do_epilog: False,
                },
                # Pack B on the current tile.
                MatmulAction {
                    kind: MatmulActionKind.PackB,
                    tile_sizes: VariadicList[Int](),  # unused
                    tiled_dimension: 0,  # unused
                    do_epilog: False,  # TODO: assert false within unsupported actions.
                },
                # Tile on M Dimension.
                MatmulAction {
                    kind: MatmulActionKind.TileStatic,
                    tile_sizes: VariadicList[Int](
                        config.a_row_size, 4, 3, 2, 1
                    ),
                    tiled_dimension: GemmIdentifiers.DimM,
                    do_epilog: True,  # Epilog applied within each of the panels.
                },
                # Call the micro kernel.
                #  TODO: add micro kernel selector.
                MatmulAction {
                    kind: MatmulActionKind.MicroKernel,
                    tile_sizes: VariadicList[Int](),  # unused
                    tiled_dimension: 0,  # unused
                    do_epilog: False,
                },
            ),
        ](
            MatmulDynamicState[data_type].get[
                # matmul config
                config,
                # matmul data layout
                data_layout,
            ](
                c,
                a,
                b,
                # global offset
                global_tile_offset,
                # dynamic tile bound
                global_tile_shape,
            ),
            VariadicList[TileShape](
                TileShape(tile_n_k[1]),  # K
                # the N tile_sizes can't be less than pack_inner_size
                # because the inner dim of the tile is fixed to be pack_inner_size
                TileShape(tile_n_k[0], config.pack_inner_size),  # N
                TileShape(),
                TileShape(),
                TileShape(),
            ),
        )
