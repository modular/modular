# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu import barrier, WARP_SIZE
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.id import block_idx, thread_idx
from gpu.intrinsics import threadfence
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.fillers import arange
from layout.int_tuple import to_int
from layout.layout import print_layout
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core_async import (
    tile_layout_k_major,
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
)
from layout.tensor_core import get_accum_type
from layout.tma_async import TMATensorTile, create_tma_tile, TMABarrier
from memory import bitcast
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from sys import sizeof


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 2](0, 1))
fn tma_wgmma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = True,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout],
    num_iters: UInt,
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias accum_type = get_accum_type[a_type]()
    wgmma_op = TensorCoreAsync[
        accum_type, a_type, b_type, wgmma_shape, transpose_b
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[1]
    alias num_m_mmas = BM // wgmma_shape[0]
    alias num_n_mmas = BN // wgmma_shape[1]

    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    alias expected_bytes = a_smem_layout.size() * sizeof[
        a_type
    ]() + b_smem_layout.size() * sizeof[b_type]()
    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()

    mbar = TMABarrier()
    if thread_idx.x == 0:
        mbar.init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar.expect_bytes(expected_bytes)
            a_tma_op.async_copy(
                a_smem_tile,
                mbar,
                (UInt(i) * BK, block_idx.y * BM),
            )
            b_tma_op.async_copy(
                b_smem_tile, mbar, (UInt(i) * BK, block_idx.x * BN)
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar.wait(phase)
        phase ^= 1

        wgmma_op.arrive()
        wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        wgmma_op.wait_for_all()

        barrier()

    c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    warp_id = thread_idx.x // WARP_SIZE

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            # (m_mma, n_mma) is coordinates for a warp group's tile.
            # A warp group is 4x1 warps.
            warp_tile = c_gmem_tile.tile[wgmma_shape[0] // 4, wgmma_shape[1]](
                m_mma * 4 + warp_id, n_mma
            )

            # Tile at (mma_id, 0) is a long vector containing all fragments
            # for this warp.
            c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

            # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
            # elementwise. This pattern repeates to fill the warp tile.
            copy_local_to_dram[Layout.row_major(8, 4)](
                warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
            )


def test_tma_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = True,
](ctx: DeviceContext):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias WGMMA_M = wgmma_shape[0]
    alias WGMMA_N = wgmma_shape[1]
    alias WGMMA_K = wgmma_shape[2]

    print(
        "== wgmma_bf16_bf16_f32_"
        + String(BM)
        + "x"
        + String(BN)
        + "x"
        + String(BK)
        + "_transb_"
        + String(WGMMA_M)
        + "x"
        + String(WGMMA_N)
        + "x"
        + String(WGMMA_K)
    )

    constrained[transpose_b, "Only support transpose_b for now"]()

    alias M = prob_shape[0]
    alias N = prob_shape[1]
    alias K = prob_shape[2]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(a.tensor())
    var b = ManagedLayoutTensor[
        b_type,
        Layout.row_major(N, K),
    ](ctx)
    arange(b.tensor())

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    # Shared memory tile layouts
    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK]()

    a_tma_op = create_tma_tile[a_type, 2, Index(BM, BK)](ctx, a.device_tensor())
    b_tma_op = create_tma_tile[b_type, 2, Index(BN, BK)](ctx, b.device_tensor())

    alias kernel = tma_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        Layout.row_major(BM, BK),
        Layout.row_major(BN, BK),
        Layout.row_major(M, N),
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        block_tile_shape,
        wgmma_shape,
        a_smem_layout,
        b_smem_layout,
        transpose_b=True,
    ]
    var func = ctx.compile_function[
        kernel,
        _target = _get_gpu_target["sm_90"](),
    ]()

    ctx.enqueue_function(
        func,
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        K // BK,
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()

    print(c.tensor())
    _ = a^
    _ = b^
    _ = c^


# CHECK-LABEL: wgmma_bf16_bf16_f32_64x8x16_transb_64x8x32
# CHECK: 10432.0 26240.0 42240.0 58112.0 73728.0 89600.0 105472.0 121344.0
# CHECK: 26240.0 74752.0 123392.0 172032.0 221184.0 270336.0 317440.0 366592.0
# CHECK: 42240.0 123392.0 204800.0 286720.0 368640.0 448512.0 532480.0 610304.0
# CHECK: 58112.0 172032.0 286720.0 401408.0 514048.0 630784.0 741376.0 856064.0
# CHECK: 73728.0 221184.0 368640.0 514048.0 663552.0 806912.0 954368.0 1105920.0
# CHECK: 89600.0 270336.0 448512.0 630784.0 806912.0 987136.0 1171456.0 1351680.0
# CHECK: 105472.0 317440.0 532480.0 741376.0 954368.0 1171456.0 1384448.0 1589248.0
# CHECK: 121344.0 366592.0 610304.0 856064.0 1105920.0 1351680.0 1589248.0 1835008.0
# CHECK: 137216.0 415744.0 692224.0 970752.0 1253376.0 1523712.0 1802240.0 2080768.0
# CHECK: 153600.0 464896.0 774144.0 1089536.0 1392640.0 1703936.0 2015232.0 2326528.0
# CHECK: 168960.0 512000.0 856064.0 1196032.0 1540096.0 1884160.0 2228224.0 2572288.0
# CHECK: 185344.0 561152.0 937984.0 1310720.0 1687552.0 2064384.0 2441216.0 2818048.0
# CHECK: 200704.0 610304.0 1019904.0 1425408.0 1835008.0 2244608.0 2654208.0 3063808.0
# CHECK: 217088.0 659456.0 1097728.0 1540096.0 1982464.0 2424832.0 2867200.0 3309568.0
# CHECK: 232448.0 708608.0 1179648.0 1654784.0 2129920.0 2605056.0 3080192.0 3555328.0
# CHECK: 248832.0 757760.0 1261568.0 1769472.0 2277376.0 2785280.0 3293184.0 3801088.0
# CHECK: 264192.0 802816.0 1343488.0 1884160.0 2424832.0 2965504.0 3506176.0 4046848.0
# CHECK: 280576.0 851968.0 1425408.0 1998848.0 2572288.0 3145728.0 3719168.0 4292608.0
# CHECK: 296960.0 901120.0 1507328.0 2113536.0 2719744.0 3325952.0 3932160.0 4521984.0
# CHECK: 311296.0 950272.0 1589248.0 2228224.0 2867200.0 3506176.0 4145152.0 4784128.0
# CHECK: 327680.0 999424.0 1671168.0 2342912.0 3014656.0 3686400.0 4358144.0 5013504.0
# CHECK: 344064.0 1048576.0 1753088.0 2457600.0 3162112.0 3866624.0 4554752.0 5275648.0
# CHECK: 360448.0 1097728.0 1835008.0 2572288.0 3309568.0 4046848.0 4784128.0 5505024.0
# CHECK: 374784.0 1146880.0 1916928.0 2686976.0 3457024.0 4227072.0 4980736.0 5767168.0
# CHECK: 391168.0 1196032.0 1998848.0 2801664.0 3604480.0 4390912.0 5210112.0 5996544.0
# CHECK: 407552.0 1245184.0 2080768.0 2916352.0 3751936.0 4587520.0 5406720.0 6258688.0
# CHECK: 423936.0 1294336.0 2162688.0 3031040.0 3899392.0 4751360.0 5636096.0 6488064.0
# CHECK: 438272.0 1343488.0 2244608.0 3145728.0 4046848.0 4947968.0 5832704.0 6750208.0
# CHECK: 454656.0 1384448.0 2326528.0 3260416.0 4194304.0 5111808.0 6062080.0 6979584.0
# CHECK: 471040.0 1433600.0 2408448.0 3375104.0 4325376.0 5308416.0 6258688.0 7241728.0
# CHECK: 487424.0 1482752.0 2490368.0 3489792.0 4489216.0 5472256.0 6488064.0 7471104.0
# CHECK: 501760.0 1531904.0 2572288.0 3604480.0 4620288.0 5668864.0 6684672.0 7733248.0
# CHECK: 518144.0 1581056.0 2654208.0 3719168.0 4784128.0 5832704.0 6914048.0 7962624.0
# CHECK: 532480.0 1630208.0 2736128.0 3833856.0 4915200.0 6029312.0 7110656.0 8224768.0
# CHECK: 548864.0 1679360.0 2818048.0 3932160.0 5079040.0 6193152.0 7340032.0 8454144.0
# CHECK: 565248.0 1728512.0 2883584.0 4046848.0 5210112.0 6389760.0 7536640.0 8716288.0
# CHECK: 581632.0 1777664.0 2965504.0 4161536.0 5373952.0 6553600.0 7766016.0 8978432.0
# CHECK: 598016.0 1826816.0 3047424.0 4292608.0 5505024.0 6750208.0 7962624.0 9175040.0
# CHECK: 614400.0 1875968.0 3129344.0 4390912.0 5668864.0 6914048.0 8192000.0 9437184.0
# CHECK: 630784.0 1925120.0 3211264.0 4521984.0 5799936.0 7110656.0 8388608.0 9699328.0
# CHECK: 647168.0 1974272.0 3293184.0 4620288.0 5963776.0 7274496.0 8585216.0 9961472.0
# CHECK: 659456.0 2023424.0 3375104.0 4751360.0 6094848.0 7471104.0 8847360.0 10158080.0
# CHECK: 675840.0 2072576.0 3457024.0 4849664.0 6258688.0 7634944.0 9043968.0 10420224.0
# CHECK: 692224.0 2113536.0 3538944.0 4980736.0 6389760.0 7831552.0 9240576.0 10682368.0
# CHECK: 708608.0 2162688.0 3620864.0 5079040.0 6553600.0 7995392.0 9437184.0 10944512.0
# CHECK: 724992.0 2211840.0 3702784.0 5210112.0 6684672.0 8192000.0 9699328.0 11141120.0
# CHECK: 741376.0 2260992.0 3784704.0 5308416.0 6848512.0 8355840.0 9895936.0 11403264.0
# CHECK: 757760.0 2310144.0 3866624.0 5439488.0 6979584.0 8519680.0 10092544.0 11665408.0
# CHECK: 774144.0 2359296.0 3948544.0 5537792.0 7143424.0 8716288.0 10289152.0 11862016.0
# CHECK: 786432.0 2408448.0 4030464.0 5668864.0 7274496.0 8912896.0 10485760.0 12124160.0
# CHECK: 802816.0 2457600.0 4112384.0 5767168.0 7405568.0 9043968.0 10747904.0 12386304.0
# CHECK: 819200.0 2506752.0 4194304.0 5865472.0 7569408.0 9240576.0 10944512.0 12648448.0
# CHECK: 835584.0 2555904.0 4259840.0 5996544.0 7700480.0 9437184.0 11141120.0 12845056.0
# CHECK: 851968.0 2605056.0 4358144.0 6094848.0 7864320.0 9633792.0 11337728.0 13107200.0
# CHECK: 868352.0 2654208.0 4423680.0 6225920.0 7995392.0 9764864.0 11599872.0 13369344.0
# CHECK: 884736.0 2703360.0 4521984.0 6324224.0 8159232.0 9961472.0 11796480.0 13631488.0
# CHECK: 901120.0 2752512.0 4587520.0 6455296.0 8290304.0 10158080.0 11993088.0 13828096.0
# CHECK: 913408.0 2801664.0 4685824.0 6553600.0 8454144.0 10354688.0 12189696.0 14090240.0
# CHECK: 929792.0 2850816.0 4751360.0 6684672.0 8585216.0 10485760.0 12451840.0 14352384.0
# CHECK: 946176.0 2899968.0 4849664.0 6782976.0 8716288.0 10682368.0 12648448.0 14614528.0
# CHECK: 962560.0 2949120.0 4915200.0 6914048.0 8912896.0 10878976.0 12845056.0 14811136.0
# CHECK: 978944.0 2998272.0 5013504.0 7012352.0 9043968.0 11075584.0 13041664.0 15073280.0
# CHECK: 995328.0 3047424.0 5079040.0 7143424.0 9175040.0 11206656.0 13303808.0 15335424.0
# CHECK: 1011712.0 3096576.0 5177344.0 7241728.0 9306112.0 11403264.0 13500416.0 15597568.0
alias test_tma_wgmma_64x8x32 = test_tma_wgmma[
    DType.bfloat16,
    DType.bfloat16,
    DType.bfloat16,
    Index(64, 8, 32),
    Index(64, 8, 32),
    Index(64, 8, 16),
]

# CHECK-LABEL: == wgmma_bf16_bf16_f32_64x8x64_transb_64x8x16
# CHECK: 85504.0 214016.0 344064.0 473088.0 602112.0 729088.0 860160.0 987136.0
# CHECK: 214016.0 606208.0 995328.0 1384448.0 1777664.0 2162688.0 2555904.0 2949120.0
# CHECK: 344064.0 995328.0 1646592.0 2310144.0 2949120.0 3604480.0 4259840.0 4915200.0
# CHECK: 473088.0 1384448.0 2310144.0 3211264.0 4128768.0 5046272.0 5963776.0 6881280.0
# CHECK: 602112.0 1777664.0 2949120.0 4128768.0 5308416.0 6488064.0 7667712.0 8847360.0
# CHECK: 729088.0 2162688.0 3604480.0 5046272.0 6488064.0 7929856.0 9371648.0 10813440.0
# CHECK: 860160.0 2555904.0 4259840.0 5963776.0 7667712.0 9371648.0 11075584.0 12779520.0
# CHECK: 987136.0 2949120.0 4915200.0 6881280.0 8847360.0 10813440.0 12779520.0 14745600.0
# CHECK: 1114112.0 3342336.0 5570560.0 7798784.0 10027008.0 12255232.0 14483456.0 16711680.0
# CHECK: 1245184.0 3735552.0 6225920.0 8716288.0 11206656.0 13697024.0 16187392.0 18612224.0
# CHECK: 1376256.0 4128768.0 6881280.0 9633792.0 12386304.0 15138816.0 17825792.0 20578304.0
# CHECK: 1507328.0 4521984.0 7536640.0 10551296.0 13565952.0 16580608.0 19529728.0 22544384.0
# CHECK: 1630208.0 4915200.0 8192000.0 11468800.0 14745600.0 17956864.0 21233664.0 24510464.0
# CHECK: 1761280.0 5308416.0 8847360.0 12386304.0 15925248.0 19398656.0 22937600.0 26476544.0
# CHECK: 1892352.0 5701632.0 9502720.0 13303808.0 17039360.0 20840448.0 24641536.0 28442624.0
# CHECK: 2023424.0 6094848.0 10158080.0 14221312.0 18219008.0 22282240.0 26345472.0 30408704.0
# CHECK: 2146304.0 6488064.0 10813440.0 15138816.0 19398656.0 23724032.0 28049408.0 32374784.0
# CHECK: 2277376.0 6848512.0 11468800.0 16056320.0 20578304.0 25165824.0 29753344.0 34340864.0
# CHECK: 2408448.0 7241728.0 12124160.0 16908288.0 21757952.0 26607616.0 31457280.0 36438016.0
# CHECK: 2539520.0 7634944.0 12779520.0 17825792.0 22937600.0 28049408.0 33161216.0 38273024.0
# CHECK: 2670592.0 8028160.0 13434880.0 18743296.0 24117248.0 29491200.0 34865150.0 40370176.0
# CHECK: 2801664.0 8454144.0 14090240.0 19660800.0 25296896.0 30932992.0 36700160.0 42205184.0
# CHECK: 2916352.0 8847360.0 14745600.0 20578304.0 26476544.0 32374784.0 38273024.0 44302336.0
# CHECK: 3047424.0 9240576.0 15400960.0 21495808.0 27656192.0 33816576.0 40108032.0 46137344.0
# CHECK: 3178496.0 9633792.0 15990784.0 22413312.0 28835840.0 35389440.0 41680896.0 48234496.0
# CHECK: 3309568.0 9961472.0 16646144.0 23330816.0 30015488.0 36700160.0 43515904.0 50069504.0
# CHECK: 3440640.0 10354688.0 17301504.0 24248320.0 31195136.0 38273024.0 45088770.0 52166656.0
# CHECK: 3571712.0 10747904.0 17956864.0 25165824.0 32374784.0 39583744.0 46923776.0 54001664.0
# CHECK: 3702784.0 11141120.0 18612224.0 26083328.0 33554432.0 41156610.0 48496640.0 56098816.0
# CHECK: 3833856.0 11534336.0 19267584.0 27000832.0 34865150.0 42467330.0 50331650.0 57933824.0
# CHECK: 3948544.0 11927552.0 19922944.0 27918336.0 35913730.0 44040190.0 51904512.0 60030976.0
# CHECK: 4079616.0 12320768.0 20578304.0 28835840.0 37224450.0 45350910.0 53739520.0 61865984.0
# CHECK: 4227072.0 12713984.0 21233664.0 29753344.0 38273024.0 46923776.0 55312384.0 63963136.0
# CHECK: 4358144.0 13107200.0 21889024.0 30670848.0 39583744.0 48234496.0 57147392.0 65798144.0
# CHECK: 4489216.0 13500416.0 22544384.0 31588352.0 40632320.0 49807360.0 58720256.0 67633150.0
# CHECK: 4587520.0 13893632.0 23199744.0 32505856.0 41943040.0 51118080.0 60555264.0 69730300.0
# CHECK: 4718592.0 14286848.0 23855104.0 33423360.0 42991616.0 52690944.0 62128130.0 71827460.0
# CHECK: 4849664.0 14680064.0 24510464.0 34340864.0 44302336.0 54001664.0 63700990.0 73400320.0
# CHECK: 4980736.0 15073280.0 25165824.0 35389440.0 45350910.0 55312384.0 65536000.0 75497470.0
# CHECK: 5111808.0 15466496.0 25821184.0 36175870.0 46399490.0 56885250.0 67108864.0 77594624.0
# CHECK: 5242880.0 15859712.0 26476544.0 36962304.0 47710210.0 58195970.0 68681730.0 79691780.0
# CHECK: 5373952.0 16252928.0 27131904.0 38010880.0 48758784.0 59768832.0 70778880.0 81264640.0
# CHECK: 5505024.0 16646144.0 27787264.0 38797310.0 50069504.0 61079552.0 72351740.0 83361790.0
# CHECK: 5636096.0 17039360.0 28442624.0 39845890.0 51118080.0 62652416.0 73924610.0 85458944.0
# CHECK: 5767168.0 17432576.0 29097984.0 40632320.0 52428800.0 63963136.0 75497470.0 87556100.0
# CHECK: 5898240.0 17825792.0 29753344.0 41680896.0 53477376.0 65536000.0 77594624.0 89128960.0
# CHECK: 6029312.0 18219008.0 30408704.0 42467330.0 54788096.0 66846720.0 79167490.0 91226110.0
# CHECK: 6160384.0 18612224.0 31064064.0 43515904.0 55836670.0 68157440.0 80740350.0 93323264.0
# CHECK: 6291456.0 19005440.0 31719424.0 44302336.0 57147392.0 69730300.0 82313220.0 95420420.0
# CHECK: 6422528.0 19398656.0 32374784.0 45350910.0 58195970.0 71303170.0 84410370.0 96993280.0
# CHECK: 6553600.0 19791872.0 33030144.0 46137344.0 59506690.0 72876030.0 85983230.0 99090430.0
# CHECK: 6651904.0 20185088.0 33554432.0 47185920.0 60555264.0 73924610.0 87556100.0 101187584.0
# CHECK: 6782976.0 20578304.0 34340864.0 47972352.0 61865984.0 75497470.0 89128960.0 103284740.0
# CHECK: 6914048.0 20971520.0 34865150.0 49020930.0 62914560.0 77070340.0 91226110.0 104857600.0
# CHECK: 7045120.0 21364736.0 35651584.0 49807360.0 64225280.0 78643200.0 92798980.0 106954750.0
# CHECK: 7176192.0 21757952.0 36175870.0 50855936.0 65273856.0 79691780.0 94371840.0 109051900.0
# CHECK: 7307264.0 22151168.0 36962304.0 51642370.0 66584576.0 81264640.0 95944700.0 111149060.0
# CHECK: 7438336.0 22544384.0 37486590.0 52690944.0 67633150.0 82837504.0 98041860.0 112721920.0
# CHECK: 7569408.0 22937600.0 38273024.0 53477376.0 68681730.0 84410370.0 99614720.0 114819070.0
# CHECK: 7700480.0 23330816.0 38797310.0 54525950.0 70254590.0 85458944.0 101187584.0 116916220.0
# CHECK: 7831552.0 23724032.0 39583744.0 55312384.0 71303170.0 87031810.0 102760450.0 119013380.0
# CHECK: 7962624.0 24117248.0 40108032.0 56360960.0 72351740.0 88604670.0 104857600.0 120586240.0
# CHECK: 8093696.0 24510464.0 40894464.0 57147392.0 73400320.0 90177540.0 106430460.0 122683390.0
# CHECK: 8224768.0 24903680.0 41418752.0 58195970.0 74973184.0 91226110.0 108003330.0 124780540.0
alias test_tma_wgmma_64x8x64 = test_tma_wgmma[
    DType.bfloat16,
    DType.bfloat16,
    DType.bfloat16,
    Index(64, 8, 64),
    Index(64, 8, 64),
    Index(64, 8, 16),
]


def main():
    with DeviceContext() as ctx:
        test_tma_wgmma_64x8x32(ctx)
        test_tma_wgmma_64x8x64(ctx)
