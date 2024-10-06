# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# DISABLED:: %mojo-no-debug %s | FileCheck %s
# RUN: true

from builtin.io import _printf
from gpu import barrier
from gpu.host import DeviceContext
from gpu.host._compile import _get_nvptx_target
from gpu.id import ThreadIdx
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
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.fillers import arange
from layout.int_tuple import to_int
from layout.layout import print_layout


fn wgmma_tf32_tf32_f32_kernel[
    M: Int,
    N: Int,
    K: Int,
    WMMA_M: Int,
    WMMA_N: Int,
    WMMA_K: Int,
    smem_operand_a_layout: Layout,
    smem_operand_b_layout: Layout,
](
    operand_a: LayoutTensor[DType.float32, Layout.row_major(M, K)],
    operand_b: LayoutTensor[DType.float32, Layout.row_major(K, N)],
    result_c: LayoutTensor[DType.float32, Layout.row_major(M, N)],
):
    var smem_operand_a = LayoutTensor[
        DType.float32,
        smem_operand_a_layout,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var smem_operand_b = LayoutTensor[
        DType.float32,
        smem_operand_b_layout,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var c_reg = SIMD[DType.float32, 4](0)

    for k_i in range(K // WMMA_K):
        var operand_a_tile = operand_a.tile[M, WMMA_K](0, k_i)
        var operand_b_tile = operand_b.tile[WMMA_K, N](k_i, 0)
        var operand_a_sm_tile = smem_operand_a.tile[M, WMMA_K](0, k_i)
        var operand_b_sm_tile = smem_operand_b.tile[WMMA_K, N](k_i, 0)

        if ThreadIdx.x() == 0:
            operand_a_sm_tile.copy_from(operand_a_tile)
            operand_b_sm_tile.copy_from(operand_b_tile)

        barrier()

        var mat_a_desc = WGMMADescriptor.create[8, 64](operand_a_sm_tile.ptr)
        var mat_b_desc = WGMMADescriptor.create[1, 8](operand_b_sm_tile.ptr)

        wgmma_fence_aligned()

        c_reg = wgmma_async[
            WMMA_M,
            WMMA_N,
            WMMA_K,
            a_type = DType.tensor_float32,
            b_type = DType.tensor_float32,
        ](mat_a_desc, mat_b_desc, c_reg)
        wgmma_commit_group_sync()
        wgmma_wait_group_sync()
        threadfence()
        wgmma_fence_aligned()

    var warp_id = ThreadIdx.x() // 32
    var lan_id = ThreadIdx.x() % 32
    # Refer to this layout:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N8-D.png
    # Each warp updates a 16x8 tile, and within each tile,
    # every thread updates a 1x2 vector. The resulting distribution layout
    # is as follows:
    var th_local_res = result_c.tile[16, 8](warp_id, 0).vectorize[
        1, 2
    ]().distribute[Layout.row_major(8, 4)](lan_id)
    th_local_res[0][0] = c_reg[0]
    th_local_res[0][1] = c_reg[1]
    th_local_res[1][0] = c_reg[2]
    th_local_res[1][1] = c_reg[3]


# CHECK-LABEL: wgmma_tf32_tf32_f32_64x8x8
# CHECK: 1120.0 1148.0 1176.0 1204.0 1232.0 1260.0 1288.0 1316.0
# CHECK: 2912.0 3004.0 3096.0 3188.0 3280.0 3372.0 3464.0 3556.0
# CHECK: 4704.0 4860.0 5016.0 5172.0 5328.0 5484.0 5640.0 5796.0
# CHECK: 6496.0 6716.0 6936.0 7156.0 7376.0 7596.0 7816.0 8036.0
# CHECK: 8288.0 8572.0 8856.0 9140.0 9424.0 9708.0 9992.0 10276.0
# CHECK: 10080.0 10428.0 10776.0 11124.0 11472.0 11820.0 12168.0 12516.0
# CHECK: 11872.0 12284.0 12696.0 13108.0 13520.0 13932.0 14344.0 14756.0
# CHECK: 13664.0 14140.0 14616.0 15092.0 15568.0 16044.0 16520.0 16996.0
# CHECK: 15456.0 15996.0 16536.0 17076.0 17616.0 18156.0 18696.0 19236.0
# CHECK: 17248.0 17852.0 18456.0 19060.0 19664.0 20268.0 20872.0 21476.0
# CHECK: 19040.0 19708.0 20376.0 21044.0 21712.0 22380.0 23048.0 23716.0
# CHECK: 20832.0 21564.0 22296.0 23028.0 23760.0 24492.0 25224.0 25956.0
# CHECK: 22624.0 23420.0 24216.0 25012.0 25808.0 26604.0 27400.0 28196.0
# CHECK: 24416.0 25276.0 26136.0 26996.0 27856.0 28716.0 29576.0 30436.0
# CHECK: 26208.0 27132.0 28056.0 28980.0 29904.0 30828.0 31752.0 32676.0
# CHECK: 28000.0 28988.0 29976.0 30964.0 31952.0 32940.0 33928.0 34916.0
# CHECK: 29792.0 30844.0 31896.0 32948.0 34000.0 35052.0 36104.0 37156.0
# CHECK: 31584.0 32700.0 33816.0 34932.0 36048.0 37164.0 38280.0 39396.0
# CHECK: 33376.0 34556.0 35736.0 36916.0 38096.0 39276.0 40456.0 41636.0
# CHECK: 35168.0 36412.0 37656.0 38900.0 40144.0 41388.0 42632.0 43876.0
# CHECK: 36960.0 38268.0 39576.0 40884.0 42192.0 43500.0 44808.0 46116.0
# CHECK: 38752.0 40124.0 41496.0 42868.0 44240.0 45612.0 46984.0 48356.0
# CHECK: 40544.0 41980.0 43416.0 44852.0 46288.0 47724.0 49160.0 50596.0
# CHECK: 42336.0 43836.0 45336.0 46836.0 48336.0 49836.0 51336.0 52836.0
# CHECK: 44128.0 45692.0 47256.0 48820.0 50384.0 51948.0 53512.0 55076.0
# CHECK: 45920.0 47548.0 49176.0 50804.0 52432.0 54060.0 55688.0 57316.0
# CHECK: 47712.0 49404.0 51096.0 52788.0 54480.0 56172.0 57864.0 59556.0
# CHECK: 49504.0 51260.0 53016.0 54772.0 56528.0 58284.0 60040.0 61796.0
# CHECK: 51296.0 53116.0 54936.0 56756.0 58576.0 60396.0 62216.0 64036.0
# CHECK: 53088.0 54972.0 56856.0 58740.0 60624.0 62508.0 64392.0 66276.0
# CHECK: 54880.0 56828.0 58776.0 60724.0 62672.0 64620.0 66568.0 68516.0
# CHECK: 56672.0 58684.0 60696.0 62708.0 64720.0 66732.0 68744.0 70756.0
# CHECK: 58464.0 60540.0 62616.0 64692.0 66768.0 68844.0 70920.0 72996.0
# CHECK: 60256.0 62396.0 64536.0 66676.0 68816.0 70956.0 73096.0 75236.0
# CHECK: 62048.0 64252.0 66456.0 68660.0 70864.0 73068.0 75272.0 77476.0
# CHECK: 63840.0 66108.0 68376.0 70644.0 72912.0 75180.0 77448.0 79716.0
# CHECK: 65632.0 67964.0 70296.0 72628.0 74960.0 77292.0 79624.0 81956.0
# CHECK: 67424.0 69820.0 72216.0 74612.0 77008.0 79404.0 81800.0 84196.0
# CHECK: 69216.0 71676.0 74136.0 76596.0 79056.0 81516.0 83976.0 86436.0
# CHECK: 71008.0 73532.0 76056.0 78580.0 81104.0 83628.0 86152.0 88676.0
# CHECK: 72800.0 75388.0 77976.0 80564.0 83152.0 85740.0 88328.0 90916.0
# CHECK: 74592.0 77244.0 79896.0 82548.0 85200.0 87852.0 90504.0 93156.0
# CHECK: 76384.0 79100.0 81816.0 84532.0 87248.0 89964.0 92680.0 95396.0
# CHECK: 78176.0 80956.0 83736.0 86516.0 89296.0 92076.0 94856.0 97636.0
# CHECK: 79968.0 82812.0 85656.0 88500.0 91344.0 94188.0 97032.0 99876.0
# CHECK: 81760.0 84668.0 87576.0 90484.0 93392.0 96300.0 99208.0 102116.0
# CHECK: 83552.0 86524.0 89496.0 92468.0 95440.0 98412.0 101384.0 104356.0
# CHECK: 85344.0 88380.0 91416.0 94452.0 97488.0 100524.0 103560.0 106596.0
# CHECK: 87136.0 90236.0 93336.0 96436.0 99536.0 102636.0 105736.0 108836.0
# CHECK: 88928.0 92092.0 95256.0 98420.0 101584.0 104748.0 107912.0 111076.0
# CHECK: 90720.0 93948.0 97176.0 100404.0 103632.0 106860.0 110088.0 113316.0
# CHECK: 92512.0 95804.0 99096.0 102388.0 105680.0 108972.0 112264.0 115556.0
# CHECK: 94304.0 97660.0 101016.0 104372.0 107728.0 111084.0 114440.0 117796.0
# CHECK: 96096.0 99516.0 102936.0 106356.0 109776.0 113196.0 116616.0 120036.0
# CHECK: 97888.0 101372.0 104856.0 108340.0 111824.0 115308.0 118792.0 122276.0
# CHECK: 99680.0 103228.0 106776.0 110324.0 113872.0 117420.0 120968.0 124516.0
# CHECK: 101472.0 105084.0 108696.0 112308.0 115920.0 119532.0 123144.0 126756.0
# CHECK: 103264.0 106940.0 110616.0 114292.0 117968.0 121644.0 125320.0 128996.0
# CHECK: 105056.0 108796.0 112536.0 116276.0 120016.0 123756.0 127496.0 131236.0
# CHECK: 106848.0 110652.0 114456.0 118260.0 122064.0 125868.0 129672.0 133476.0
# CHECK: 108640.0 112508.0 116376.0 120244.0 124112.0 127980.0 131848.0 135716.0
# CHECK: 110432.0 114364.0 118296.0 122228.0 126160.0 130092.0 134024.0 137956.0
# CHECK: 112224.0 116220.0 120216.0 124212.0 128208.0 132204.0 136200.0 140196.0
# CHECK: 114016.0 118076.0 122136.0 126196.0 130256.0 134316.0 138376.0 142436.0
def wgmma_tf32_tf32_f32_64x8x8(ctx: DeviceContext):
    print("== wgmma_tf32_tf32_f32_64x8x8")
    alias M = 64
    alias N = 8
    alias K = 8

    var lhs = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M, K), gpu_managed_alloc, gpu_free
    ]()
    arange(lhs.tensor)

    var rhs = ManagedLayoutTensor[
        DType.float32, Layout.row_major(K, N), gpu_managed_alloc, gpu_free
    ]()
    arange(rhs.tensor)

    var res = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M, N), gpu_managed_alloc, gpu_free
    ]()

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N8-core-matrices-A.png
    alias a_smem_layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(4, 2)),
        IntTuple(IntTuple(4, 32), IntTuple(1, 256)),
    )

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N8-core-matrices-B.png
    alias b_smem_layout = Layout(
        IntTuple(IntTuple(4, 2), 8), IntTuple(IntTuple(1, 32), 4)
    )
    alias wgmma_tf32_tf32_f32_kernel_fn = wgmma_tf32_tf32_f32_kernel[
        M,
        N,
        K,
        64,
        8,
        8,
        a_smem_layout,
        b_smem_layout,
    ]
    var func = ctx.compile_function[
        wgmma_tf32_tf32_f32_kernel_fn, target = _get_nvptx_target["sm_90"]()
    ]()

    ctx.enqueue_function(
        func,
        lhs.tensor,
        rhs.tensor,
        res.tensor,
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(res.tensor)


# CHECK-LABEL: wgmma_tf32_tf32_f32_64x8x8_inst_64x8x8
# CHECK: 9920.0 10040.0 10160.0 10280.0 10400.0 10520.0 10640.0 10760.0
# CHECK: 25280.0 25656.0 26032.0 26408.0 26784.0 27160.0 27536.0 27912.0
# CHECK: 40640.0 41272.0 41904.0 42536.0 43168.0 43800.0 44432.0 45064.0
# CHECK: 56000.0 56888.0 57776.0 58664.0 59552.0 60440.0 61328.0 62216.0
# CHECK: 71360.0 72504.0 73648.0 74792.0 75936.0 77080.0 78224.0 79368.0
# CHECK: 86720.0 88120.0 89520.0 90920.0 92320.0 93720.0 95120.0 96520.0
# CHECK: 102080.0 103736.0 105392.0 107048.0 108704.0 110360.0 112016.0 113672.0
# CHECK: 117440.0 119352.0 121264.0 123176.0 125088.0 127000.0 128912.0 130824.0
# CHECK: 132800.0 134968.0 137136.0 139304.0 141472.0 143640.0 145808.0 147976.0
# CHECK: 148160.0 150584.0 153008.0 155432.0 157856.0 160280.0 162704.0 165128.0
# CHECK: 163520.0 166200.0 168880.0 171560.0 174240.0 176920.0 179600.0 182280.0
# CHECK: 178880.0 181816.0 184752.0 187688.0 190624.0 193560.0 196496.0 199432.0
# CHECK: 194240.0 197432.0 200624.0 203816.0 207008.0 210200.0 213392.0 216584.0
# CHECK: 209600.0 213048.0 216496.0 219944.0 223392.0 226840.0 230288.0 233736.0
# CHECK: 224960.0 228664.0 232368.0 236072.0 239776.0 243480.0 247184.0 250888.0
# CHECK: 240320.0 244280.0 248240.0 252200.0 256160.0 260120.0 264080.0 268040.0
# CHECK: 255680.0 259896.0 264112.0 268328.0 272544.0 276760.0 280976.0 285192.0
# CHECK: 271040.0 275512.0 279984.0 284456.0 288928.0 293400.0 297872.0 302344.0
# CHECK: 286400.0 291128.0 295856.0 300584.0 305312.0 310040.0 314768.0 319496.0
# CHECK: 301760.0 306744.0 311728.0 316712.0 321696.0 326680.0 331664.0 336648.0
# CHECK: 317120.0 322360.0 327600.0 332840.0 338080.0 343320.0 348560.0 353800.0
# CHECK: 332480.0 337976.0 343472.0 348968.0 354464.0 359960.0 365456.0 370952.0
# CHECK: 347840.0 353592.0 359344.0 365096.0 370848.0 376600.0 382352.0 388104.0
# CHECK: 363200.0 369208.0 375216.0 381224.0 387232.0 393240.0 399248.0 405256.0
# CHECK: 378560.0 384824.0 391088.0 397352.0 403616.0 409880.0 416144.0 422408.0
# CHECK: 393920.0 400440.0 406960.0 413480.0 420000.0 426520.0 433040.0 439560.0
# CHECK: 409280.0 416056.0 422832.0 429608.0 436384.0 443160.0 449936.0 456712.0
# CHECK: 424640.0 431672.0 438704.0 445736.0 452768.0 459800.0 466832.0 473864.0
# CHECK: 440000.0 447288.0 454576.0 461864.0 469152.0 476440.0 483728.0 491016.0
# CHECK: 455360.0 462904.0 470448.0 477992.0 485536.0 493080.0 500624.0 508168.0
# CHECK: 470720.0 478520.0 486320.0 494120.0 501920.0 509720.0 517520.0 525320.0
# CHECK: 486080.0 494136.0 502192.0 510248.0 518304.0 526360.0 534416.0 542472.0
# CHECK: 501440.0 509752.0 518064.0 526376.0 534688.0 543000.0 551312.0 559624.0
# CHECK: 516800.0 525368.0 533936.0 542504.0 551072.0 559640.0 568208.0 576776.0
# CHECK: 532160.0 540984.0 549808.0 558632.0 567456.0 576280.0 585104.0 593928.0
# CHECK: 547520.0 556600.0 565680.0 574760.0 583840.0 592920.0 602000.0 611080.0
# CHECK: 562880.0 572216.0 581552.0 590888.0 600224.0 609560.0 618896.0 628232.0
# CHECK: 578240.0 587832.0 597424.0 607016.0 616608.0 626200.0 635792.0 645384.0
# CHECK: 593600.0 603448.0 613296.0 623144.0 632992.0 642840.0 652688.0 662536.0
# CHECK: 608960.0 619064.0 629168.0 639272.0 649376.0 659480.0 669584.0 679688.0
# CHECK: 624320.0 634680.0 645040.0 655400.0 665760.0 676120.0 686480.0 696840.0
# CHECK: 639680.0 650296.0 660912.0 671528.0 682144.0 692760.0 703376.0 713992.0
# CHECK: 655040.0 665912.0 676784.0 687656.0 698528.0 709400.0 720272.0 731144.0
# CHECK: 670400.0 681528.0 692656.0 703784.0 714912.0 726040.0 737168.0 748296.0
# CHECK: 685760.0 697144.0 708528.0 719912.0 731296.0 742680.0 754064.0 765448.0
# CHECK: 701120.0 712760.0 724400.0 736040.0 747680.0 759320.0 770960.0 782600.0
# CHECK: 716480.0 728376.0 740272.0 752168.0 764064.0 775960.0 787856.0 799752.0
# CHECK: 731840.0 743992.0 756144.0 768296.0 780448.0 792600.0 804752.0 816904.0
# CHECK: 747200.0 759608.0 772016.0 784424.0 796832.0 809240.0 821648.0 834056.0
# CHECK: 762560.0 775224.0 787888.0 800552.0 813216.0 825880.0 838544.0 851208.0
# CHECK: 777920.0 790840.0 803760.0 816680.0 829600.0 842520.0 855440.0 868360.0
# CHECK: 793280.0 806456.0 819632.0 832808.0 845984.0 859160.0 872336.0 885512.0
# CHECK: 808640.0 822072.0 835504.0 848936.0 862368.0 875800.0 889232.0 902664.0
# CHECK: 824000.0 837688.0 851376.0 865064.0 878752.0 892440.0 906128.0 919816.0
# CHECK: 839360.0 853304.0 867248.0 881192.0 895136.0 909080.0 923024.0 936968.0
# CHECK: 854720.0 868920.0 883120.0 897320.0 911520.0 925720.0 939920.0 954120.0
# CHECK: 870080.0 884536.0 898992.0 913448.0 927904.0 942360.0 956816.0 971272.0
# CHECK: 885440.0 900152.0 914864.0 929576.0 944288.0 959000.0 973712.0 988424.0
# CHECK: 900800.0 915768.0 930736.0 945704.0 960672.0 975640.0 990608.0 1005576.0
# CHECK: 916160.0 931384.0 946608.0 961832.0 977056.0 992280.0 1007504.0 1022728.0
# CHECK: 931520.0 947000.0 962480.0 977960.0 993440.0 1008920.0 1024400.0 1039880.0
# CHECK: 946880.0 962616.0 978352.0 994088.0 1009824.0 1025560.0 1041296.0 1057032.0
# CHECK: 962240.0 978232.0 994224.0 1010216.0 1026208.0 1042200.0 1058192.0 1074184.0
# CHECK: 977600.0 993848.0 1010096.0 1026344.0 1042592.0 1058840.0 1075088.0 1091336.0


def wgmma_tf32_tf32_f32_64x8x8_inst_64x8x8(ctx: DeviceContext):
    print("== wgmma_tf32_tf32_f32_64x8x8_inst_64x8x8")
    alias M = 64
    alias N = 8
    alias K = 16

    var lhs = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M, K), gpu_managed_alloc, gpu_free
    ]()
    arange(lhs.tensor)

    var rhs = ManagedLayoutTensor[
        DType.float32, Layout.row_major(K, N), gpu_managed_alloc, gpu_free
    ]()
    arange(rhs.tensor)

    var res = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M, N), gpu_managed_alloc, gpu_free
    ]()

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N8-core-matrices-A.png
    alias a_smem_layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(4, 2)),
        IntTuple(IntTuple(4, 32), IntTuple(1, 256)),
    )

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N8-core-matrices-B.png
    alias b_smem_layout = Layout(
        IntTuple(IntTuple(4, 2), 8), IntTuple(IntTuple(1, 32), 4)
    )
    alias wgmma_tf32_tf32_f32_kernel_fn = wgmma_tf32_tf32_f32_kernel[
        M,
        N,
        K,
        64,
        8,
        8,
        a_smem_layout,
        b_smem_layout,
    ]
    var func = ctx.compile_function[
        wgmma_tf32_tf32_f32_kernel_fn, target = _get_nvptx_target["sm_90"]()
    ]()

    ctx.enqueue_function(
        func,
        lhs.tensor,
        rhs.tensor,
        res.tensor,
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(res.tensor)


def main():
    with DeviceContext() as ctx:
        wgmma_tf32_tf32_f32_64x8x8(ctx)
        wgmma_tf32_tf32_f32_64x8x8_inst_64x8x8(ctx)
