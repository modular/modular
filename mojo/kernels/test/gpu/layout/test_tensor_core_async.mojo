# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.memory import AddressSpace
from gpu.host.memory_v1 import _make_ctx_current
from gpu.host.nvidia_cuda import CUDA
from gpu.id import ThreadIdx
from gpu.sync import barrier


from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout import Layout, LayoutTensor, IntTuple
from layout.fillers import arange
from layout.tensor_core_async import TensorCoreAsync
from layout.layout_tensor import copy_dram_to_sram

from utils import Index


fn tensor_core_async_tf32_tf32_kernel[
    dst_layout: Layout,
    lhs_layout: Layout,
    rhs_layout: Layout,
](
    dst: LayoutTensor[DType.float32, dst_layout],
    lhs: LayoutTensor[DType.float32, lhs_layout],
    rhs: LayoutTensor[DType.float32, rhs_layout],
):
    tensor_core_async = TensorCoreAsync[
        DType.float32, DType.float32, Index(64, 8, 8)
    ]()

    smem_operand_a = tensor_core_async.allocate_lhs()
    smem_operand_b = tensor_core_async.allocate_rhs()

    if ThreadIdx.x == 0:
        smem_operand_a.copy_from(lhs)
        smem_operand_b.copy_from(rhs)

    barrier()

    var c_reg = tensor_core_async.allocate_result(0)

    c_reg = tensor_core_async(smem_operand_a, smem_operand_b, c_reg)
    tensor_core_async.commit_group()
    tensor_core_async.wait_for_all()

    tensor_core_async.store_result(dst, c_reg)


def test_tensor_core_async_tf32_tf32_64x8x8(ctx: DeviceContext):
    print("== test_tensor_core_async")

    alias M = 64
    alias N = 8
    alias K = 8

    lhs = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M, K), gpu_managed_alloc, gpu_free
    ]()
    arange(lhs.tensor)

    rhs = ManagedLayoutTensor[
        DType.float32, Layout.row_major(K, N), gpu_managed_alloc, gpu_free
    ]()
    arange(rhs.tensor)

    dst = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M, N), gpu_managed_alloc, gpu_free
    ]()

    func = ctx.compile_function[
        tensor_core_async_tf32_tf32_kernel[dst.layout, lhs.layout, rhs.layout],
        _target = _get_gpu_target["sm_90"](),
    ]()

    ctx.enqueue_function(
        func,
        dst.tensor,
        lhs.tensor,
        rhs.tensor,
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(dst.tensor)

    _ = lhs^
    _ = rhs^
    _ = dst^
    _ = func^


def main():
    with DeviceContext() as ctx:
        # CHECK-LABEL: test_tensor_core_async
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
        var prev_ctx = _make_ctx_current(CUDA(ctx))
        test_tensor_core_async_tf32_tf32_64x8x8(ctx)
        _ = _make_ctx_current(prev_ctx)
