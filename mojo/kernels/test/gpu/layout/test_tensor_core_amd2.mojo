# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: NVIDIA-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from tensor_core_kernels import (
    mma_load_and_multiply,
    mma_load_and_print_operands_kernel_ldmatrix,
    mma_write_operand_kernel,
    test_load_and_mma_and_multiply_operands,
    test_load_operands_ldmatrix,
    test_write_res_operand,
)

from utils.index import Index, IndexList

from builtin.io import _get_stdout_stream
from sys.ffi import c_char, OpaquePointer
from sys._libc import setvbuf, BufferMode
from memory import UnsafePointer


# CHECK-LABEL: == test_load_and_mma_f32_f16_16x16x16
# CHECK: thread 0 a_vals=[0 4 8 12], b_vals=[0 64 128 192], d_vals=[19841 50561 81281 112001]
# CHECK: thread 1 a_vals=[16 20 24 28], b_vals=[1 65 129 193], d_vals=[19961 50937 81913 112889]
# CHECK: thread 2 a_vals=[32 36 40 44], b_vals=[2 66 130 194], d_vals=[20081 51313 82545 113777]
# CHECK: thread 3 a_vals=[48 52 56 60], b_vals=[3 67 131 195], d_vals=[20201 51689 83177 114665]
# CHECK: thread 4 a_vals=[64 68 72 76], b_vals=[4 68 132 196], d_vals=[20321 52065 83809 115553]
# CHECK: thread 5 a_vals=[80 84 88 92], b_vals=[5 69 133 197], d_vals=[20441 52441 84441 116441]
# CHECK: thread 6 a_vals=[96 100 104 108], b_vals=[6 70 134 198], d_vals=[20561 52817 85073 117329]
# CHECK: thread 7 a_vals=[112 116 120 124], b_vals=[7 71 135 199], d_vals=[20681 53193 85705 118217]
# CHECK: thread 8 a_vals=[128 132 136 140], b_vals=[8 72 136 200], d_vals=[20801 53569 86337 119105]
# CHECK: thread 9 a_vals=[144 148 152 156], b_vals=[9 73 137 201], d_vals=[20921 53945 86969 119993]
# CHECK: thread 10 a_vals=[160 164 168 172], b_vals=[10 74 138 202], d_vals=[21041 54321 87601 120881]
# CHECK: thread 11 a_vals=[176 180 184 188], b_vals=[11 75 139 203], d_vals=[21161 54697 88233 121769]
# CHECK: thread 12 a_vals=[192 196 200 204], b_vals=[12 76 140 204], d_vals=[21281 55073 88865 122657]
# CHECK: thread 13 a_vals=[208 212 216 220], b_vals=[13 77 141 205], d_vals=[21401 55449 89497 123545]
# CHECK: thread 14 a_vals=[224 228 232 236], b_vals=[14 78 142 206], d_vals=[21521 55825 90129 124433]
# CHECK: thread 15 a_vals=[240 244 248 252], b_vals=[15 79 143 207], d_vals=[21641 56201 90761 125321]
# CHECK: thread 16 a_vals=[1 5 9 13], b_vals=[16 80 144 208], d_vals=[142721 173441 204161 234881]
# CHECK: thread 17 a_vals=[17 21 25 29], b_vals=[17 81 145 209], d_vals=[143865 174841 205817 236793]
# CHECK: thread 18 a_vals=[33 37 41 45], b_vals=[18 82 146 210], d_vals=[145009 176241 207473 238705]
# CHECK: thread 19 a_vals=[49 53 57 61], b_vals=[19 83 147 211], d_vals=[146153 177641 209129 240617]
# CHECK: thread 20 a_vals=[65 69 73 77], b_vals=[20 84 148 212], d_vals=[147297 179041 210785 242529]
# CHECK: thread 21 a_vals=[81 85 89 93], b_vals=[21 85 149 213], d_vals=[148441 180441 212441 244441]
# CHECK: thread 22 a_vals=[97 101 105 109], b_vals=[22 86 150 214], d_vals=[149585 181841 214097 246353]
# CHECK: thread 23 a_vals=[113 117 121 125], b_vals=[23 87 151 215], d_vals=[150729 183241 215753 248265]
# CHECK: thread 24 a_vals=[129 133 137 141], b_vals=[24 88 152 216], d_vals=[151873 184641 217409 250177]
# CHECK: thread 25 a_vals=[145 149 153 157], b_vals=[25 89 153 217], d_vals=[153017 186041 219065 252089]
# CHECK: thread 26 a_vals=[161 165 169 173], b_vals=[26 90 154 218], d_vals=[154161 187441 220721 254001]
# CHECK: thread 27 a_vals=[177 181 185 189], b_vals=[27 91 155 219], d_vals=[155305 188841 222377 255913]
# CHECK: thread 28 a_vals=[193 197 201 205], b_vals=[28 92 156 220], d_vals=[156449 190241 224033 257825]
# CHECK: thread 29 a_vals=[209 213 217 221], b_vals=[29 93 157 221], d_vals=[157593 191641 225689 259737]
# CHECK: thread 30 a_vals=[225 229 233 237], b_vals=[30 94 158 222], d_vals=[158737 193041 227345 261649]
# CHECK: thread 31 a_vals=[241 245 249 253], b_vals=[31 95 159 223], d_vals=[159881 194441 229001 263561]
# CHECK: thread 32 a_vals=[2 6 10 14], b_vals=[32 96 160 224], d_vals=[265601 296321 327041 357761]
# CHECK: thread 33 a_vals=[18 22 26 30], b_vals=[33 97 161 225], d_vals=[267769 298745 329721 360697]
# CHECK: thread 34 a_vals=[34 38 42 46], b_vals=[34 98 162 226], d_vals=[269937 301169 332401 363633]
# CHECK: thread 35 a_vals=[50 54 58 62], b_vals=[35 99 163 227], d_vals=[272105 303593 335081 366569]
# CHECK: thread 36 a_vals=[66 70 74 78], b_vals=[36 100 164 228], d_vals=[274273 306017 337761 369505]
# CHECK: thread 37 a_vals=[82 86 90 94], b_vals=[37 101 165 229], d_vals=[276441 308441 340441 372441]
# CHECK: thread 38 a_vals=[98 102 106 110], b_vals=[38 102 166 230], d_vals=[278609 310865 343121 375377]
# CHECK: thread 39 a_vals=[114 118 122 126], b_vals=[39 103 167 231], d_vals=[280777 313289 345801 378313]
# CHECK: thread 40 a_vals=[130 134 138 142], b_vals=[40 104 168 232], d_vals=[282945 315713 348481 381249]
# CHECK: thread 41 a_vals=[146 150 154 158], b_vals=[41 105 169 233], d_vals=[285113 318137 351161 384185]
# CHECK: thread 42 a_vals=[162 166 170 174], b_vals=[42 106 170 234], d_vals=[287281 320561 353841 387121]
# CHECK: thread 43 a_vals=[178 182 186 190], b_vals=[43 107 171 235], d_vals=[289449 322985 356521 390057]
# CHECK: thread 44 a_vals=[194 198 202 206], b_vals=[44 108 172 236], d_vals=[291617 325409 359201 392993]
# CHECK: thread 45 a_vals=[210 214 218 222], b_vals=[45 109 173 237], d_vals=[293785 327833 361881 395929]
# CHECK: thread 46 a_vals=[226 230 234 238], b_vals=[46 110 174 238], d_vals=[295953 330257 364561 398865]
# CHECK: thread 47 a_vals=[242 246 250 254], b_vals=[47 111 175 239], d_vals=[298121 332681 367241 401801]
# CHECK: thread 48 a_vals=[3 7 11 15], b_vals=[48 112 176 240], d_vals=[388481 419201 449921 480641]
# CHECK: thread 49 a_vals=[19 23 27 31], b_vals=[49 113 177 241], d_vals=[391673 422649 453625 484601]
# CHECK: thread 50 a_vals=[35 39 43 47], b_vals=[50 114 178 242], d_vals=[394865 426097 457329 488561]
# CHECK: thread 51 a_vals=[51 55 59 63], b_vals=[51 115 179 243], d_vals=[398057 429545 461033 492521]
# CHECK: thread 52 a_vals=[67 71 75 79], b_vals=[52 116 180 244], d_vals=[401249 432993 464737 496481]
# CHECK: thread 53 a_vals=[83 87 91 95], b_vals=[53 117 181 245], d_vals=[404441 436441 468441 500441]
# CHECK: thread 54 a_vals=[99 103 107 111], b_vals=[54 118 182 246], d_vals=[407633 439889 472145 504401]
# CHECK: thread 55 a_vals=[115 119 123 127], b_vals=[55 119 183 247], d_vals=[410825 443337 475849 508361]
# CHECK: thread 56 a_vals=[131 135 139 143], b_vals=[56 120 184 248], d_vals=[414017 446785 479553 512321]
# CHECK: thread 57 a_vals=[147 151 155 159], b_vals=[57 121 185 249], d_vals=[417209 450233 483257 516281]
# CHECK: thread 58 a_vals=[163 167 171 175], b_vals=[58 122 186 250], d_vals=[420401 453681 486961 520241]
# CHECK: thread 59 a_vals=[179 183 187 191], b_vals=[59 123 187 251], d_vals=[423593 457129 490665 524201]
# CHECK: thread 60 a_vals=[195 199 203 207], b_vals=[60 124 188 252], d_vals=[426785 460577 494369 528161]
# CHECK: thread 61 a_vals=[211 215 219 223], b_vals=[61 125 189 253], d_vals=[429977 464025 498073 532121]
# CHECK: thread 62 a_vals=[227 231 235 239], b_vals=[62 126 190 254], d_vals=[433169 467473 501777 536081]
# CHECK: thread 63 a_vals=[243 247 251 255], b_vals=[63 127 191 255], d_vals=[436361 470921 505481 540041]
def test_load_and_mma_f32_f16_16x16x16(ctx: DeviceContext):
    print("== test_load_and_mma_f32_f16_16x16x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: == test_load_and_mma_f32_bf16_16x16x16
# CHECK: thread 0 a_vals=[0 4 8 12], b_vals=[0 64 128 192], d_vals=[19841 50561 81281 112001]
# CHECK: thread 1 a_vals=[16 20 24 28], b_vals=[1 65 129 193], d_vals=[19961 50937 81913 112889]
# CHECK: thread 2 a_vals=[32 36 40 44], b_vals=[2 66 130 194], d_vals=[20081 51313 82545 113777]
# CHECK: thread 3 a_vals=[48 52 56 60], b_vals=[3 67 131 195], d_vals=[20201 51689 83177 114665]
# CHECK: thread 4 a_vals=[64 68 72 76], b_vals=[4 68 132 196], d_vals=[20321 52065 83809 115553]
# CHECK: thread 5 a_vals=[80 84 88 92], b_vals=[5 69 133 197], d_vals=[20441 52441 84441 116441]
# CHECK: thread 6 a_vals=[96 100 104 108], b_vals=[6 70 134 198], d_vals=[20561 52817 85073 117329]
# CHECK: thread 7 a_vals=[112 116 120 124], b_vals=[7 71 135 199], d_vals=[20681 53193 85705 118217]
# CHECK: thread 8 a_vals=[128 132 136 140], b_vals=[8 72 136 200], d_vals=[20801 53569 86337 119105]
# CHECK: thread 9 a_vals=[144 148 152 156], b_vals=[9 73 137 201], d_vals=[20921 53945 86969 119993]
# CHECK: thread 10 a_vals=[160 164 168 172], b_vals=[10 74 138 202], d_vals=[21041 54321 87601 120881]
# CHECK: thread 11 a_vals=[176 180 184 188], b_vals=[11 75 139 203], d_vals=[21161 54697 88233 121769]
# CHECK: thread 12 a_vals=[192 196 200 204], b_vals=[12 76 140 204], d_vals=[21281 55073 88865 122657]
# CHECK: thread 13 a_vals=[208 212 216 220], b_vals=[13 77 141 205], d_vals=[21401 55449 89497 123545]
# CHECK: thread 14 a_vals=[224 228 232 236], b_vals=[14 78 142 206], d_vals=[21521 55825 90129 124433]
# CHECK: thread 15 a_vals=[240 244 248 252], b_vals=[15 79 143 207], d_vals=[21641 56201 90761 125321]
# CHECK: thread 16 a_vals=[1 5 9 13], b_vals=[16 80 144 208], d_vals=[142721 173441 204161 234881]
# CHECK: thread 17 a_vals=[17 21 25 29], b_vals=[17 81 145 209], d_vals=[143865 174841 205817 236793]
# CHECK: thread 18 a_vals=[33 37 41 45], b_vals=[18 82 146 210], d_vals=[145009 176241 207473 238705]
# CHECK: thread 19 a_vals=[49 53 57 61], b_vals=[19 83 147 211], d_vals=[146153 177641 209129 240617]
# CHECK: thread 20 a_vals=[65 69 73 77], b_vals=[20 84 148 212], d_vals=[147297 179041 210785 242529]
# CHECK: thread 21 a_vals=[81 85 89 93], b_vals=[21 85 149 213], d_vals=[148441 180441 212441 244441]
# CHECK: thread 22 a_vals=[97 101 105 109], b_vals=[22 86 150 214], d_vals=[149585 181841 214097 246353]
# CHECK: thread 23 a_vals=[113 117 121 125], b_vals=[23 87 151 215], d_vals=[150729 183241 215753 248265]
# CHECK: thread 24 a_vals=[129 133 137 141], b_vals=[24 88 152 216], d_vals=[151873 184641 217409 250177]
# CHECK: thread 25 a_vals=[145 149 153 157], b_vals=[25 89 153 217], d_vals=[153017 186041 219065 252089]
# CHECK: thread 26 a_vals=[161 165 169 173], b_vals=[26 90 154 218], d_vals=[154161 187441 220721 254001]
# CHECK: thread 27 a_vals=[177 181 185 189], b_vals=[27 91 155 219], d_vals=[155305 188841 222377 255913]
# CHECK: thread 28 a_vals=[193 197 201 205], b_vals=[28 92 156 220], d_vals=[156449 190241 224033 257825]
# CHECK: thread 29 a_vals=[209 213 217 221], b_vals=[29 93 157 221], d_vals=[157593 191641 225689 259737]
# CHECK: thread 30 a_vals=[225 229 233 237], b_vals=[30 94 158 222], d_vals=[158737 193041 227345 261649]
# CHECK: thread 31 a_vals=[241 245 249 253], b_vals=[31 95 159 223], d_vals=[159881 194441 229001 263561]
# CHECK: thread 32 a_vals=[2 6 10 14], b_vals=[32 96 160 224], d_vals=[265601 296321 327041 357761]
# CHECK: thread 33 a_vals=[18 22 26 30], b_vals=[33 97 161 225], d_vals=[267769 298745 329721 360697]
# CHECK: thread 34 a_vals=[34 38 42 46], b_vals=[34 98 162 226], d_vals=[269937 301169 332401 363633]
# CHECK: thread 35 a_vals=[50 54 58 62], b_vals=[35 99 163 227], d_vals=[272105 303593 335081 366569]
# CHECK: thread 36 a_vals=[66 70 74 78], b_vals=[36 100 164 228], d_vals=[274273 306017 337761 369505]
# CHECK: thread 37 a_vals=[82 86 90 94], b_vals=[37 101 165 229], d_vals=[276441 308441 340441 372441]
# CHECK: thread 38 a_vals=[98 102 106 110], b_vals=[38 102 166 230], d_vals=[278609 310865 343121 375377]
# CHECK: thread 39 a_vals=[114 118 122 126], b_vals=[39 103 167 231], d_vals=[280777 313289 345801 378313]
# CHECK: thread 40 a_vals=[130 134 138 142], b_vals=[40 104 168 232], d_vals=[282945 315713 348481 381249]
# CHECK: thread 41 a_vals=[146 150 154 158], b_vals=[41 105 169 233], d_vals=[285113 318137 351161 384185]
# CHECK: thread 42 a_vals=[162 166 170 174], b_vals=[42 106 170 234], d_vals=[287281 320561 353841 387121]
# CHECK: thread 43 a_vals=[178 182 186 190], b_vals=[43 107 171 235], d_vals=[289449 322985 356521 390057]
# CHECK: thread 44 a_vals=[194 198 202 206], b_vals=[44 108 172 236], d_vals=[291617 325409 359201 392993]
# CHECK: thread 45 a_vals=[210 214 218 222], b_vals=[45 109 173 237], d_vals=[293785 327833 361881 395929]
# CHECK: thread 46 a_vals=[226 230 234 238], b_vals=[46 110 174 238], d_vals=[295953 330257 364561 398865]
# CHECK: thread 47 a_vals=[242 246 250 254], b_vals=[47 111 175 239], d_vals=[298121 332681 367241 401801]
# CHECK: thread 48 a_vals=[3 7 11 15], b_vals=[48 112 176 240], d_vals=[388481 419201 449921 480641]
# CHECK: thread 49 a_vals=[19 23 27 31], b_vals=[49 113 177 241], d_vals=[391673 422649 453625 484601]
# CHECK: thread 50 a_vals=[35 39 43 47], b_vals=[50 114 178 242], d_vals=[394865 426097 457329 488561]
# CHECK: thread 51 a_vals=[51 55 59 63], b_vals=[51 115 179 243], d_vals=[398057 429545 461033 492521]
# CHECK: thread 52 a_vals=[67 71 75 79], b_vals=[52 116 180 244], d_vals=[401249 432993 464737 496481]
# CHECK: thread 53 a_vals=[83 87 91 95], b_vals=[53 117 181 245], d_vals=[404441 436441 468441 500441]
# CHECK: thread 54 a_vals=[99 103 107 111], b_vals=[54 118 182 246], d_vals=[407633 439889 472145 504401]
# CHECK: thread 55 a_vals=[115 119 123 127], b_vals=[55 119 183 247], d_vals=[410825 443337 475849 508361]
# CHECK: thread 56 a_vals=[131 135 139 143], b_vals=[56 120 184 248], d_vals=[414017 446785 479553 512321]
# CHECK: thread 57 a_vals=[147 151 155 159], b_vals=[57 121 185 249], d_vals=[417209 450233 483257 516281]
# CHECK: thread 58 a_vals=[163 167 171 175], b_vals=[58 122 186 250], d_vals=[420401 453681 486961 520241]
# CHECK: thread 59 a_vals=[179 183 187 191], b_vals=[59 123 187 251], d_vals=[423593 457129 490665 524201]
# CHECK: thread 60 a_vals=[195 199 203 207], b_vals=[60 124 188 252], d_vals=[426785 460577 494369 528161]
# CHECK: thread 61 a_vals=[211 215 219 223], b_vals=[61 125 189 253], d_vals=[429977 464025 498073 532121]
# CHECK: thread 62 a_vals=[227 231 235 239], b_vals=[62 126 190 254], d_vals=[433169 467473 501777 536081]
# CHECK: thread 63 a_vals=[243 247 251 255], b_vals=[63 127 191 255], d_vals=[436361 470921 505481 540041]
def test_load_and_mma_f32_bf16_16x16x16(ctx: DeviceContext):
    print("== test_load_and_mma_f32_bf16_16x16x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.bfloat16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: == test_load_and_mma_f32_f32_16x16x4
# CHECK: thread 0 a_vals=[0], b_vals=[0], d_vals=[225 609 993 1377]
# CHECK: thread 1 a_vals=[4], b_vals=[1], d_vals=[231 631 1031 1431]
# CHECK: thread 2 a_vals=[8], b_vals=[2], d_vals=[237 653 1069 1485]
# CHECK: thread 3 a_vals=[12], b_vals=[3], d_vals=[243 675 1107 1539]
# CHECK: thread 4 a_vals=[16], b_vals=[4], d_vals=[249 697 1145 1593]
# CHECK: thread 5 a_vals=[20], b_vals=[5], d_vals=[255 719 1183 1647]
# CHECK: thread 6 a_vals=[24], b_vals=[6], d_vals=[261 741 1221 1701]
# CHECK: thread 7 a_vals=[28], b_vals=[7], d_vals=[267 763 1259 1755]
# CHECK: thread 8 a_vals=[32], b_vals=[8], d_vals=[273 785 1297 1809]
# CHECK: thread 9 a_vals=[36], b_vals=[9], d_vals=[279 807 1335 1863]
# CHECK: thread 10 a_vals=[40], b_vals=[10], d_vals=[285 829 1373 1917]
# CHECK: thread 11 a_vals=[44], b_vals=[11], d_vals=[291 851 1411 1971]
# CHECK: thread 12 a_vals=[48], b_vals=[12], d_vals=[297 873 1449 2025]
# CHECK: thread 13 a_vals=[52], b_vals=[13], d_vals=[303 895 1487 2079]
# CHECK: thread 14 a_vals=[56], b_vals=[14], d_vals=[309 917 1525 2133]
# CHECK: thread 15 a_vals=[60], b_vals=[15], d_vals=[315 939 1563 2187]
# CHECK: thread 16 a_vals=[1], b_vals=[16], d_vals=[1761 2145 2529 2913]
# CHECK: thread 17 a_vals=[5], b_vals=[17], d_vals=[1831 2231 2631 3031]
# CHECK: thread 18 a_vals=[9], b_vals=[18], d_vals=[1901 2317 2733 3149]
# CHECK: thread 19 a_vals=[13], b_vals=[19], d_vals=[1971 2403 2835 3267]
# CHECK: thread 20 a_vals=[17], b_vals=[20], d_vals=[2041 2489 2937 3385]
# CHECK: thread 21 a_vals=[21], b_vals=[21], d_vals=[2111 2575 3039 3503]
# CHECK: thread 22 a_vals=[25], b_vals=[22], d_vals=[2181 2661 3141 3621]
# CHECK: thread 23 a_vals=[29], b_vals=[23], d_vals=[2251 2747 3243 3739]
# CHECK: thread 24 a_vals=[33], b_vals=[24], d_vals=[2321 2833 3345 3857]
# CHECK: thread 25 a_vals=[37], b_vals=[25], d_vals=[2391 2919 3447 3975]
# CHECK: thread 26 a_vals=[41], b_vals=[26], d_vals=[2461 3005 3549 4093]
# CHECK: thread 27 a_vals=[45], b_vals=[27], d_vals=[2531 3091 3651 4211]
# CHECK: thread 28 a_vals=[49], b_vals=[28], d_vals=[2601 3177 3753 4329]
# CHECK: thread 29 a_vals=[53], b_vals=[29], d_vals=[2671 3263 3855 4447]
# CHECK: thread 30 a_vals=[57], b_vals=[30], d_vals=[2741 3349 3957 4565]
# CHECK: thread 31 a_vals=[61], b_vals=[31], d_vals=[2811 3435 4059 4683]
# CHECK: thread 32 a_vals=[2], b_vals=[32], d_vals=[3297 3681 4065 4449]
# CHECK: thread 33 a_vals=[6], b_vals=[33], d_vals=[3431 3831 4231 4631]
# CHECK: thread 34 a_vals=[10], b_vals=[34], d_vals=[3565 3981 4397 4813]
# CHECK: thread 35 a_vals=[14], b_vals=[35], d_vals=[3699 4131 4563 4995]
# CHECK: thread 36 a_vals=[18], b_vals=[36], d_vals=[3833 4281 4729 5177]
# CHECK: thread 37 a_vals=[22], b_vals=[37], d_vals=[3967 4431 4895 5359]
# CHECK: thread 38 a_vals=[26], b_vals=[38], d_vals=[4101 4581 5061 5541]
# CHECK: thread 39 a_vals=[30], b_vals=[39], d_vals=[4235 4731 5227 5723]
# CHECK: thread 40 a_vals=[34], b_vals=[40], d_vals=[4369 4881 5393 5905]
# CHECK: thread 41 a_vals=[38], b_vals=[41], d_vals=[4503 5031 5559 6087]
# CHECK: thread 42 a_vals=[42], b_vals=[42], d_vals=[4637 5181 5725 6269]
# CHECK: thread 43 a_vals=[46], b_vals=[43], d_vals=[4771 5331 5891 6451]
# CHECK: thread 44 a_vals=[50], b_vals=[44], d_vals=[4905 5481 6057 6633]
# CHECK: thread 45 a_vals=[54], b_vals=[45], d_vals=[5039 5631 6223 6815]
# CHECK: thread 46 a_vals=[58], b_vals=[46], d_vals=[5173 5781 6389 6997]
# CHECK: thread 47 a_vals=[62], b_vals=[47], d_vals=[5307 5931 6555 7179]
# CHECK: thread 48 a_vals=[3], b_vals=[48], d_vals=[4833 5217 5601 5985]
# CHECK: thread 49 a_vals=[7], b_vals=[49], d_vals=[5031 5431 5831 6231]
# CHECK: thread 50 a_vals=[11], b_vals=[50], d_vals=[5229 5645 6061 6477]
# CHECK: thread 51 a_vals=[15], b_vals=[51], d_vals=[5427 5859 6291 6723]
# CHECK: thread 52 a_vals=[19], b_vals=[52], d_vals=[5625 6073 6521 6969]
# CHECK: thread 53 a_vals=[23], b_vals=[53], d_vals=[5823 6287 6751 7215]
# CHECK: thread 54 a_vals=[27], b_vals=[54], d_vals=[6021 6501 6981 7461]
# CHECK: thread 55 a_vals=[31], b_vals=[55], d_vals=[6219 6715 7211 7707]
# CHECK: thread 56 a_vals=[35], b_vals=[56], d_vals=[6417 6929 7441 7953]
# CHECK: thread 57 a_vals=[39], b_vals=[57], d_vals=[6615 7143 7671 8199]
# CHECK: thread 58 a_vals=[43], b_vals=[58], d_vals=[6813 7357 7901 8445]
# CHECK: thread 59 a_vals=[47], b_vals=[59], d_vals=[7011 7571 8131 8691]
# CHECK: thread 60 a_vals=[51], b_vals=[60], d_vals=[7209 7785 8361 8937]
# CHECK: thread 61 a_vals=[55], b_vals=[61], d_vals=[7407 7999 8591 9183]
# CHECK: thread 62 a_vals=[59], b_vals=[62], d_vals=[7605 8213 8821 9429]
# CHECK: thread 63 a_vals=[63], b_vals=[63], d_vals=[7803 8427 9051 9675]
def test_load_and_mma_f32_f32_16x16x4(ctx: DeviceContext):
    print("== test_load_and_mma_f32_f32_16x16x4")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float32, Index(16, 16, 4)
    ](ctx)


# CHECK-LABEL: test_write_f32_f32_16x16x4
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
def test_write_f32_f32_16x16x4(ctx: DeviceContext):
    print("== test_write_f32_f32_16x16x4")
    test_write_res_operand[DType.float32, DType.float32, Index(16, 16, 4)](ctx)


# CHECK-LABEL: test_write_f32_f16_16x16x16
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
def test_write_f32_f16_16x16x16(ctx: DeviceContext):
    print("== test_write_f32_f16_16x16x16")
    test_write_res_operand[DType.float32, DType.float16, Index(16, 16, 16)](ctx)


# CHECK-LABEL: test_write_f32_bf16_16x16x16
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
def test_write_f32_bf16_16x16x16(ctx: DeviceContext):
    print("== test_write_f32_bf16_16x16x16")
    test_write_res_operand[DType.float32, DType.bfloat16, Index(16, 16, 16)](
        ctx
    )


# CHECK-LABEL: test_load_f32_f16_16x16x16_ldmatrix
# CHECK: thread 0 a_vals=[0 4 8 12], b_vals=[0 64 128 192]
# CHECK: thread 1 a_vals=[16 20 24 28], b_vals=[1 65 129 193]
# CHECK: thread 2 a_vals=[32 36 40 44], b_vals=[2 66 130 194]
# CHECK: thread 3 a_vals=[48 52 56 60], b_vals=[3 67 131 195]
# CHECK: thread 4 a_vals=[64 68 72 76], b_vals=[4 68 132 196]
# CHECK: thread 5 a_vals=[80 84 88 92], b_vals=[5 69 133 197]
# CHECK: thread 6 a_vals=[96 100 104 108], b_vals=[6 70 134 198]
# CHECK: thread 7 a_vals=[112 116 120 124], b_vals=[7 71 135 199]
# CHECK: thread 8 a_vals=[128 132 136 140], b_vals=[8 72 136 200]
# CHECK: thread 9 a_vals=[144 148 152 156], b_vals=[9 73 137 201]
# CHECK: thread 10 a_vals=[160 164 168 172], b_vals=[10 74 138 202]
# CHECK: thread 11 a_vals=[176 180 184 188], b_vals=[11 75 139 203]
# CHECK: thread 12 a_vals=[192 196 200 204], b_vals=[12 76 140 204]
# CHECK: thread 13 a_vals=[208 212 216 220], b_vals=[13 77 141 205]
# CHECK: thread 14 a_vals=[224 228 232 236], b_vals=[14 78 142 206]
# CHECK: thread 15 a_vals=[240 244 248 252], b_vals=[15 79 143 207]
# CHECK: thread 16 a_vals=[1 5 9 13], b_vals=[16 80 144 208]
# CHECK: thread 17 a_vals=[17 21 25 29], b_vals=[17 81 145 209]
# CHECK: thread 18 a_vals=[33 37 41 45], b_vals=[18 82 146 210]
# CHECK: thread 19 a_vals=[49 53 57 61], b_vals=[19 83 147 211]
# CHECK: thread 20 a_vals=[65 69 73 77], b_vals=[20 84 148 212]
# CHECK: thread 21 a_vals=[81 85 89 93], b_vals=[21 85 149 213]
# CHECK: thread 22 a_vals=[97 101 105 109], b_vals=[22 86 150 214]
# CHECK: thread 23 a_vals=[113 117 121 125], b_vals=[23 87 151 215]
# CHECK: thread 24 a_vals=[129 133 137 141], b_vals=[24 88 152 216]
# CHECK: thread 25 a_vals=[145 149 153 157], b_vals=[25 89 153 217]
# CHECK: thread 26 a_vals=[161 165 169 173], b_vals=[26 90 154 218]
# CHECK: thread 27 a_vals=[177 181 185 189], b_vals=[27 91 155 219]
# CHECK: thread 28 a_vals=[193 197 201 205], b_vals=[28 92 156 220]
# CHECK: thread 29 a_vals=[209 213 217 221], b_vals=[29 93 157 221]
# CHECK: thread 30 a_vals=[225 229 233 237], b_vals=[30 94 158 222]
# CHECK: thread 31 a_vals=[241 245 249 253], b_vals=[31 95 159 223]
# CHECK: thread 32 a_vals=[2 6 10 14], b_vals=[32 96 160 224]
# CHECK: thread 33 a_vals=[18 22 26 30], b_vals=[33 97 161 225]
# CHECK: thread 34 a_vals=[34 38 42 46], b_vals=[34 98 162 226]
# CHECK: thread 35 a_vals=[50 54 58 62], b_vals=[35 99 163 227]
# CHECK: thread 36 a_vals=[66 70 74 78], b_vals=[36 100 164 228]
# CHECK: thread 37 a_vals=[82 86 90 94], b_vals=[37 101 165 229]
# CHECK: thread 38 a_vals=[98 102 106 110], b_vals=[38 102 166 230]
# CHECK: thread 39 a_vals=[114 118 122 126], b_vals=[39 103 167 231]
# CHECK: thread 40 a_vals=[130 134 138 142], b_vals=[40 104 168 232]
# CHECK: thread 41 a_vals=[146 150 154 158], b_vals=[41 105 169 233]
# CHECK: thread 42 a_vals=[162 166 170 174], b_vals=[42 106 170 234]
# CHECK: thread 43 a_vals=[178 182 186 190], b_vals=[43 107 171 235]
# CHECK: thread 44 a_vals=[194 198 202 206], b_vals=[44 108 172 236]
# CHECK: thread 45 a_vals=[210 214 218 222], b_vals=[45 109 173 237]
# CHECK: thread 46 a_vals=[226 230 234 238], b_vals=[46 110 174 238]
# CHECK: thread 47 a_vals=[242 246 250 254], b_vals=[47 111 175 239]
# CHECK: thread 48 a_vals=[3 7 11 15], b_vals=[48 112 176 240]
# CHECK: thread 49 a_vals=[19 23 27 31], b_vals=[49 113 177 241]
# CHECK: thread 50 a_vals=[35 39 43 47], b_vals=[50 114 178 242]
# CHECK: thread 51 a_vals=[51 55 59 63], b_vals=[51 115 179 243]
# CHECK: thread 52 a_vals=[67 71 75 79], b_vals=[52 116 180 244]
# CHECK: thread 53 a_vals=[83 87 91 95], b_vals=[53 117 181 245]
# CHECK: thread 54 a_vals=[99 103 107 111], b_vals=[54 118 182 246]
# CHECK: thread 55 a_vals=[115 119 123 127], b_vals=[55 119 183 247]
# CHECK: thread 56 a_vals=[131 135 139 143], b_vals=[56 120 184 248]
# CHECK: thread 57 a_vals=[147 151 155 159], b_vals=[57 121 185 249]
# CHECK: thread 58 a_vals=[163 167 171 175], b_vals=[58 122 186 250]
# CHECK: thread 59 a_vals=[179 183 187 191], b_vals=[59 123 187 251]
# CHECK: thread 60 a_vals=[195 199 203 207], b_vals=[60 124 188 252]
# CHECK: thread 61 a_vals=[211 215 219 223], b_vals=[61 125 189 253]
# CHECK: thread 62 a_vals=[227 231 235 239], b_vals=[62 126 190 254]
# CHECK: thread 63 a_vals=[243 247 251 255], b_vals=[63 127 191 255]
def test_load_f32_f16_16x16x16_ldmatrix(ctx: DeviceContext):
    print("== test_load_f32_f16_16x16x16_ldmatrix")
    test_load_operands_ldmatrix[
        DType.float32, DType.float16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: test_load_f32_bf16_16x16x16_ldmatrix
# CHECK: thread 0 a_vals=[0 4 8 12], b_vals=[0 64 128 192]
# CHECK: thread 1 a_vals=[16 20 24 28], b_vals=[1 65 129 193]
# CHECK: thread 2 a_vals=[32 36 40 44], b_vals=[2 66 130 194]
# CHECK: thread 3 a_vals=[48 52 56 60], b_vals=[3 67 131 195]
# CHECK: thread 4 a_vals=[64 68 72 76], b_vals=[4 68 132 196]
# CHECK: thread 5 a_vals=[80 84 88 92], b_vals=[5 69 133 197]
# CHECK: thread 6 a_vals=[96 100 104 108], b_vals=[6 70 134 198]
# CHECK: thread 7 a_vals=[112 116 120 124], b_vals=[7 71 135 199]
# CHECK: thread 8 a_vals=[128 132 136 140], b_vals=[8 72 136 200]
# CHECK: thread 9 a_vals=[144 148 152 156], b_vals=[9 73 137 201]
# CHECK: thread 10 a_vals=[160 164 168 172], b_vals=[10 74 138 202]
# CHECK: thread 11 a_vals=[176 180 184 188], b_vals=[11 75 139 203]
# CHECK: thread 12 a_vals=[192 196 200 204], b_vals=[12 76 140 204]
# CHECK: thread 13 a_vals=[208 212 216 220], b_vals=[13 77 141 205]
# CHECK: thread 14 a_vals=[224 228 232 236], b_vals=[14 78 142 206]
# CHECK: thread 15 a_vals=[240 244 248 252], b_vals=[15 79 143 207]
# CHECK: thread 16 a_vals=[1 5 9 13], b_vals=[16 80 144 208]
# CHECK: thread 17 a_vals=[17 21 25 29], b_vals=[17 81 145 209]
# CHECK: thread 18 a_vals=[33 37 41 45], b_vals=[18 82 146 210]
# CHECK: thread 19 a_vals=[49 53 57 61], b_vals=[19 83 147 211]
# CHECK: thread 20 a_vals=[65 69 73 77], b_vals=[20 84 148 212]
# CHECK: thread 21 a_vals=[81 85 89 93], b_vals=[21 85 149 213]
# CHECK: thread 22 a_vals=[97 101 105 109], b_vals=[22 86 150 214]
# CHECK: thread 23 a_vals=[113 117 121 125], b_vals=[23 87 151 215]
# CHECK: thread 24 a_vals=[129 133 137 141], b_vals=[24 88 152 216]
# CHECK: thread 25 a_vals=[145 149 153 157], b_vals=[25 89 153 217]
# CHECK: thread 26 a_vals=[161 165 169 173], b_vals=[26 90 154 218]
# CHECK: thread 27 a_vals=[177 181 185 189], b_vals=[27 91 155 219]
# CHECK: thread 28 a_vals=[193 197 201 205], b_vals=[28 92 156 220]
# CHECK: thread 29 a_vals=[209 213 217 221], b_vals=[29 93 157 221]
# CHECK: thread 30 a_vals=[225 229 233 237], b_vals=[30 94 158 222]
# CHECK: thread 31 a_vals=[241 245 249 253], b_vals=[31 95 159 223]
# CHECK: thread 32 a_vals=[2 6 10 14], b_vals=[32 96 160 224]
# CHECK: thread 33 a_vals=[18 22 26 30], b_vals=[33 97 161 225]
# CHECK: thread 34 a_vals=[34 38 42 46], b_vals=[34 98 162 226]
# CHECK: thread 35 a_vals=[50 54 58 62], b_vals=[35 99 163 227]
# CHECK: thread 36 a_vals=[66 70 74 78], b_vals=[36 100 164 228]
# CHECK: thread 37 a_vals=[82 86 90 94], b_vals=[37 101 165 229]
# CHECK: thread 38 a_vals=[98 102 106 110], b_vals=[38 102 166 230]
# CHECK: thread 39 a_vals=[114 118 122 126], b_vals=[39 103 167 231]
# CHECK: thread 40 a_vals=[130 134 138 142], b_vals=[40 104 168 232]
# CHECK: thread 41 a_vals=[146 150 154 158], b_vals=[41 105 169 233]
# CHECK: thread 42 a_vals=[162 166 170 174], b_vals=[42 106 170 234]
# CHECK: thread 43 a_vals=[178 182 186 190], b_vals=[43 107 171 235]
# CHECK: thread 44 a_vals=[194 198 202 206], b_vals=[44 108 172 236]
# CHECK: thread 45 a_vals=[210 214 218 222], b_vals=[45 109 173 237]
# CHECK: thread 46 a_vals=[226 230 234 238], b_vals=[46 110 174 238]
# CHECK: thread 47 a_vals=[242 246 250 254], b_vals=[47 111 175 239]
# CHECK: thread 48 a_vals=[3 7 11 15], b_vals=[48 112 176 240]
# CHECK: thread 49 a_vals=[19 23 27 31], b_vals=[49 113 177 241]
# CHECK: thread 50 a_vals=[35 39 43 47], b_vals=[50 114 178 242]
# CHECK: thread 51 a_vals=[51 55 59 63], b_vals=[51 115 179 243]
# CHECK: thread 52 a_vals=[67 71 75 79], b_vals=[52 116 180 244]
# CHECK: thread 53 a_vals=[83 87 91 95], b_vals=[53 117 181 245]
# CHECK: thread 54 a_vals=[99 103 107 111], b_vals=[54 118 182 246]
# CHECK: thread 55 a_vals=[115 119 123 127], b_vals=[55 119 183 247]
# CHECK: thread 56 a_vals=[131 135 139 143], b_vals=[56 120 184 248]
# CHECK: thread 57 a_vals=[147 151 155 159], b_vals=[57 121 185 249]
# CHECK: thread 58 a_vals=[163 167 171 175], b_vals=[58 122 186 250]
# CHECK: thread 59 a_vals=[179 183 187 191], b_vals=[59 123 187 251]
# CHECK: thread 60 a_vals=[195 199 203 207], b_vals=[60 124 188 252]
# CHECK: thread 61 a_vals=[211 215 219 223], b_vals=[61 125 189 253]
# CHECK: thread 62 a_vals=[227 231 235 239], b_vals=[62 126 190 254]
# CHECK: thread 63 a_vals=[243 247 251 255], b_vals=[63 127 191 255]
def test_load_f32_bf16_16x16x16_ldmatrix(ctx: DeviceContext):
    print("== test_load_f32_bf16_16x16x16_ldmatrix")
    test_load_operands_ldmatrix[
        DType.float32, DType.bfloat16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: test_load_f32_f32_16x16x4_ldmatrix
# CHECK: thread 0 a_vals=[0], b_vals=[0]
# CHECK: thread 1 a_vals=[4], b_vals=[1]
# CHECK: thread 2 a_vals=[8], b_vals=[2]
# CHECK: thread 3 a_vals=[12], b_vals=[3]
# CHECK: thread 4 a_vals=[16], b_vals=[4]
# CHECK: thread 5 a_vals=[20], b_vals=[5]
# CHECK: thread 6 a_vals=[24], b_vals=[6]
# CHECK: thread 7 a_vals=[28], b_vals=[7]
# CHECK: thread 8 a_vals=[32], b_vals=[8]
# CHECK: thread 9 a_vals=[36], b_vals=[9]
# CHECK: thread 10 a_vals=[40], b_vals=[10]
# CHECK: thread 11 a_vals=[44], b_vals=[11]
# CHECK: thread 12 a_vals=[48], b_vals=[12]
# CHECK: thread 13 a_vals=[52], b_vals=[13]
# CHECK: thread 14 a_vals=[56], b_vals=[14]
# CHECK: thread 15 a_vals=[60], b_vals=[15]
# CHECK: thread 16 a_vals=[1], b_vals=[16]
# CHECK: thread 17 a_vals=[5], b_vals=[17]
# CHECK: thread 18 a_vals=[9], b_vals=[18]
# CHECK: thread 19 a_vals=[13], b_vals=[19]
# CHECK: thread 20 a_vals=[17], b_vals=[20]
# CHECK: thread 21 a_vals=[21], b_vals=[21]
# CHECK: thread 22 a_vals=[25], b_vals=[22]
# CHECK: thread 23 a_vals=[29], b_vals=[23]
# CHECK: thread 24 a_vals=[33], b_vals=[24]
# CHECK: thread 25 a_vals=[37], b_vals=[25]
# CHECK: thread 26 a_vals=[41], b_vals=[26]
# CHECK: thread 27 a_vals=[45], b_vals=[27]
# CHECK: thread 28 a_vals=[49], b_vals=[28]
# CHECK: thread 29 a_vals=[53], b_vals=[29]
# CHECK: thread 30 a_vals=[57], b_vals=[30]
# CHECK: thread 31 a_vals=[61], b_vals=[31]
# CHECK: thread 32 a_vals=[2], b_vals=[32]
# CHECK: thread 33 a_vals=[6], b_vals=[33]
# CHECK: thread 34 a_vals=[10], b_vals=[34]
# CHECK: thread 35 a_vals=[14], b_vals=[35]
# CHECK: thread 36 a_vals=[18], b_vals=[36]
# CHECK: thread 37 a_vals=[22], b_vals=[37]
# CHECK: thread 38 a_vals=[26], b_vals=[38]
# CHECK: thread 39 a_vals=[30], b_vals=[39]
# CHECK: thread 40 a_vals=[34], b_vals=[40]
# CHECK: thread 41 a_vals=[38], b_vals=[41]
# CHECK: thread 42 a_vals=[42], b_vals=[42]
# CHECK: thread 43 a_vals=[46], b_vals=[43]
# CHECK: thread 44 a_vals=[50], b_vals=[44]
# CHECK: thread 45 a_vals=[54], b_vals=[45]
# CHECK: thread 46 a_vals=[58], b_vals=[46]
# CHECK: thread 47 a_vals=[62], b_vals=[47]
# CHECK: thread 48 a_vals=[3], b_vals=[48]
# CHECK: thread 49 a_vals=[7], b_vals=[49]
# CHECK: thread 50 a_vals=[11], b_vals=[50]
# CHECK: thread 51 a_vals=[15], b_vals=[51]
# CHECK: thread 52 a_vals=[19], b_vals=[52]
# CHECK: thread 53 a_vals=[23], b_vals=[53]
# CHECK: thread 54 a_vals=[27], b_vals=[54]
# CHECK: thread 55 a_vals=[31], b_vals=[55]
# CHECK: thread 56 a_vals=[35], b_vals=[56]
# CHECK: thread 57 a_vals=[39], b_vals=[57]
# CHECK: thread 58 a_vals=[43], b_vals=[58]
# CHECK: thread 59 a_vals=[47], b_vals=[59]
# CHECK: thread 60 a_vals=[51], b_vals=[60]
# CHECK: thread 61 a_vals=[55], b_vals=[61]
# CHECK: thread 62 a_vals=[59], b_vals=[62]
# CHECK: thread 63 a_vals=[63], b_vals=[63]
def test_load_f32_f32_16x16x4_ldmatrix(ctx: DeviceContext):
    print("== test_load_f32_f32_16x16x4_ldmatrix")
    test_load_operands_ldmatrix[DType.float32, DType.float32, Index(16, 16, 4)](
        ctx
    )


def main():
    var stdout_stream = _get_stdout_stream()
    if (
        setvbuf(
            stdout_stream, UnsafePointer[c_char](), BufferMode.line_buffered, 0
        )
        != 0
    ):
        raise Error("failed to set line buffering")
    with DeviceContext() as ctx:
        test_load_and_mma_f32_f16_16x16x16(ctx)
        test_load_and_mma_f32_bf16_16x16x16(ctx)
        test_load_and_mma_f32_f32_16x16x4(ctx)
        test_write_f32_f32_16x16x4(ctx)
        test_write_f32_f16_16x16x16(ctx)
        test_write_f32_bf16_16x16x16(ctx)

        # ldmatrix
        test_load_f32_f16_16x16x16_ldmatrix(ctx)
        test_load_f32_bf16_16x16x16_ldmatrix(ctx)
        test_load_f32_f32_16x16x4_ldmatrix(ctx)
