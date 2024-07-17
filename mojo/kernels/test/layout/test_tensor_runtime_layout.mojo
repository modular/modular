# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo %s | FileCheck %s
# COM: TODO(KERN-645)

from layout import LayoutTensor, Layout, RuntimeLayout, RuntimeTuple
from layout.int_tuple import IntTuple, UNKNOWN_VALUE


#  CHECK-LABEL: test_fill_and_print
def test_fill_and_print():
    print("== test_fill_and_print")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](4, 8), RuntimeTuple[layout.stride](8, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)

    tensor.linspace()

    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    tensor.print()

    storage.free()


#  CHECK-LABEL: test_set_and_get_items
def test_set_and_get_items():
    print("== test_set_and_get_items")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](4, 4), RuntimeTuple[layout.stride](4, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)

    for i in range(4):
        for j in range(4):
            tensor[i, j] = i * 4 + j + 2

    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    tensor.print()

    storage.free()


#  CHECK-LABEL: test_tile
def test_tile():
    print("== test_tile")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](4, 4), RuntimeTuple[layout.stride](4, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)
    tensor.linspace()

    # CHECK: ((2, 2):(-1, 1))
    print(tensor.tile[2, 2](0, 0).layout)

    # CHECK: ((2, 2):(4, 1))
    print(tensor.tile[2, 2](0, 0).runtime_layout)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tile_2x2 = tensor.tile[2, 2](tile_i, tile_j)
            tile_2x2.print()

    storage.free()


fn test_tile_and_distribute():
    print("== test_tile_and_distribute")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](8, 8), RuntimeTuple[layout.stride](8, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)
    tensor.linspace()

    # ---tile-data[ 0 , 0 ]----
    # 0.0 1.0 2.0 3.0
    # 8.0 9.0 10.0 11.0
    # 16.0 17.0 18.0 19.0
    # 24.0 25.0 26.0 27.0
    # ----fragments-data[ 0 ]----
    # 0.0 2.0
    # 16.0 18.0
    # ----fragments-data[ 1 ]----
    # 1.0 3.0
    # 17.0 19.0
    # ----fragments-data[ 2 ]----
    # 8.0 10.0
    # 24.0 26.0
    # ----fragments-data[ 3 ]----
    # 9.0 11.0
    # 25.0 27.0
    # ----tile-data[ 0 , 1 ]----
    # 4.0 5.0 6.0 7.0
    # 12.0 13.0 14.0 15.0
    # 20.0 21.0 22.0 23.0
    # 28.0 29.0 30.0 31.0
    # ----fragments-data[ 0 ]----
    # 4.0 6.0
    # 20.0 22.0
    # ----fragments-data[ 1 ]----
    # 5.0 7.0
    # 21.0 23.0
    # ----fragments-data[ 2 ]----
    # 12.0 14.0
    # 28.0 30.0
    # ----fragments-data[ 3 ]----
    # 13.0 15.0
    # 29.0 31.0
    # ----tile-data[ 1 , 0 ]----
    # 32.0 33.0 34.0 35.0
    # 40.0 41.0 42.0 43.0
    # 48.0 49.0 50.0 51.0
    # 56.0 57.0 58.0 59.0
    # ----fragments-data[ 0 ]----
    # 32.0 34.0
    # 48.0 50.0
    # ----fragments-data[ 1 ]----
    # 33.0 35.0
    # 49.0 51.0
    # ----fragments-data[ 2 ]----
    # 40.0 42.0
    # 56.0 58.0
    # ----fragments-data[ 3 ]----
    # 41.0 43.0
    # 57.0 59.0
    # ----tile-data[ 1 , 1 ]----
    # 36.0 37.0 38.0 39.0
    # 44.0 45.0 46.0 47.0
    # 52.0 53.0 54.0 55.0
    # 60.0 61.0 62.0 63.0
    # ----fragments-data[ 0 ]----
    # 36.0 38.0
    # 52.0 54.0
    # ----fragments-data[ 1 ]----
    # 37.0 39.0
    # 53.0 55.0
    # ----fragments-data[ 2 ]----
    # 44.0 46.0
    # 60.0 62.0
    # ----fragments-data[ 3 ]----
    # 45.0 47.0
    # 61.0 63.0
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tile_4x4 = tensor.tile[4, 4](tile_i, tile_j)
            tile_4x4.print()
            for th_i in range(4):
                var tile_2x2 = tile_4x4.distribute[Layout.row_major(2, 2)](th_i)
                print("----fragments-data[", th_i, "]----")
                tile_2x2.print()


# CHECK-LABEL: test_tile_and_vectorize
fn test_tile_and_vectorize():
    print("== test_tile_and_vectorize")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](16, 16), RuntimeTuple[layout.stride](16, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)
    tensor.linspace()

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [0.0, 16.0, 32.0, 48.0, 1.0, 17.0, 33.0, 49.0, 2.0, 18.0, 34.0, 50.0, 3.0, 19.0, 35.0, 51.0] [4.0, 20.0, 36.0, 52.0, 5.0, 21.0, 37.0, 53.0, 6.0, 22.0, 38.0, 54.0, 7.0, 23.0, 39.0, 55.0]
    # CHECK: [64.0, 80.0, 96.0, 112.0, 65.0, 81.0, 97.0, 113.0, 66.0, 82.0, 98.0, 114.0, 67.0, 83.0, 99.0, 115.0] [68.0, 84.0, 100.0, 116.0, 69.0, 85.0, 101.0, 117.0, 70.0, 86.0, 102.0, 118.0, 71.0, 87.0, 103.0, 119.0]
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [8.0, 24.0, 40.0, 56.0, 9.0, 25.0, 41.0, 57.0, 10.0, 26.0, 42.0, 58.0, 11.0, 27.0, 43.0, 59.0] [12.0, 28.0, 44.0, 60.0, 13.0, 29.0, 45.0, 61.0, 14.0, 30.0, 46.0, 62.0, 15.0, 31.0, 47.0, 63.0]
    # CHECK: [72.0, 88.0, 104.0, 120.0, 73.0, 89.0, 105.0, 121.0, 74.0, 90.0, 106.0, 122.0, 75.0, 91.0, 107.0, 123.0] [76.0, 92.0, 108.0, 124.0, 77.0, 93.0, 109.0, 125.0, 78.0, 94.0, 110.0, 126.0, 79.0, 95.0, 111.0, 127.0]
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [128.0, 144.0, 160.0, 176.0, 129.0, 145.0, 161.0, 177.0, 130.0, 146.0, 162.0, 178.0, 131.0, 147.0, 163.0, 179.0] [132.0, 148.0, 164.0, 180.0, 133.0, 149.0, 165.0, 181.0, 134.0, 150.0, 166.0, 182.0, 135.0, 151.0, 167.0, 183.0]
    # CHECK: [192.0, 208.0, 224.0, 240.0, 193.0, 209.0, 225.0, 241.0, 194.0, 210.0, 226.0, 242.0, 195.0, 211.0, 227.0, 243.0] [196.0, 212.0, 228.0, 244.0, 197.0, 213.0, 229.0, 245.0, 198.0, 214.0, 230.0, 246.0, 199.0, 215.0, 231.0, 247.0]
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [136.0, 152.0, 168.0, 184.0, 137.0, 153.0, 169.0, 185.0, 138.0, 154.0, 170.0, 186.0, 139.0, 155.0, 171.0, 187.0] [140.0, 156.0, 172.0, 188.0, 141.0, 157.0, 173.0, 189.0, 142.0, 158.0, 174.0, 190.0, 143.0, 159.0, 175.0, 191.0]
    # CHECK: [200.0, 216.0, 232.0, 248.0, 201.0, 217.0, 233.0, 249.0, 202.0, 218.0, 234.0, 250.0, 203.0, 219.0, 235.0, 251.0] [204.0, 220.0, 236.0, 252.0, 205.0, 221.0, 237.0, 253.0, 206.0, 222.0, 238.0, 254.0, 207.0, 223.0, 239.0, 255.0]

    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tensor_8x8 = tensor.tile[8, 8](tile_i, tile_j)
            tensor_8x8.print()
            var tensor_v_2x2 = tensor_8x8.vectorize[4, 4]()
            print("----vectorized-matrix----")
            tensor_v_2x2.print()

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [0.0, 16.0, 32.0, 48.0] [1.0, 17.0, 33.0, 49.0] [2.0, 18.0, 34.0, 50.0] [3.0, 19.0, 35.0, 51.0] [4.0, 20.0, 36.0, 52.0] [5.0, 21.0, 37.0, 53.0] [6.0, 22.0, 38.0, 54.0] [7.0, 23.0, 39.0, 55.0]
    # CHECK: [64.0, 80.0, 96.0, 112.0] [65.0, 81.0, 97.0, 113.0] [66.0, 82.0, 98.0, 114.0] [67.0, 83.0, 99.0, 115.0] [68.0, 84.0, 100.0, 116.0] [69.0, 85.0, 101.0, 117.0] [70.0, 86.0, 102.0, 118.0] [71.0, 87.0, 103.0, 119.0]
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [8.0, 24.0, 40.0, 56.0] [9.0, 25.0, 41.0, 57.0] [10.0, 26.0, 42.0, 58.0] [11.0, 27.0, 43.0, 59.0] [12.0, 28.0, 44.0, 60.0] [13.0, 29.0, 45.0, 61.0] [14.0, 30.0, 46.0, 62.0] [15.0, 31.0, 47.0, 63.0]
    # CHECK: [72.0, 88.0, 104.0, 120.0] [73.0, 89.0, 105.0, 121.0] [74.0, 90.0, 106.0, 122.0] [75.0, 91.0, 107.0, 123.0] [76.0, 92.0, 108.0, 124.0] [77.0, 93.0, 109.0, 125.0] [78.0, 94.0, 110.0, 126.0] [79.0, 95.0, 111.0, 127.0]
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [128.0, 144.0, 160.0, 176.0] [129.0, 145.0, 161.0, 177.0] [130.0, 146.0, 162.0, 178.0] [131.0, 147.0, 163.0, 179.0] [132.0, 148.0, 164.0, 180.0] [133.0, 149.0, 165.0, 181.0] [134.0, 150.0, 166.0, 182.0] [135.0, 151.0, 167.0, 183.0]
    # CHECK: [192.0, 208.0, 224.0, 240.0] [193.0, 209.0, 225.0, 241.0] [194.0, 210.0, 226.0, 242.0] [195.0, 211.0, 227.0, 243.0] [196.0, 212.0, 228.0, 244.0] [197.0, 213.0, 229.0, 245.0] [198.0, 214.0, 230.0, 246.0] [199.0, 215.0, 231.0, 247.0]
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [136.0, 152.0, 168.0, 184.0] [137.0, 153.0, 169.0, 185.0] [138.0, 154.0, 170.0, 186.0] [139.0, 155.0, 171.0, 187.0] [140.0, 156.0, 172.0, 188.0] [141.0, 157.0, 173.0, 189.0] [142.0, 158.0, 174.0, 190.0] [143.0, 159.0, 175.0, 191.0]
    # CHECK: [200.0, 216.0, 232.0, 248.0] [201.0, 217.0, 233.0, 249.0] [202.0, 218.0, 234.0, 250.0] [203.0, 219.0, 235.0, 251.0] [204.0, 220.0, 236.0, 252.0] [205.0, 221.0, 237.0, 253.0] [206.0, 222.0, 238.0, 254.0] [207.0, 223.0, 239.0, 255.0]
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tensor_8x8 = tensor.tile[8, 8](tile_i, tile_j)
            tensor_8x8.print()
            var tensor_v_2x8 = tensor_8x8.vectorize[4, 1]()
            print("----vectorized-matrix----")
            tensor_v_2x8.print()

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [64.0, 65.0, 66.0, 67.0] [68.0, 69.0, 70.0, 71.0]
    # CHECK: [80.0, 81.0, 82.0, 83.0] [84.0, 85.0, 86.0, 87.0]
    # CHECK: [96.0, 97.0, 98.0, 99.0] [100.0, 101.0, 102.0, 103.0]
    # CHECK: [112.0, 113.0, 114.0, 115.0] [116.0, 117.0, 118.0, 119.0]
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    # CHECK: [72.0, 73.0, 74.0, 75.0] [76.0, 77.0, 78.0, 79.0]
    # CHECK: [88.0, 89.0, 90.0, 91.0] [92.0, 93.0, 94.0, 95.0]
    # CHECK: [104.0, 105.0, 106.0, 107.0] [108.0, 109.0, 110.0, 111.0]
    # CHECK: [120.0, 121.0, 122.0, 123.0] [124.0, 125.0, 126.0, 127.0]
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [128.0, 129.0, 130.0, 131.0] [132.0, 133.0, 134.0, 135.0]
    # CHECK: [144.0, 145.0, 146.0, 147.0] [148.0, 149.0, 150.0, 151.0]
    # CHECK: [160.0, 161.0, 162.0, 163.0] [164.0, 165.0, 166.0, 167.0]
    # CHECK: [176.0, 177.0, 178.0, 179.0] [180.0, 181.0, 182.0, 183.0]
    # CHECK: [192.0, 193.0, 194.0, 195.0] [196.0, 197.0, 198.0, 199.0]
    # CHECK: [208.0, 209.0, 210.0, 211.0] [212.0, 213.0, 214.0, 215.0]
    # CHECK: [224.0, 225.0, 226.0, 227.0] [228.0, 229.0, 230.0, 231.0]
    # CHECK: [240.0, 241.0, 242.0, 243.0] [244.0, 245.0, 246.0, 247.0]
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [136.0, 137.0, 138.0, 139.0] [140.0, 141.0, 142.0, 143.0]
    # CHECK: [152.0, 153.0, 154.0, 155.0] [156.0, 157.0, 158.0, 159.0]
    # CHECK: [168.0, 169.0, 170.0, 171.0] [172.0, 173.0, 174.0, 175.0]
    # CHECK: [184.0, 185.0, 186.0, 187.0] [188.0, 189.0, 190.0, 191.0]
    # CHECK: [200.0, 201.0, 202.0, 203.0] [204.0, 205.0, 206.0, 207.0]
    # CHECK: [216.0, 217.0, 218.0, 219.0] [220.0, 221.0, 222.0, 223.0]
    # CHECK: [232.0, 233.0, 234.0, 235.0] [236.0, 237.0, 238.0, 239.0]
    # CHECK: [248.0, 249.0, 250.0, 251.0] [252.0, 253.0, 254.0, 255.0]
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tensor_8x8 = tensor.tile[8, 8](tile_i, tile_j)
            tensor_8x8.print()
            var tensor_v_8x2 = tensor_8x8.vectorize[1, 4]()
            print("----vectorized-matrix----")
            tensor_v_8x2.print()


# CHECK-LABEL: test_copy_from
fn test_copy_from():
    print("== test_copy_from")
    alias layout = Layout(
        IntTuple(8, 8), IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)
    )

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](8, 8), RuntimeTuple[layout.stride](8, 1)
    )
    var src_tensor = LayoutTensor[DType.float32, layout](
        DTypePointer[DType.float32].alloc(dynamic_layout.size()), dynamic_layout
    )
    src_tensor.linspace()
    var dst_tensor = LayoutTensor[DType.float32, layout](
        DTypePointer[DType.float32].alloc(dynamic_layout.size()), dynamic_layout
    )
    dst_tensor.fill(0)
    dst_tensor.print()
    dst_tensor.copy_from(src_tensor)
    dst_tensor.print()


def main():
    test_fill_and_print()
    test_set_and_get_items()
    test_tile()
    test_tile_and_distribute()
    test_tile_and_vectorize()
    test_copy_from()
