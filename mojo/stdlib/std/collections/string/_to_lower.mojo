# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.collections.string._utf8 import _utf8_first_byte_sequence_length

comptime Diff = SIMD[DType.int16, 4]


@always_inline
def _to_lower(a: UInt8, b: UInt8) -> Diff:
    """Computes byte adjustments for lowercasing a 2-byte UTF-8 character.

    Args:
        a: The leading byte of the UTF-8 sequence.
        b: The continuation byte of the UTF-8 sequence.

    Returns:
        A `Diff` of `(delta_byte0, delta_byte1, extra_byte, output_len)`.
        `delta_byte0` and `delta_byte1` are added to `a` and `b` respectively.
        `extra_byte` is written directly as the third output byte (not a delta).
        `output_len` is the byte length of the result (1 or 2). Returns
        `Diff(0, 0, 0, 2)` if no case conversion applies.
    """
    if a == 195:
        if (b >= 128 and b <= 150) or (b >= 152 and b <= 158):
            return Diff(0, 32, 0, 2)
    elif a == 196:
        if (
            (b >= 128 and b <= 174 and b & 1 == 0)
            or (b >= 178 and b <= 182 and b & 1 == 0)
            or (b >= 185 and b <= 189 and b & 1 == 1)
        ):
            return Diff(0, 1, 0, 2)
        elif b == 176:
            return Diff(-91, -176, 0, 1)
        elif b == 191:
            return Diff(1, -63, 0, 2)
    elif a == 197:
        if (
            (b >= 129 and b <= 135 and b & 1 == 1)
            or (b >= 138 and b <= 182 and b & 1 == 0)
            or (b >= 185 and b <= 189 and b & 1 == 1)
        ):
            return Diff(0, 1, 0, 2)
        elif b == 184:
            return Diff(-2, 7, 0, 2)
    elif a == 198:
        if b == 129:
            return Diff(3, 18, 0, 2)
        elif (
            (b == 130)
            or (b == 132)
            or (b == 135)
            or (b == 139)
            or (b == 145)
            or (b == 152)
            or (b >= 160 and b <= 164 and b & 1 == 0)
            or (b == 167)
            or (b == 172)
            or (b == 175)
            or (b == 179)
            or (b == 181)
            or (b == 184)
            or (b == 188)
        ):
            return Diff(0, 1, 0, 2)
        elif b == 134:
            return Diff(3, 14, 0, 2)
        elif (b == 137) or (b == 138) or (b == 147):
            return Diff(3, 13, 0, 2)
        elif b == 142:
            return Diff(1, 15, 0, 2)
        elif b == 143:
            return Diff(3, 10, 0, 2)
        elif b == 144:
            return Diff(3, 11, 0, 2)
        elif b == 148:
            return Diff(3, 15, 0, 2)
        elif (b == 150) or (b == 156):
            return Diff(3, 19, 0, 2)
        elif b == 151:
            return Diff(3, 17, 0, 2)
        elif b == 157:
            return Diff(3, 21, 0, 2)
        elif b == 159:
            return Diff(3, 22, 0, 2)
        elif (b == 166) or (b == 169) or (b == 174):
            return Diff(4, -38, 0, 2)
        elif (b == 177) or (b == 178):
            return Diff(4, -39, 0, 2)
        elif b == 183:
            return Diff(4, -37, 0, 2)
    elif a == 199:
        if (b == 132) or (b == 135) or (b == 138) or (b == 177):
            return Diff(0, 2, 0, 2)
        elif (
            (b == 133)
            or (b == 136)
            or (b >= 139 and b <= 155 and b & 1 == 1)
            or (b >= 158 and b <= 174 and b & 1 == 0)
            or (b == 178)
            or (b == 180)
            or (b >= 184 and b <= 190 and b & 1 == 0)
        ):
            return Diff(0, 1, 0, 2)
        elif b == 182:
            return Diff(-1, -33, 0, 2)
        elif b == 183:
            return Diff(-1, 8, 0, 2)
    elif a == 200:
        if (
            (b >= 128 and b <= 158 and b & 1 == 0)
            or (b >= 162 and b <= 178 and b & 1 == 0)
            or (b == 187)
        ):
            return Diff(0, 1, 0, 2)
        elif b == 160:
            return Diff(-2, -2, 0, 2)
        elif b == 186:
            return Diff(26, -9, 165, 3)
        elif b == 189:
            return Diff(-2, -35, 0, 2)
        elif b == 190:
            return Diff(26, -13, 166, 3)
    elif a == 201:
        if (b == 129) or (b >= 134 and b <= 142 and b & 1 == 0):
            return Diff(0, 1, 0, 2)
        elif b == 131:
            return Diff(-3, -3, 0, 2)
        elif b == 132:
            return Diff(1, 5, 0, 2)
        elif b == 133:
            return Diff(1, 7, 0, 2)
    elif a == 205:
        if (b == 176) or (b == 178) or (b == 182):
            return Diff(0, 1, 0, 2)
        elif b == 191:
            return Diff(2, -12, 0, 2)
    elif a == 206:
        if b == 134:
            return Diff(0, 38, 0, 2)
        elif b >= 136 and b <= 138:
            return Diff(0, 37, 0, 2)
        elif b == 140:
            return Diff(1, 0, 0, 2)
        elif (b == 142) or (b == 143):
            return Diff(1, -1, 0, 2)
        elif b >= 145 and b <= 159:
            return Diff(0, 32, 0, 2)
        elif (b == 160) or (b == 161) or (b >= 163 and b <= 171):
            return Diff(1, -32, 0, 2)
    elif a == 207:
        if b == 143:
            return Diff(0, 8, 0, 2)
        elif (b >= 152 and b <= 174 and b & 1 == 0) or (b == 183) or (b == 186):
            return Diff(0, 1, 0, 2)
        elif b == 180:
            return Diff(-1, 4, 0, 2)
        elif b == 185:
            return Diff(0, -7, 0, 2)
        elif b >= 189 and b <= 191:
            return Diff(-2, -2, 0, 2)
    elif a == 208:
        if b >= 128 and b <= 143:
            return Diff(1, 16, 0, 2)
        elif b >= 144 and b <= 159:
            return Diff(0, 32, 0, 2)
        elif b >= 160 and b <= 175:
            return Diff(1, -32, 0, 2)
    elif a == 209:
        if b >= 160 and b <= 190 and b & 1 == 0:
            return Diff(0, 1, 0, 2)
    elif a == 210:
        if (b == 128) or (b >= 138 and b <= 190 and b & 1 == 0):
            return Diff(0, 1, 0, 2)
    elif a == 211:
        if b == 128:
            return Diff(0, 15, 0, 2)
        elif (b >= 129 and b <= 141 and b & 1 == 1) or (
            b >= 144 and b <= 190 and b & 1 == 0
        ):
            return Diff(0, 1, 0, 2)
    elif a == 212:
        if b >= 128 and b <= 174 and b & 1 == 0:
            return Diff(0, 1, 0, 2)
        elif b >= 177 and b <= 191:
            return Diff(1, -16, 0, 2)
    elif a == 213:
        if b >= 128 and b <= 143:
            return Diff(0, 48, 0, 2)
        elif b >= 144 and b <= 150:
            return Diff(1, -16, 0, 2)
    return Diff(0, 0, 0, 2)


@always_inline
def _to_lower(a: UInt8, b: UInt8, c: UInt8) -> Diff:
    """Computes byte adjustments for lowercasing a 3-byte UTF-8 character.

    Args:
        a: The leading byte of the UTF-8 sequence.
        b: The first continuation byte of the UTF-8 sequence.
        c: The second continuation byte of the UTF-8 sequence.

    Returns:
        A `Diff` of `(delta_byte0, delta_byte1, delta_byte2, output_len)`.
        Each delta is added to the corresponding input byte to produce the
        output. `output_len` is the byte length of the result (1, 2, or 3).
        Returns `Diff(0, 0, 0, 3)` if no case conversion applies.
    """
    if a == 225:
        if b == 130 and c >= 160 and c <= 191:
            return Diff(1, 50, -32, 3)
        elif b == 131 and ((c >= 128 and c <= 133) or (c == 135) or (c == 141)):
            return Diff(1, 49, 32, 3)
        elif b == 142:
            if c >= 160 and c <= 175:
                return Diff(9, 31, 16, 3)
            elif c >= 176 and c <= 191:
                return Diff(9, 32, -48, 3)
        elif b == 143:
            if c >= 128 and c <= 175:
                return Diff(9, 31, 16, 3)
            elif c >= 176 and c <= 181:
                return Diff(0, 0, 8, 3)
        elif b == 178 and ((c >= 144 and c <= 186) or (c >= 189 and c <= 191)):
            return Diff(0, -47, 0, 3)
        elif b == 184 and c >= 128 and c <= 190 and c & 1 == 0:
            return Diff(0, 0, 1, 3)
        elif b == 185 and c >= 128 and c <= 190 and c & 1 == 0:
            return Diff(0, 0, 1, 3)
        elif b == 186:
            if c >= 128 and c <= 148 and c & 1 == 0:
                return Diff(0, 0, 1, 3)
            elif c == 158:
                return Diff(-30, -27, -158, 2)
            elif c >= 160 and c <= 190 and c & 1 == 0:
                return Diff(0, 0, 1, 3)
        elif b == 187 and c >= 128 and c <= 190 and c & 1 == 0:
            return Diff(0, 0, 1, 3)
        elif b == 188 and (
            (c >= 136 and c <= 143)
            or (c >= 152 and c <= 157)
            or (c >= 168 and c <= 175)
            or (c >= 184 and c <= 191)
        ):
            return Diff(0, 0, -8, 3)
        elif b == 189 and (
            (c >= 136 and c <= 141)
            or (c >= 153 and c <= 159 and c & 1 == 1)
            or (c >= 168 and c <= 175)
        ):
            return Diff(0, 0, -8, 3)
        elif b == 190:
            if c >= 136 and c <= 143:
                return Diff(0, 0, -8, 3)
            elif c >= 152 and c <= 159:
                return Diff(0, 0, -8, 3)
            elif c >= 168 and c <= 175:
                return Diff(0, 0, -8, 3)
            elif c == 184:
                return Diff(0, 0, -8, 3)
            elif c == 185:
                return Diff(0, 0, -8, 3)
            elif c == 186:
                return Diff(0, -1, -10, 3)
            elif c == 187:
                return Diff(0, -1, -10, 3)
            elif c == 188:
                return Diff(0, 0, -9, 3)
        elif b == 191:
            if c >= 136 and c <= 139:
                return Diff(0, -2, 42, 3)
            elif c == 140:
                return Diff(0, 0, -9, 3)
            elif c == 152:
                return Diff(0, 0, -8, 3)
            elif c == 153:
                return Diff(0, 0, -8, 3)
            elif c == 154:
                return Diff(0, -2, 28, 3)
            elif c == 155:
                return Diff(0, -2, 28, 3)
            elif c == 168:
                return Diff(0, 0, -8, 3)
            elif c == 169:
                return Diff(0, 0, -8, 3)
            elif c == 170:
                return Diff(0, -2, 16, 3)
            elif c == 171:
                return Diff(0, -2, 16, 3)
            elif c == 172:
                return Diff(0, 0, -7, 3)
            elif c == 184:
                return Diff(0, -2, 0, 3)
            elif c == 185:
                return Diff(0, -2, 0, 3)
            elif c == 186:
                return Diff(0, -2, 2, 3)
            elif c == 187:
                return Diff(0, -2, 2, 3)
            elif c == 188:
                return Diff(0, 0, -9, 3)
    elif a == 226:
        if b == 132:
            if c == 166:
                return Diff(-19, 5, -166, 2)
            elif c == 170:
                return Diff(-119, -132, -170, 1)
            elif c == 171:
                return Diff(-31, 33, -171, 2)
            elif c == 178:
                return Diff(0, 1, -36, 3)
        elif b == 133 and c >= 160 and c <= 175:
            return Diff(0, 0, 16, 3)
        elif b == 134 and c == 131:
            return Diff(0, 0, 1, 3)
        elif b == 146 and c >= 182 and c <= 191:
            return Diff(0, 1, -38, 3)
        elif b == 147 and c >= 128 and c <= 143:
            return Diff(0, 0, 26, 3)
        elif b == 176:
            if c >= 128 and c <= 143:
                return Diff(0, 0, 48, 3)
            elif c >= 144 and c <= 175:
                return Diff(0, 1, -16, 3)
        elif b == 177:
            if c == 160:
                return Diff(0, 0, 1, 3)
            elif c == 162:
                return Diff(-25, -6, -162, 2)
            elif c == 163:
                return Diff(-1, 4, 26, 3)
            elif c == 164:
                return Diff(-25, 12, -164, 2)
            elif c >= 167 and c <= 171 and c & 1 == 1:
                return Diff(0, 0, 1, 3)
            elif c == 173:
                return Diff(-25, -32, -173, 2)
            elif c == 174:
                return Diff(-25, 0, -174, 2)
            elif c == 175:
                return Diff(-25, -33, -175, 2)
            elif c == 176:
                return Diff(-25, -31, -176, 2)
            elif c == 178:
                return Diff(0, 0, 1, 3)
            elif c == 181:
                return Diff(0, 0, 1, 3)
            elif c == 190:
                return Diff(-26, 14, -190, 2)
            elif c == 191:
                return Diff(-25, -49, -191, 2)
        elif b == 178 and c >= 128 and c <= 190 and c & 1 == 0:
            return Diff(0, 0, 1, 3)
        elif b == 179 and (
            (c >= 128 and c <= 162 and c & 1 == 0)
            or (c == 171)
            or (c == 173)
            or (c == 178)
        ):
            return Diff(0, 0, 1, 3)
    elif a == 234:
        if b == 153 and c >= 128 and c <= 172 and c & 1 == 0:
            return Diff(0, 0, 1, 3)
        elif b == 154 and c >= 128 and c <= 154 and c & 1 == 0:
            return Diff(0, 0, 1, 3)
        elif b == 156 and (
            (c >= 162 and c <= 174 and c & 1 == 0)
            or (c >= 178 and c <= 190 and c & 1 == 0)
        ):
            return Diff(0, 0, 1, 3)
        elif b == 157:
            if c >= 128 and c <= 174 and c & 1 == 0:
                return Diff(0, 0, 1, 3)
            elif c == 185:
                return Diff(0, 0, 1, 3)
            elif c == 187:
                return Diff(0, 0, 1, 3)
            elif c == 189:
                return Diff(-9, 24, -4, 3)
            elif c == 190:
                return Diff(0, 0, 1, 3)
        elif b == 158:
            if c >= 128 and c <= 134 and c & 1 == 0:
                return Diff(0, 0, 1, 3)
            elif c == 139:
                return Diff(0, 0, 1, 3)
            elif c == 141:
                return Diff(-33, 7, -141, 2)
            elif c == 144:
                return Diff(0, 0, 1, 3)
            elif c == 146:
                return Diff(0, 0, 1, 3)
            elif c >= 150 and c <= 168 and c & 1 == 0:
                return Diff(0, 0, 1, 3)
            elif c == 170:
                return Diff(-33, 8, -170, 2)
            elif c == 171:
                return Diff(-33, -2, -171, 2)
            elif c == 172:
                return Diff(-33, 3, -172, 2)
            elif c == 173:
                return Diff(-33, 14, -173, 2)
            elif c == 174:
                return Diff(-33, 12, -174, 2)
            elif c == 176:
                return Diff(-32, 0, -176, 2)
            elif c == 177:
                return Diff(-32, -23, -177, 2)
            elif c == 178:
                return Diff(-32, -1, -178, 2)
            elif c == 179:
                return Diff(0, 15, -32, 3)
            elif c >= 180 and c <= 190 and c & 1 == 0:
                return Diff(0, 0, 1, 3)
        elif b == 159:
            if c == 128:
                return Diff(0, 0, 1, 3)
            elif c == 130:
                return Diff(0, 0, 1, 3)
            elif c == 132:
                return Diff(0, -1, 16, 3)
            elif c == 133:
                return Diff(-32, -29, -133, 2)
            elif c == 134:
                return Diff(-9, 23, 8, 3)
            elif c == 135:
                return Diff(0, 0, 1, 3)
            elif c == 137:
                return Diff(0, 0, 1, 3)
            elif c == 144:
                return Diff(0, 0, 1, 3)
            elif c == 150:
                return Diff(0, 0, 1, 3)
            elif c == 152:
                return Diff(0, 0, 1, 3)
            elif c == 181:
                return Diff(0, 0, 1, 3)
    elif a == 239:
        if b == 188 and c >= 161 and c <= 186:
            return Diff(0, 1, -32, 3)
    return Diff(0, 0, 0, 3)


@always_inline
def _to_lower(a: UInt8, b: UInt8, c: UInt8, d: UInt8) -> Diff:
    """Computes byte adjustments for lowercasing a 4-byte UTF-8 character.

    4-byte characters always produce a 4-byte result, so no output length is
    encoded in the return value.

    Args:
        a: The leading byte of the UTF-8 sequence.
        b: The first continuation byte of the UTF-8 sequence.
        c: The second continuation byte of the UTF-8 sequence.
        d: The third continuation byte of the UTF-8 sequence.

    Returns:
        A `Diff` of `(delta_byte0, delta_byte1, delta_byte2, delta_byte3)`.
        Each delta is added to the corresponding input byte to produce the
        output. Returns `Diff(0, 0, 0, 0)` if no case conversion applies.
    """
    if a == 240:
        if b == 144:
            if c == 144:
                if d >= 128 and d <= 151:
                    return Diff(0, 0, 0, 40)
                elif d >= 152 and d <= 167:
                    return Diff(0, 0, 1, -24)
            elif c == 146 and d >= 176 and d <= 191:
                return Diff(0, 0, 1, -24)
            elif c == 147 and d >= 128 and d <= 147:
                return Diff(0, 0, 0, 40)
            elif c == 149 and (
                (d >= 176 and d <= 186) or (d >= 188 and d <= 191)
            ):
                return Diff(0, 0, 1, -25)
            elif c == 150 and (
                (d >= 128 and d <= 138)
                or (d >= 140 and d <= 146)
                or (d == 148)
                or (d == 149)
            ):
                return Diff(0, 0, 0, 39)
            elif c == 178 and d >= 128 and d <= 178:
                return Diff(0, 0, 1, 0)
        elif b == 145 and c == 162 and d >= 160 and d <= 191:
            return Diff(0, 0, 1, -32)
        elif b == 150 and c == 185 and d >= 128 and d <= 159:
            return Diff(0, 0, 0, 32)
        elif b == 158 and c == 164:
            if d >= 128 and d <= 157:
                return Diff(0, 0, 0, 34)
            elif d >= 158 and d <= 161:
                return Diff(0, 0, 1, -30)
    return Diff(0, 0, 0, 0)


def lower_utf8[ascii_optimized: Bool = True](s: String) -> String:
    """Converts a UTF-8 encoded string to lowercase.

    Uses a byte-level decision tree to compute lowercase transformations for
    multi-byte Unicode characters. When `ascii_optimized` is `True`, processes
    8 ASCII bytes at a time via SIMD before falling back to per-character
    handling.

    Parameters:
        ascii_optimized: If `True`, enables a SIMD fast path that processes
            8 ASCII bytes per iteration before falling back to per-character
            Unicode handling.

    Args:
        s: The input string to convert.

    Returns:
        A new `String` with all characters converted to lowercase.
    """
    comptime SIMD_WIDTH = 8
    # The width is set to 8 bytes based on performance benchmark for pure ASCII examples
    # and with examples where ASCII is mixed with non ASCII characters.
    # It is imaginable to promote SIMD_WIDTH from constant to be a function parameter.
    var byte_len = s.byte_length()
    var result = String(capacity=byte_len // 2 * 3 + 1)
    var offset = 0
    var p = s.unsafe_ptr()
    var buf = result.unsafe_ptr_mut()
    var count = 0
    while offset < byte_len:
        comptime if ascii_optimized:
            if offset + SIMD_WIDTH <= byte_len:
                var vec = (p + offset).load[SIMD_WIDTH]()
                if vec.le(127).reduce_and():
                    var is_upper = vec.ge(65) & vec.le(90)
                    var delta = is_upper.select(
                        SIMD[DType.uint8, SIMD_WIDTH](32),
                        SIMD[DType.uint8, SIMD_WIDTH](0),
                    )
                    buf.store(vec ^ delta)
                    buf += SIMD_WIDTH
                    count += SIMD_WIDTH
                    offset += SIMD_WIDTH
                    continue
        var b0 = p[offset]
        var char_length = _utf8_first_byte_sequence_length(b0)
        if char_length == 1:
            buf[0] = buf[0] = b0 ^ (Byte((b0 >= 65) & (b0 <= 90)) << 5)
            buf += 1
            count += 1
        elif char_length == 2:
            var b1 = p[offset + 1]
            var diff = _to_lower(b0, b1)
            var out_len = Int(diff[3])
            buf[0] = Byte(Int16(b0) + diff[0])
            buf[1] = Byte(Int16(b1) + diff[1])
            buf[2] = Byte(diff[2])
            buf += out_len
            count += out_len
        elif char_length == 3:
            var b1 = p[offset + 1]
            var b2 = p[offset + 2]
            var diff = _to_lower(b0, b1, b2)
            var out_len = Int(diff[3])
            buf[0] = Byte(Int16(b0) + diff[0])
            buf[1] = Byte(Int16(b1) + diff[1])
            buf[2] = Byte(Int16(b2) + diff[2])
            buf += out_len
            count += out_len
        elif char_length == 4:
            var b1 = p[offset + 1]
            var b2 = p[offset + 2]
            var b3 = p[offset + 3]
            var diff = _to_lower(b0, b1, b2, b3)
            buf[0] = Byte(Int16(b0) + diff[0])
            buf[1] = Byte(Int16(b1) + diff[1])
            buf[2] = Byte(Int16(b2) + diff[2])
            buf[3] = Byte(Int16(b3) + diff[3])
            buf += 4
            count += 4
        offset += char_length
    buf[0] = 0
    result.resize(unsafe_uninit_length=count)
    return result
