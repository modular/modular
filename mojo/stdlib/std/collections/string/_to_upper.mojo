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

comptime Diff = SIMD[DType.int16, 8]


@always_inline
def _to_upper(a: UInt8, b: UInt8) -> Diff:
    """Computes byte adjustments for uppercasing a 2-byte UTF-8 character.

    Args:
        a: The leading byte of the UTF-8 sequence.
        b: The continuation byte of the UTF-8 sequence.

    Returns:
        A `Diff` of `(delta_byte0, delta_byte1, extra_byte2, extra_byte3,
        extra_byte4, extra_byte5, output_len, 0)`. `delta_byte0` and
        `delta_byte1` are added to `a` and `b` respectively to form the first
        two output bytes. `extra_byte2` through `extra_byte5` are written
        directly as subsequent output bytes (not deltas). `output_len` is the
        byte length of the result (1 to 6). Returns
        `Diff(0, 0, 0, 0, 0, 0, 2, 0)` if no case conversion applies.
    """
    if a == 194:
        if b == 181:
            return Diff(12, -25, 0, 0, 0, 0, 2, 0)
    elif a == 195:
        if b == 159:
            return Diff(-112, -76, 0, 0, 0, 0, 2, 0)
        elif (b >= 160 and b <= 182) or (b >= 184 and b <= 190):
            return Diff(0, -32, 0, 0, 0, 0, 2, 0)
        elif b == 191:
            return Diff(2, -7, 0, 0, 0, 0, 2, 0)
    elif a == 196:
        if (
            (b >= 129 and b <= 175 and b & 1 == 1)
            or (b >= 179 and b <= 183 and b & 1 == 1)
            or (b >= 186 and b <= 190 and b & 1 == 0)
        ):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 177:
            return Diff(-123, 0, 0, 0, 0, 0, 1, 0)
    elif a == 197:
        if b == 128:
            return Diff(-1, 63, 0, 0, 0, 0, 2, 0)
        elif (
            (b >= 130 and b <= 136 and b & 1 == 0)
            or (b >= 139 and b <= 183 and b & 1 == 1)
            or (b >= 186 and b <= 190 and b & 1 == 0)
        ):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 137:
            return Diff(5, 51, 78, 0, 0, 0, 3, 0)
        elif b == 191:
            return Diff(-114, 0, 0, 0, 0, 0, 1, 0)
    elif a == 198:
        if b == 128:
            return Diff(3, 3, 0, 0, 0, 0, 2, 0)
        elif (
            (b == 131)
            or (b == 133)
            or (b == 136)
            or (b == 140)
            or (b == 146)
            or (b == 153)
            or (b >= 161 and b <= 165 and b & 1 == 1)
            or (b == 168)
            or (b == 173)
            or (b == 176)
            or (b == 180)
            or (b == 182)
            or (b == 185)
            or (b == 189)
        ):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 149:
            return Diff(1, 33, 0, 0, 0, 0, 2, 0)
        elif b == 154:
            return Diff(2, 35, 0, 0, 0, 0, 2, 0)
        elif b == 158:
            return Diff(2, 2, 0, 0, 0, 0, 2, 0)
        elif b == 191:
            return Diff(1, -8, 0, 0, 0, 0, 2, 0)
    elif a == 199:
        if (b == 132) or (b == 135) or (b == 138) or (b == 177):
            return Diff(0, 1, 0, 0, 0, 0, 2, 0)
        elif (
            (b == 134)
            or (b == 137)
            or (b >= 140 and b <= 156 and b & 1 == 0)
            or (b >= 159 and b <= 175 and b & 1 == 1)
            or (b == 179)
            or (b == 181)
            or (b >= 185 and b <= 191 and b & 1 == 1)
        ):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 157:
            return Diff(-1, -15, 0, 0, 0, 0, 2, 0)
        elif b == 176:
            return Diff(-125, 28, 140, 0, 0, 0, 3, 0)
    elif a == 200:
        if (
            (b >= 129 and b <= 159 and b & 1 == 1)
            or (b >= 163 and b <= 179 and b & 1 == 1)
            or (b == 188)
        ):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 191:
            return Diff(26, -14, 190, 0, 0, 0, 3, 0)
    elif a == 201:
        if b == 128:
            return Diff(25, 49, 191, 0, 0, 0, 3, 0)
        elif (b == 130) or (b >= 135 and b <= 143 and b & 1 == 1):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 144:
            return Diff(25, 33, 175, 0, 0, 0, 3, 0)
        elif b == 145:
            return Diff(25, 32, 173, 0, 0, 0, 3, 0)
        elif b == 146:
            return Diff(25, 31, 176, 0, 0, 0, 3, 0)
        elif b == 147:
            return Diff(-3, -18, 0, 0, 0, 0, 2, 0)
        elif b == 148:
            return Diff(-3, -14, 0, 0, 0, 0, 2, 0)
        elif (b == 150) or (b == 151) or (b == 160):
            return Diff(-3, -13, 0, 0, 0, 0, 2, 0)
        elif b == 153:
            return Diff(-3, -10, 0, 0, 0, 0, 2, 0)
        elif b == 155:
            return Diff(-3, -11, 0, 0, 0, 0, 2, 0)
        elif b == 156:
            return Diff(33, 2, 171, 0, 0, 0, 3, 0)
        elif b == 161:
            return Diff(33, -3, 172, 0, 0, 0, 3, 0)
        elif b == 163:
            return Diff(-3, -15, 0, 0, 0, 0, 2, 0)
        elif b == 165:
            return Diff(33, -7, 141, 0, 0, 0, 3, 0)
        elif b == 166:
            return Diff(33, -8, 170, 0, 0, 0, 3, 0)
        elif b == 168:
            return Diff(-3, -17, 0, 0, 0, 0, 2, 0)
        elif (b == 169) or (b == 175):
            return Diff(-3, -19, 0, 0, 0, 0, 2, 0)
        elif b == 170:
            return Diff(33, -12, 174, 0, 0, 0, 3, 0)
        elif b == 171:
            return Diff(25, 6, 162, 0, 0, 0, 3, 0)
        elif b == 172:
            return Diff(33, -14, 173, 0, 0, 0, 3, 0)
        elif b == 177:
            return Diff(25, 0, 174, 0, 0, 0, 3, 0)
        elif b == 178:
            return Diff(-3, -21, 0, 0, 0, 0, 2, 0)
        elif b == 181:
            return Diff(-3, -22, 0, 0, 0, 0, 2, 0)
        elif b == 189:
            return Diff(25, -12, 164, 0, 0, 0, 3, 0)
    elif a == 202:
        if (b == 128) or (b == 131) or (b == 136):
            return Diff(-4, 38, 0, 0, 0, 0, 2, 0)
        elif b == 130:
            return Diff(32, 29, 133, 0, 0, 0, 3, 0)
        elif b == 135:
            return Diff(32, 23, 177, 0, 0, 0, 3, 0)
        elif b == 137:
            return Diff(-1, -5, 0, 0, 0, 0, 2, 0)
        elif (b == 138) or (b == 139):
            return Diff(-4, 39, 0, 0, 0, 0, 2, 0)
        elif b == 140:
            return Diff(-1, -7, 0, 0, 0, 0, 2, 0)
        elif b == 146:
            return Diff(-4, 37, 0, 0, 0, 0, 2, 0)
        elif b == 157:
            return Diff(32, 1, 178, 0, 0, 0, 3, 0)
        elif b == 158:
            return Diff(32, 0, 176, 0, 0, 0, 3, 0)
    elif a == 205:
        if b == 133:
            return Diff(1, 20, 0, 0, 0, 0, 2, 0)
        elif (b == 177) or (b == 179) or (b == 183):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b >= 187 and b <= 189:
            return Diff(2, 2, 0, 0, 0, 0, 2, 0)
    elif a == 206:
        if b == 144:
            return Diff(0, 9, 204, 136, 204, 129, 6, 0)
        elif b == 172:
            return Diff(0, -38, 0, 0, 0, 0, 2, 0)
        elif b >= 173 and b <= 175:
            return Diff(0, -37, 0, 0, 0, 0, 2, 0)
        elif b == 176:
            return Diff(0, -11, 204, 136, 204, 129, 6, 0)
        elif b >= 177 and b <= 191:
            return Diff(0, -32, 0, 0, 0, 0, 2, 0)
    elif a == 207:
        if (b == 128) or (b == 129) or (b >= 131 and b <= 139):
            return Diff(-1, 32, 0, 0, 0, 0, 2, 0)
        elif b == 130:
            return Diff(-1, 33, 0, 0, 0, 0, 2, 0)
        elif b == 140:
            return Diff(-1, 0, 0, 0, 0, 0, 2, 0)
        elif (b == 141) or (b == 142):
            return Diff(-1, 1, 0, 0, 0, 0, 2, 0)
        elif b == 144:
            return Diff(-1, 2, 0, 0, 0, 0, 2, 0)
        elif b == 145:
            return Diff(-1, 7, 0, 0, 0, 0, 2, 0)
        elif b == 149:
            return Diff(-1, 17, 0, 0, 0, 0, 2, 0)
        elif b == 150:
            return Diff(-1, 10, 0, 0, 0, 0, 2, 0)
        elif b == 151:
            return Diff(0, -8, 0, 0, 0, 0, 2, 0)
        elif (b >= 153 and b <= 175 and b & 1 == 1) or (b == 184) or (b == 187):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 176:
            return Diff(-1, -22, 0, 0, 0, 0, 2, 0)
        elif b == 177:
            return Diff(-1, -16, 0, 0, 0, 0, 2, 0)
        elif b == 178:
            return Diff(0, 7, 0, 0, 0, 0, 2, 0)
        elif b == 179:
            return Diff(-2, 12, 0, 0, 0, 0, 2, 0)
        elif b == 181:
            return Diff(-1, -32, 0, 0, 0, 0, 2, 0)
    elif a == 208:
        if b >= 176 and b <= 191:
            return Diff(0, -32, 0, 0, 0, 0, 2, 0)
    elif a == 209:
        if b >= 128 and b <= 143:
            return Diff(-1, 32, 0, 0, 0, 0, 2, 0)
        elif b >= 144 and b <= 159:
            return Diff(-1, -16, 0, 0, 0, 0, 2, 0)
        elif b >= 161 and b <= 191 and b & 1 == 1:
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
    elif a == 210:
        if (b == 129) or (b >= 139 and b <= 191 and b & 1 == 1):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
    elif a == 211:
        if (b >= 130 and b <= 142 and b & 1 == 0) or (
            b >= 145 and b <= 191 and b & 1 == 1
        ):
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
        elif b == 143:
            return Diff(0, -15, 0, 0, 0, 0, 2, 0)
    elif a == 212:
        if b >= 129 and b <= 175 and b & 1 == 1:
            return Diff(0, -1, 0, 0, 0, 0, 2, 0)
    elif a == 213:
        if b >= 161 and b <= 175:
            return Diff(-1, 16, 0, 0, 0, 0, 2, 0)
        elif b >= 176 and b <= 191:
            return Diff(0, -48, 0, 0, 0, 0, 2, 0)
    elif a == 214:
        if b >= 128 and b <= 134:
            return Diff(-1, 16, 0, 0, 0, 0, 2, 0)
        elif b == 135:
            return Diff(-2, 46, 213, 146, 0, 0, 4, 0)
    return Diff(0, 0, 0, 0, 0, 0, 2, 0)


@always_inline
def _to_upper(a: UInt8, b: UInt8, c: UInt8) -> Diff:
    """Computes byte adjustments for uppercasing a 3-byte UTF-8 character.

    Args:
        a: The leading byte of the UTF-8 sequence.
        b: The first continuation byte of the UTF-8 sequence.
        c: The second continuation byte of the UTF-8 sequence.

    Returns:
        A `Diff` of `(delta_byte0, delta_byte1, delta_byte2, extra_byte3,
        extra_byte4, extra_byte5, output_len, 0)`. Each delta is added to the
        corresponding input byte to form the first three output bytes.
        `extra_byte3` through `extra_byte5` are written directly as subsequent
        output bytes (not deltas). `output_len` is the byte length of the
        result (1 to 6). Returns `Diff(0, 0, 0, 0, 0, 0, 3, 0)` if no case
        conversion applies.
    """
    if a == 225:
        if b == 143 and c >= 184 and c <= 189:
            return Diff(0, 0, -8, 0, 0, 0, 3, 0)
        elif b == 178:
            if c == 128:
                return Diff(-17, -32, 0, 0, 0, 0, 2, 0)
            elif c == 129:
                return Diff(-17, -30, 0, 0, 0, 0, 2, 0)
            elif c == 130:
                return Diff(-17, -20, 0, 0, 0, 0, 2, 0)
            elif c == 131:
                return Diff(-17, -17, 0, 0, 0, 0, 2, 0)
            elif c == 132:
                return Diff(-17, -16, 0, 0, 0, 0, 2, 0)
            elif c == 133:
                return Diff(-17, -16, 0, 0, 0, 0, 2, 0)
            elif c == 134:
                return Diff(-17, -8, 0, 0, 0, 0, 2, 0)
            elif c == 135:
                return Diff(-16, -16, 0, 0, 0, 0, 2, 0)
            elif c == 136:
                return Diff(9, -25, 2, 0, 0, 0, 3, 0)
        elif b == 181:
            if c == 185:
                return Diff(9, -24, 4, 0, 0, 0, 3, 0)
            elif c == 189:
                return Diff(1, -4, -26, 0, 0, 0, 3, 0)
        elif b == 182 and c == 142:
            return Diff(9, -23, -8, 0, 0, 0, 3, 0)
        elif b == 184 and c >= 129 and c <= 191 and c & 1 == 1:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 185 and c >= 129 and c <= 191 and c & 1 == 1:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 186:
            if c >= 129 and c <= 149 and c & 1 == 1:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 150:
                return Diff(-153, 18, 27, 0, 0, 0, 3, 0)
            elif c == 151:
                return Diff(-141, 18, -15, 0, 0, 0, 3, 0)
            elif c == 152:
                return Diff(-138, 18, -14, 0, 0, 0, 3, 0)
            elif c == 153:
                return Diff(-136, 18, -15, 0, 0, 0, 3, 0)
            elif c == 154:
                return Diff(-160, 16, 36, 0, 0, 0, 3, 0)
            elif c == 155:
                return Diff(0, -1, 5, 0, 0, 0, 3, 0)
            elif c >= 161 and c <= 191 and c & 1 == 1:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 187 and c >= 129 and c <= 191 and c & 1 == 1:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 188 and (
            (c >= 128 and c <= 135)
            or (c >= 144 and c <= 149)
            or (c >= 160 and c <= 167)
            or (c >= 176 and c <= 183)
        ):
            return Diff(0, 0, 8, 0, 0, 0, 3, 0)
        elif b == 189:
            if c >= 128 and c <= 133:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 144:
                return Diff(-19, -24, 60, 147, 0, 0, 4, 0)
            elif c >= 145 and c <= 151 and c & 1 == 1:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 146:
                return Diff(-19, -24, 58, 147, 204, 128, 6, 0)
            elif c == 148:
                return Diff(-19, -24, 56, 147, 204, 129, 6, 0)
            elif c == 150:
                return Diff(-19, -24, 54, 147, 205, 130, 6, 0)
            elif c >= 160 and c <= 167:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 176:
                return Diff(0, 1, 10, 0, 0, 0, 3, 0)
            elif c == 177:
                return Diff(0, 1, 10, 0, 0, 0, 3, 0)
            elif c >= 178 and c <= 181:
                return Diff(0, 2, -42, 0, 0, 0, 3, 0)
            elif c == 182:
                return Diff(0, 2, -28, 0, 0, 0, 3, 0)
            elif c == 183:
                return Diff(0, 2, -28, 0, 0, 0, 3, 0)
            elif c == 184:
                return Diff(0, 2, 0, 0, 0, 0, 3, 0)
            elif c == 185:
                return Diff(0, 2, 0, 0, 0, 0, 3, 0)
            elif c == 186:
                return Diff(0, 2, -16, 0, 0, 0, 3, 0)
            elif c == 187:
                return Diff(0, 2, -16, 0, 0, 0, 3, 0)
            elif c == 188:
                return Diff(0, 2, -2, 0, 0, 0, 3, 0)
            elif c == 189:
                return Diff(0, 2, -2, 0, 0, 0, 3, 0)
        elif b == 190:
            if c >= 128 and c <= 135:
                return Diff(0, -2, 8, 206, 153, 0, 5, 0)
            elif c >= 136 and c <= 143:
                return Diff(0, -2, 0, 206, 153, 0, 5, 0)
            elif c >= 144 and c <= 151:
                return Diff(0, -2, 24, 206, 153, 0, 5, 0)
            elif c >= 152 and c <= 159:
                return Diff(0, -2, 16, 206, 153, 0, 5, 0)
            elif c >= 160 and c <= 167:
                return Diff(0, -1, 8, 206, 153, 0, 5, 0)
            elif c >= 168 and c <= 175:
                return Diff(0, -1, 0, 206, 153, 0, 5, 0)
            elif c == 176:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 177:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 178:
                return Diff(0, 0, 8, 206, 153, 0, 5, 0)
            elif c == 179:
                return Diff(-19, -45, 27, 153, 0, 0, 4, 0)
            elif c == 180:
                return Diff(-19, -56, 26, 153, 0, 0, 4, 0)
            elif c == 182:
                return Diff(-19, -45, 23, 130, 0, 0, 4, 0)
            elif c == 183:
                return Diff(-19, -45, 22, 130, 206, 153, 6, 0)
            elif c == 188:
                return Diff(-19, -45, 18, 153, 0, 0, 4, 0)
            elif c == 190:
                return Diff(-19, -37, 0, 0, 0, 0, 2, 0)
        elif b == 191:
            if c == 130:
                return Diff(0, 0, 8, 206, 153, 0, 5, 0)
            elif c == 131:
                return Diff(-19, -40, 75, 153, 0, 0, 4, 0)
            elif c == 132:
                return Diff(-19, -54, 74, 153, 0, 0, 4, 0)
            elif c == 134:
                return Diff(-19, -40, 71, 130, 0, 0, 4, 0)
            elif c == 135:
                return Diff(-19, -40, 70, 130, 206, 153, 6, 0)
            elif c == 140:
                return Diff(-19, -40, 66, 153, 0, 0, 4, 0)
            elif c == 144:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 145:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 146:
                return Diff(-19, -38, 58, 136, 204, 128, 6, 0)
            elif c == 147:
                return Diff(-19, -38, 57, 136, 204, 129, 6, 0)
            elif c == 150:
                return Diff(-19, -38, 55, 130, 0, 0, 4, 0)
            elif c == 151:
                return Diff(-19, -38, 53, 136, 205, 130, 6, 0)
            elif c == 160:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 161:
                return Diff(0, 0, 8, 0, 0, 0, 3, 0)
            elif c == 162:
                return Diff(-19, -26, 42, 136, 204, 128, 6, 0)
            elif c == 163:
                return Diff(-19, -26, 41, 136, 204, 129, 6, 0)
            elif c == 164:
                return Diff(-19, -30, 40, 147, 0, 0, 4, 0)
            elif c == 165:
                return Diff(0, 0, 7, 0, 0, 0, 3, 0)
            elif c == 166:
                return Diff(-19, -26, 39, 130, 0, 0, 4, 0)
            elif c == 167:
                return Diff(-19, -26, 37, 136, 205, 130, 6, 0)
            elif c == 178:
                return Diff(0, 0, 8, 206, 153, 0, 5, 0)
            elif c == 179:
                return Diff(-19, -22, 27, 153, 0, 0, 4, 0)
            elif c == 180:
                return Diff(-19, -48, 26, 153, 0, 0, 4, 0)
            elif c == 182:
                return Diff(-19, -22, 23, 130, 0, 0, 4, 0)
            elif c == 183:
                return Diff(-19, -22, 22, 130, 206, 153, 6, 0)
            elif c == 188:
                return Diff(-19, -22, 18, 153, 0, 0, 4, 0)
    elif a == 226:
        if b == 133:
            if c == 142:
                return Diff(0, -1, 36, 0, 0, 0, 3, 0)
            elif c >= 176 and c <= 191:
                return Diff(0, 0, -16, 0, 0, 0, 3, 0)
        elif b == 134 and c == 132:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 147:
            if c >= 144 and c <= 153:
                return Diff(0, -1, 38, 0, 0, 0, 3, 0)
            elif c >= 154 and c <= 169:
                return Diff(0, 0, -26, 0, 0, 0, 3, 0)
        elif b == 176 and c >= 176 and c <= 191:
            return Diff(0, 0, -48, 0, 0, 0, 3, 0)
        elif b == 177:
            if c >= 128 and c <= 159:
                return Diff(0, -1, 16, 0, 0, 0, 3, 0)
            elif c == 161:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 165:
                return Diff(-26, 9, 0, 0, 0, 0, 2, 0)
            elif c == 166:
                return Diff(-26, 13, 0, 0, 0, 0, 2, 0)
            elif c >= 168 and c <= 172 and c & 1 == 0:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 179:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 182:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 178 and c >= 129 and c <= 191 and c & 1 == 1:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 179 and (
            (c >= 129 and c <= 163 and c & 1 == 1)
            or (c == 172)
            or (c == 174)
            or (c == 179)
        ):
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 180:
            if c >= 128 and c <= 159:
                return Diff(-1, -50, 32, 0, 0, 0, 3, 0)
            elif c >= 160 and c <= 165:
                return Diff(-1, -49, -32, 0, 0, 0, 3, 0)
            elif c == 167:
                return Diff(-1, -49, -32, 0, 0, 0, 3, 0)
            elif c == 173:
                return Diff(-1, -49, -32, 0, 0, 0, 3, 0)
    elif a == 234:
        if b == 153 and c >= 129 and c <= 173 and c & 1 == 1:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 154 and c >= 129 and c <= 155 and c & 1 == 1:
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 156 and (
            (c >= 163 and c <= 175 and c & 1 == 1)
            or (c >= 179 and c <= 191 and c & 1 == 1)
        ):
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 157 and (
            (c >= 129 and c <= 175 and c & 1 == 1)
            or (c == 186)
            or (c == 188)
            or (c == 191)
        ):
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 158:
            if c >= 129 and c <= 135 and c & 1 == 1:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 140:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 145:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 147:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c == 148:
                return Diff(0, 1, -16, 0, 0, 0, 3, 0)
            elif c >= 151 and c <= 169 and c & 1 == 1:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
            elif c >= 181 and c <= 191 and c & 1 == 1:
                return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 159 and (
            (c == 129)
            or (c == 131)
            or (c == 136)
            or (c == 138)
            or (c == 145)
            or (c == 151)
            or (c == 153)
            or (c == 182)
        ):
            return Diff(0, 0, -1, 0, 0, 0, 3, 0)
        elif b == 173:
            if c == 147:
                return Diff(0, -15, 32, 0, 0, 0, 3, 0)
            elif c >= 176 and c <= 191:
                return Diff(-9, -31, -16, 0, 0, 0, 3, 0)
        elif b == 174:
            if c >= 128 and c <= 143:
                return Diff(-9, -32, 48, 0, 0, 0, 3, 0)
            elif c >= 144 and c <= 191:
                return Diff(-9, -31, -16, 0, 0, 0, 3, 0)
    elif a == 239:
        if b == 172:
            if c == 128:
                return Diff(-169, -102, 0, 0, 0, 0, 2, 0)
            elif c == 129:
                return Diff(-169, -99, 0, 0, 0, 0, 2, 0)
            elif c == 130:
                return Diff(-169, -96, 0, 0, 0, 0, 2, 0)
            elif c == 131:
                return Diff(-169, -102, -58, 0, 0, 0, 3, 0)
            elif c == 132:
                return Diff(-169, -102, -56, 0, 0, 0, 3, 0)
            elif c == 133:
                return Diff(-156, -88, 0, 0, 0, 0, 2, 0)
            elif c == 134:
                return Diff(-156, -88, 0, 0, 0, 0, 2, 0)
            elif c == 147:
                return Diff(-26, -40, 66, 134, 0, 0, 4, 0)
            elif c == 148:
                return Diff(-26, -40, 64, 181, 0, 0, 4, 0)
            elif c == 149:
                return Diff(-26, -40, 63, 187, 0, 0, 4, 0)
            elif c == 150:
                return Diff(-26, -30, 63, 134, 0, 0, 4, 0)
            elif c == 151:
                return Diff(-26, -40, 61, 189, 0, 0, 4, 0)
        elif b == 189 and c >= 129 and c <= 154:
            return Diff(0, -1, 32, 0, 0, 0, 3, 0)
    return Diff(0, 0, 0, 0, 0, 0, 3, 0)


@always_inline
def _to_upper(a: UInt8, b: UInt8, c: UInt8, d: UInt8) -> Diff:
    """Computes byte adjustments for uppercasing a 4-byte UTF-8 character.

    4-byte characters always produce a 4-byte result, so no output length is
    encoded in the return value.

    Args:
        a: The leading byte of the UTF-8 sequence.
        b: The first continuation byte of the UTF-8 sequence.
        c: The second continuation byte of the UTF-8 sequence.
        d: The third continuation byte of the UTF-8 sequence.

    Returns:
        A `Diff` of `(delta_byte0, delta_byte1, delta_byte2, delta_byte3, ...)`.
        Each delta is added to the corresponding input byte to produce the
        output. Returns `Diff(0, 0, 0, 0, ...)` if no case conversion applies.
    """
    if a == 240:
        if b == 144:
            if c == 144 and d >= 168 and d <= 191:
                return Diff(0, 0, 0, -40, 0, 0, 0, 0)
            elif c == 145 and d >= 128 and d <= 143:
                return Diff(0, 0, -1, 24, 0, 0, 0, 0)
            elif c == 147:
                if d >= 152 and d <= 167:
                    return Diff(0, 0, -1, 24, 0, 0, 0, 0)
                elif d >= 168 and d <= 187:
                    return Diff(0, 0, 0, -40, 0, 0, 0, 0)
            elif c == 150:
                if d >= 151 and d <= 161:
                    return Diff(0, 0, -1, 25, 0, 0, 0, 0)
                elif d >= 163 and d <= 166:
                    return Diff(0, 0, -1, 25, 0, 0, 0, 0)
                elif d >= 167 and d <= 177:
                    return Diff(0, 0, 0, -39, 0, 0, 0, 0)
                elif d >= 179 and d <= 185:
                    return Diff(0, 0, 0, -39, 0, 0, 0, 0)
                elif d == 187:
                    return Diff(0, 0, 0, -39, 0, 0, 0, 0)
                elif d == 188:
                    return Diff(0, 0, 0, -39, 0, 0, 0, 0)
            elif c == 179 and d >= 128 and d <= 178:
                return Diff(0, 0, -1, 0, 0, 0, 0, 0)
        elif b == 145 and c == 163 and d >= 128 and d <= 159:
            return Diff(0, 0, -1, 32, 0, 0, 0, 0)
        elif b == 150 and c == 185 and d >= 160 and d <= 191:
            return Diff(0, 0, 0, -32, 0, 0, 0, 0)
        elif b == 158:
            if c == 164 and d >= 162 and d <= 191:
                return Diff(0, 0, 0, -34, 0, 0, 0, 0)
            elif c == 165 and d >= 128 and d <= 131:
                return Diff(0, 0, -1, 30, 0, 0, 0, 0)
    return Diff(0, 0, 0, 0, 0, 0, 0, 0)


def upper_utf8[ascii_optimized: Bool = True](s: String) -> String:
    """Converts a UTF-8 encoded string to uppercase.

    Uses a byte-level decision tree to compute uppercase transformations for
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
        A new `String` with all characters converted to uppercase.
    """
    comptime SIMD_WIDTH = 8
    # The width is set to 8 bytes based on performance benchmark for pure ASCII examples
    # and with examples where ASCII is mixed with non ASCII characters.
    # It is imaginable to promote SIMD_WIDTH from constant to be a function parameter.
    var byte_len = s.byte_length()
    var capacity = byte_len // 2 * 3 + 1
    var result = String(capacity=capacity)
    var buf = result.unsafe_ptr_mut()
    var offset = 0
    var p = s.unsafe_ptr()
    var count = 0
    while offset < byte_len:
        comptime if ascii_optimized:
            if offset + SIMD_WIDTH <= byte_len:
                var vec = (p + offset).load[SIMD_WIDTH]()
                if vec.le(127).reduce_and():
                    var is_lower = vec.ge(97) & vec.le(122)
                    var delta = is_lower.select(
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
        if count + 6 >= capacity:
            capacity *= 2
            buf = result.unsafe_ptr_mut(capacity)
            buf += count
        if char_length == 1:
            buf[0] = b0 ^ (Byte((b0 >= 97) & (b0 <= 122)) << 5)
            buf += 1
            count += 1
        elif char_length == 2:
            var b1 = p[offset + 1]
            var diff = _to_upper(b0, b1)
            var out_len = Int(diff[6])
            buf[0] = Byte(Int16(b0) + diff[0])
            buf[1] = Byte(Int16(b1) + diff[1])
            buf[2] = Byte(diff[2])
            buf[3] = Byte(diff[3])
            buf[4] = Byte(diff[4])
            buf[5] = Byte(diff[5])
            buf += out_len
            count += out_len
        elif char_length == 3:
            var b1 = p[offset + 1]
            var b2 = p[offset + 2]
            var diff = _to_upper(b0, b1, b2)
            var out_len = Int(diff[6])
            buf[0] = Byte(Int16(b0) + diff[0])
            buf[1] = Byte(Int16(b1) + diff[1])
            buf[2] = Byte(Int16(b2) + diff[2])
            buf[3] = Byte(diff[3])
            buf[4] = Byte(diff[4])
            buf[5] = Byte(diff[5])
            buf += out_len
            count += out_len
        elif char_length == 4:
            var b1 = p[offset + 1]
            var b2 = p[offset + 2]
            var b3 = p[offset + 3]
            var diff = _to_upper(b0, b1, b2, b3)
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
