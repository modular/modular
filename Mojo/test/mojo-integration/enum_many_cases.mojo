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

# RUN: %mojo %s | FileCheck %s

"""An `__enum` with more than 256 cases: the storage widens its discriminant
past `UInt8`, as `!kgen.variant` does."""

from std.builtin._enum import __enum_discriminant_dtype


# TODO: `ImplicitlyCopyable` only so that the payload-less cases can be used as
# runtime values; a `comptime` value of another type does not materialize.
__enum Many(ImplicitlyCopyable, Writable):
    case c0
    case c1
    case c2
    case c3
    case c4
    case c5
    case c6
    case c7
    case c8
    case c9
    case c10
    case c11
    case c12
    case c13
    case c14
    case c15
    case c16
    case c17
    case c18
    case c19
    case c20
    case c21
    case c22
    case c23
    case c24
    case c25
    case c26
    case c27
    case c28
    case c29
    case c30
    case c31
    case c32
    case c33
    case c34
    case c35
    case c36
    case c37
    case c38
    case c39
    case c40
    case c41
    case c42
    case c43
    case c44
    case c45
    case c46
    case c47
    case c48
    case c49
    case c50
    case c51
    case c52
    case c53
    case c54
    case c55
    case c56
    case c57
    case c58
    case c59
    case c60
    case c61
    case c62
    case c63
    case c64
    case c65
    case c66
    case c67
    case c68
    case c69
    case c70
    case c71
    case c72
    case c73
    case c74
    case c75
    case c76
    case c77
    case c78
    case c79
    case c80
    case c81
    case c82
    case c83
    case c84
    case c85
    case c86
    case c87
    case c88
    case c89
    case c90
    case c91
    case c92
    case c93
    case c94
    case c95
    case c96
    case c97
    case c98
    case c99
    case c100
    case c101
    case c102
    case c103
    case c104
    case c105
    case c106
    case c107
    case c108
    case c109
    case c110
    case c111
    case c112
    case c113
    case c114
    case c115
    case c116
    case c117
    case c118
    case c119
    case c120
    case c121
    case c122
    case c123
    case c124
    case c125
    case c126
    case c127
    case c128
    case c129
    case c130
    case c131
    case c132
    case c133
    case c134
    case c135
    case c136
    case c137
    case c138
    case c139
    case c140
    case c141
    case c142
    case c143
    case c144
    case c145
    case c146
    case c147
    case c148
    case c149
    case c150
    case c151
    case c152
    case c153
    case c154
    case c155
    case c156
    case c157
    case c158
    case c159
    case c160
    case c161
    case c162
    case c163
    case c164
    case c165
    case c166
    case c167
    case c168
    case c169
    case c170
    case c171
    case c172
    case c173
    case c174
    case c175
    case c176
    case c177
    case c178
    case c179
    case c180
    case c181
    case c182
    case c183
    case c184
    case c185
    case c186
    case c187
    case c188
    case c189
    case c190
    case c191
    case c192
    case c193
    case c194
    case c195
    case c196
    case c197
    case c198
    case c199
    case c200
    case c201
    case c202
    case c203
    case c204
    case c205
    case c206
    case c207
    case c208
    case c209
    case c210
    case c211
    case c212
    case c213
    case c214
    case c215
    case c216
    case c217
    case c218
    case c219
    case c220
    case c221
    case c222
    case c223
    case c224
    case c225
    case c226
    case c227
    case c228
    case c229
    case c230
    case c231
    case c232
    case c233
    case c234
    case c235
    case c236
    case c237
    case c238
    case c239
    case c240
    case c241
    case c242
    case c243
    case c244
    case c245
    case c246
    case c247
    case c248
    case c249
    case c250
    case c251
    case c252
    case c253
    case c254
    case c255
    case c256
    case last(String)


def describe(value: Many) -> String:
    match value:
        case .c0:
            return "first"
        case .c256:
            return "c256"
        case .last(text):
            return "last " + text
        case _:
            return "other"


def main():
    # The widths change exactly where `!kgen.variant`'s do.
    comptime assert __enum_discriminant_dtype(256) == DType.uint8
    comptime assert __enum_discriminant_dtype(257) == DType.uint16
    comptime assert __enum_discriminant_dtype(65536) == DType.uint16
    comptime assert __enum_discriminant_dtype(65537) == DType.uint32
    var first: Many = .c0
    var past_byte: Many = .c256
    var last: Many = .last("x")
    # CHECK: 0 256 257
    print(
        first._get_enum_discriminant(),
        past_byte._get_enum_discriminant(),
        last._get_enum_discriminant(),
    )
    var copy = last.copy()
    last = .c255
    # CHECK-NEXT: first c256 last x other
    print(describe(first), describe(past_byte), describe(copy), describe(last))
    # CHECK-NEXT: Many.c256 Many.last(x)
    print(past_byte, copy)
