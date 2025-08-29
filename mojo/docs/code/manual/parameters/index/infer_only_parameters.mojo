# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


fn dependent_type[dtype: DType, //, value: Scalar[dtype]]():
    print("Value: ", value)
    print("DType: ", dtype)


def mutate_span(span: Span[mut=True, Byte]):
    for i in range(0, len(span), 2):
        if i + 1 < len(span):
            span.swap_elements(i, i + 1)


def main():
    dependent_type[Float64(2.2)]()
    s = String("Robinson Crusoe surfed the interwebs.")
    span = s.as_bytes_mut()
    mutate_span(span)
    print(s)
