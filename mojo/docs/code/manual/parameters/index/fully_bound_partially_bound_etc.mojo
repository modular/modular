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


@fieldwise_init
struct MyType[s: String, i: Int, i2: Int, b: Bool = True]:
    pass


# Fully-bound
def my_fn1(mt: MyType["Hello", 3, 4, True]):
    pass


# Partially-bound
def my_fn2(mt: MyType["Hola", _, _, True]):
    pass


# Unbound
def my_fn3(mt: MyType[_, _, _, _]):
    pass


# Partially-bound with omitted parameters
def my_fn4(mt: MyType["Hi there!"]):
    pass


# Unbound with omitted parameters
def my_fn5(mt: MyType):
    pass


@fieldwise_init
struct KeyWordStruct[pos_or_kw: Int, *, kw_only: Int = 10]:
    pass


# Unbind both pos_or_kw and kw_only parameters
fn use_kw_struct(k: KeyWordStruct[**_]):
    pass


def main():
    # start-partially-bound-example
    alias StringKeyDict = Dict[String, _]
    var b: StringKeyDict[UInt8] = {"answer": 42}
    # end-partially-bound-example
    _ = b^

    use_kw_struct(KeyWordStruct[10, kw_only=11]())
