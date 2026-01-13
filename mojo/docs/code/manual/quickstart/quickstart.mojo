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

from python import Python, PythonObject


fn calculate_average(temps: List[Float64]) raises -> Float64:
    if len(temps) == 0:
        raise Error("No temperature data")

    total: Float64 = 0.0
    for index in range(len(temps)):
        total += temps[index]
    return total / len(temps)


fn main():
    print("Temperature Analyzer")
    temps: List[Float64] = [20.5, 22.3, 19.8, 25.1]
    print("Recorded", len(temps), "temperatures")

    for index in range(len(temps)):
        print("  Day {}: {}°C".format(index + 1, temps[index]))

    try:
        avg = calculate_average(temps)
        print("Average: {}°C".format(avg))

        if avg > 25.0:
            print("Status: Hot week")
        elif avg > 20.0:
            print("Status: Comfortable week")
        else:
            print("Status: Cool week")

        np = Python.import_module("numpy")
        pytemps: PythonObject = [20.5, 22.3, 19.8, 25.1]
        print("Temperature standard deviation:", np.std(pytemps))
        _ = pytemps^
    except e:
        print("Error:", e)
