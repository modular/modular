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
from std.testing import assert_equal


# The status and reading checks take arguments so the compiler can't fold a
# constant subject and warn that later cases are unreachable.
def status_message(status_code: Int) -> String:
    match status_code:
        case 200:
            return "OK"
        case 301 | 302:
            return "Redirect"
        case 404:
            return "Not found"
        case code if code >= 500:
            return String("Server error ", code)
        case _:
            return String("Unexpected status ", status_code)


def first_case_hits(status_code: Int) -> Int:
    var hits = 0
    match status_code:
        case 404:
            hits += 1
        case _:
            hits += 10
    return hits


def unmatched_hits(status_code: Int) -> Int:
    var hits = 0
    match status_code:
        case 200:
            hits += 1
    return hits


def forecast(temp_celsius: Optional[Int]) -> String:
    match temp_celsius:
        case .Some(temp) if temp > 20:
            return "warm"
        case .Some(_):
            return "cool"
        case .None:
            return "no reading"


def report(reading: Optional[Int]) -> String:
    if case .Some(temp) = reading:
        return String("The temperature is ", temp)
    else:
        return "No reading"


def next_reading(mut readings: List[Int]) -> Optional[Int]:
    if readings:
        return readings.pop()
    return None


def main() raises:
    # The match statement
    assert_equal(status_message(404), "Not found")
    assert_equal(status_message(200), "OK")
    assert_equal(status_message(302), "Redirect")
    assert_equal(status_message(503), "Server error 503")
    # A false guard moves on to the next case.
    assert_equal(status_message(418), "Unexpected status 418")

    # Only the first matching case runs.
    assert_equal(first_case_hits(404), 1)
    # No body runs when no case matches.
    assert_equal(unmatched_hits(404), 0)

    # Matching optional values
    assert_equal(forecast(Optional(25)), "warm")
    assert_equal(forecast(Optional(15)), "cool")
    assert_equal(forecast(None), "no reading")

    # if case
    assert_equal(report(Optional(25)), "The temperature is 25")
    assert_equal(report(None), "No reading")

    # while case
    var readings: List[Int] = [18, 22, 25]
    var seen: List[Int] = []
    var else_runs = 0
    while case .Some(temp) = next_reading(readings):
        seen.append(temp)
    else:
        else_runs += 1
    assert_equal(len(seen), 3)
    assert_equal(seen[0], 25)
    assert_equal(seen[1], 22)
    assert_equal(seen[2], 18)
    assert_equal(else_runs, 1)
