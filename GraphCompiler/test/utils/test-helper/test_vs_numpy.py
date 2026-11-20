# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
import argparse
import os
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from ops import NumpyImpl
from test_cases import get_test_case

"""
This tool executes a MO graph using `mt` and compares it to the numpy
reference implementation.
"""


parser = argparse.ArgumentParser(
    prog="test_vs_numpy",
    description="From the given python test case create a MO graph",
)
parser.add_argument(
    "-n", "--name", required=True, help="Name of the test case to export"
)
parser.add_argument(
    "-f", "--file-to-run", required=True, help="MGP file to execute"
)
args = parser.parse_args()


def test_vs_numpy(testCase, file_to_run) -> None:  # noqa: ANN001
    np.random.seed(42)

    golden_value = []

    inputs: int | None = None
    num_inputs = None

    sizes_to_test = testCase.input_sizes_to_test()
    assert len(sizes_to_test) == 1, "Only one input config supported right now."

    # Create the input data + golden test values.
    for i in range(testCase.num_runs()):
        ins = testCase.generate_inputs(sizes_to_test[0])
        # Create list lazily so we know how many to create.
        if inputs is None:
            num_inputs = len(ins)
            inputs = [[] for _ in range(num_inputs)]  # type: ignore[assignment]

        for i in range(num_inputs):  # type: ignore[arg-type]
            inputs[i].append(ins[i])  # type: ignore
        golden_value.append(testCase.run(ins, NumpyImpl()))

    if not num_inputs or not inputs:
        print("No inputs to test!")
        sys.exit(0)

    # Concat all the test data into one numpy array for each input and save out to a file which mt can load.
    inputs_as_numpy = [np.array(inputs[i]) for i in range(num_inputs)]  # type: ignore[index]

    input_launch_line = ""

    tmp_files = []
    for i in range(num_inputs):
        tmp = NamedTemporaryFile(suffix=".bin")
        tmp_files.append(tmp)
        inputs_as_numpy[i].tofile(tmp)

        # Save to file and flush so MT sees it.
        tmp.flush()

        shape = inputs_as_numpy[i].shape
        shape = "x".join([str(s) for s in shape[1:]])  # type: ignore[assignment]

        # Add the input to the MT command.
        input_launch_line += (
            f" --input-data={i}::{tmp.name} --input-shapes={shape}"
        )

    # Execute with mt to get the MO output.
    mt_command = "mt"

    if "MODULAR_MT" in os.environ:
        mt_command = os.environ["MODULAR_MT"]

    out = subprocess.run(
        (
            f"{mt_command} --execute"
            f" {file_to_run} --num-runs={testCase.num_runs()}"
            f" {input_launch_line} --result-output-style=full"
        ),
        capture_output=True,
        shell=True,
    )

    if out.returncode != 0:
        print("mt failed to execute given file")
        print("stdout:")
        print(out.stdout)
        print("stderr:")
        print(out.stderr)
        assert False, "MT failed to run"  # noqa: B011

    # The run index
    output_index = -1

    # The subtensor within that run for multiple outputs.
    tensor_within_output = 0

    # Iterate over all outputs.
    for line in out.stdout.splitlines()[1:]:
        if len(line) == 0:
            continue

        line_as_str = str(line)

        # The first line of a multi input return will contain
        # `serving_default' returned`. We use this to signal that we are
        # reading the next input rather than an input within the same return.
        if line_as_str.startswith("b\"'serving_default' returned"):
            tensor_within_output = 0
            output_index += 1
        else:
            tensor_within_output += 1

        x = re.findall(r"tensor<(\d+(?:x\d+)*)", line_as_str)

        if len(x) != 1:
            continue

        val_to_compare = golden_value[output_index]

        # For multiple return cases.
        if isinstance(val_to_compare, list):
            if tensor_within_output >= len(val_to_compare):
                print(
                    "MO returned more tensors than numpy expected for output"
                    f" {output_index}"
                )
                sys.exit(0)
            val_to_compare = val_to_compare[tensor_within_output]

        # Check the dimensions match.
        for tensor in x:
            mo_shape = tensor.split("x")
            for mo_dim, np_dim in zip(
                mo_shape, val_to_compare.shape, strict=True
            ):
                if int(mo_dim) != np_dim:
                    print(val_to_compare)
                    print(line_as_str)
                    print("Inputs: ")
                    for inp in inputs_as_numpy:
                        print(inp[output_index])

                    print(
                        f"On output {output_index} dimensions don't match! Mo"
                        f" {mo_shape} vs np {val_to_compare.shape}"
                    )
                    sys.exit(0)

        # The raw output data returned by the test
        raw_out = line_as_str[
            line_as_str.find("[") + 1 : line_as_str.find("]")
        ].split(", ")

        if val_to_compare.dtype == bool:
            for i, val in enumerate(raw_out):
                raw_out[i] = val == "True"  # type: ignore[call-overload]

        mo_out = np.array(list(raw_out), dtype=val_to_compare.dtype).reshape(
            val_to_compare.shape
        )

        if not testCase.compare_results(mo_out, val_to_compare):
            print(f"Output at {output_index} does not match!")
            sys.exit(0)

    print("All MO values match numpy")


test_vs_numpy(get_test_case(args.name), args.file_to_run)
