# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""


from evaluate_llama import NumpyDecoder, compare_values, run_llama3


def test_tiny_llama(testdata_directory):
    with open(testdata_directory / "tiny_llama_golden.json") as f:
        expected_results = NumpyDecoder().decode(f.read())
    actual = run_llama3(testdata_directory / "tiny_llama.gguf")
    compare_values(actual, expected_results)
