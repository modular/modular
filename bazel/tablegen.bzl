"""TableGen rules, kept out of api.bzl because loading them fetches LLVM."""

load("//bazel/internal:tablegen.bzl", _driver_option_tablegen = "driver_option_tablegen", _gentbl_cc_library = "gentbl_cc_library", _gentbl_mlir_dialect = "gentbl_mlir_dialect", _gentbl_modular_passes = "gentbl_modular_passes", _td_library = "td_library")  # buildifier: disable=bzl-visibility

def _noop(**_kwargs):
    pass

driver_option_tablegen = _driver_option_tablegen
gentbl_cc_library = _gentbl_cc_library
gentbl_mlir_dialect = _gentbl_mlir_dialect
gentbl_modular_passes = _gentbl_modular_passes
mlir_nanobind = _noop
td_library = _td_library
