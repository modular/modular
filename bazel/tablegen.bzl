"""TableGen rules, kept out of api.bzl because loading them fetches LLVM."""

load("//bazel/internal:tablegen.bzl", _driver_option_tablegen = "driver_option_tablegen", _gentbl_cc_library = "gentbl_cc_library", _gentbl_filegroup = "gentbl_filegroup", _gentbl_mlir_dialect = "gentbl_mlir_dialect", _gentbl_modular_passes = "gentbl_modular_passes", _td_library = "td_library")  # buildifier: disable=bzl-visibility
load("//max/python/max/_core/max_license/mlir_nanobind:tblgen.bzl", _mlir_nanobind = "mlir_nanobind")  # buildifier: disable=bzl-visibility

driver_option_tablegen = _driver_option_tablegen
gentbl_cc_library = _gentbl_cc_library
gentbl_filegroup = _gentbl_filegroup
gentbl_mlir_dialect = _gentbl_mlir_dialect
gentbl_modular_passes = _gentbl_modular_passes
td_library = _td_library

def mlir_nanobind(**kwargs):
    _mlir_nanobind(gentbl_filegroup = _gentbl_filegroup, **kwargs)
