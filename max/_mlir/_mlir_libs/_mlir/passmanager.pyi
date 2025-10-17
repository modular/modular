# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Callable
from typing import overload

import max._mlir.ir

class ExternalPass:
    def signal_pass_failure(self) -> None: ...

class PassManager:
    def __init__(
        self,
        anchor_op: str = "any",
        context: max._mlir.ir.Context | None = None,
    ) -> None:
        """Create a new PassManager for the current (or provided) Context."""

    def enable_ir_printing(
        self,
        print_before_all: bool = False,
        print_after_all: bool = True,
        print_module_scope: bool = False,
        print_after_change: bool = False,
        print_after_failure: bool = False,
        large_elements_limit: int | None = None,
        large_resource_limit: int | None = None,
        enable_debug_info: bool = False,
        print_generic_op_form: bool = False,
        tree_printing_dir_path: str | None = None,
    ) -> None:
        """Enable IR printing, default as mlir-print-ir-after-all."""

    def enable_verifier(self, enable: bool) -> None:
        """Enable / disable verify-each."""

    def enable_timing(self) -> None:
        """Enable pass timing."""

    @staticmethod
    def parse(
        pipeline: str, context: max._mlir.ir.Context | None = None
    ) -> PassManager:
        """
        Parse a textual pass-pipeline and return a top-level PassManager that can be applied on a Module. Throw a ValueError if the pipeline can't be parsed
        """

    @overload
    def add(self, pipeline: str) -> None:
        """
        Add textual pipeline elements to the pass manager. Throws a ValueError if the pipeline can't be parsed.
        """

    @overload
    def add(
        self,
        run: Callable,
        name: str | None = None,
        argument: str | None = "",
        description: str | None = "",
        op_name: str | None = "",
    ) -> None:
        """Add a python-defined pass to the pass manager."""

    def run(self, operation: max._mlir.ir._OperationBase) -> None:
        """
        Run the pass manager on the provided operation, raising an MLIRError on failure.
        """

    def __str__(self) -> str:
        """
        Print the textual representation for this PassManager, suitable to be passed to `parse` for round-tripping.
        """
