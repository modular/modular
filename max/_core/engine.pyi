# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path

class Model:
    def execute(self, *args) -> None: ...
    def load(self) -> None: ...

class InferenceSession:
    def __init__(self, config: dict = ...) -> None: ...
    def compile(self, model_path: Path, config: dict = ...) -> Model: ...
