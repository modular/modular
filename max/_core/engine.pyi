# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

class Model:
    def load(self) -> None: ...

class InferenceSession:
    def __init__(self, config: dict = ...) -> None: ...
    def compile(self, model_path: str, config: dict = ...) -> Model: ...
