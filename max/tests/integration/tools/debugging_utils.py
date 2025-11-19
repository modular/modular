from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.nn.hooks import PrintHook
from max.nn.layer import Module


@contextmanager
def add_max_hooks(
    output_directory: Path | None = None,
) -> Generator[None, None, None]:
    """Context manager that adds tensor printing hooks by patching the model class."""

    # Save original InferenceSession initializer.
    original_inference_init = InferenceSession.__init__
    hook = PrintHook()
    original_inference_init = InferenceSession.__init__

    def get_wrapped_load_state_dict(
        original_load_state_dict: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapped_load_state_dict(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            result = original_load_state_dict(self, *args, **kwargs)
            hook.name_layers(self)
            return result

        return wrapped_load_state_dict

    # If an output directory is provided, patch InferenceSession to enable debug prints.
    if output_directory is not None:

        def _patched_inference_init(
            session_self: InferenceSession, *args: Any, **kwargs: Any
        ) -> None:
            original_inference_init(session_self, *args, **kwargs)
            # Enable debug printing to file-style output when an output directory is specified.
            # If additional parameters (like output path) are supported, they can be added here.
            session_self.set_debug_print_options(
                style=PrintStyle.BINARY_MAX_CHECKPOINT,
                output_directory=output_directory,
            )

        InferenceSession.__init__ = _patched_inference_init  # type: ignore[assignment]

    original_load_state_dict = Module.load_state_dict
    Module.load_state_dict = get_wrapped_load_state_dict(  # type: ignore[method-assign]
        original_load_state_dict
    )

    try:
        yield
    finally:
        hook.remove()
        Module.load_state_dict = original_load_state_dict  # type: ignore[method-assign]
        # Restore original InferenceSession initializer if we patched it.
        InferenceSession.__init__ = original_inference_init  # type: ignore[method-assign]
