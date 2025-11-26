from pathlib import Path

import torch
from generate_llm_logits import generate_llm_logits
from max.driver.tensor import load_max_tensor
from max.entrypoints.cli import DevicesOptionType


def load_intermediate_tensors(
    model: str,
    framework: str,
    output_dir: Path = Path("/tmp/intermediate_tensors/torch"),
    device_type: str = "default",
    encoding_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Run a Transformers model using Torch with print hooks enabled and return intermediate tensors as a dictionary mapping tensor name to torch.Tensor.

    Args:
        model: Hugging Face model id (e.g., "google/gemma-3-1b-it").
        framework: Framework to run the model on (e.g., "torch", "max").
        dir: Output directory where Torch print hooks will write .pt files; may contain subdirectories.
        device_type: Device selector passed through DevicesOptionType (e.g., "default", "gpu", "cpu", "gpu:0,1,2").
        encoding_name: Optional explicit encoding/dtype (e.g., "bfloat16"). If None, the pipeline default is used.

    Returns:
        Dict keyed by emitted file name (str) with loaded torch.Tensor values.
    """

    generate_llm_logits(
        pipeline_name=model,
        framework_name=framework,
        print_intermediates=True,
        intermediates_dir=output_dir,
        device_specs=DevicesOptionType.device_specs(device_type),
        encoding_name=encoding_name if encoding_name else None,
        output_path=output_dir / "output.json",
        print_output=False,
        mini=True,
    )
    tensors_map: dict[str, torch.Tensor] = {}
    if framework == "torch":
        files = sorted(
            output_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime
        )
        for file in files:
            torch_tensor = torch.load(file)
            tensors_map[file.name] = torch_tensor
    elif framework == "max":
        files = sorted(
            output_dir.rglob("*.max"), key=lambda p: p.stat().st_mtime
        )
        for file in files:
            tensor = load_max_tensor(file)
            torch_tensor = torch.from_dlpack(tensor).cpu()
            tensors_map[file.name] = torch_tensor
    else:
        raise ValueError(f"Framework not supported: {framework}")
    return tensors_map


__all__ = ["load_intermediate_tensors"]
