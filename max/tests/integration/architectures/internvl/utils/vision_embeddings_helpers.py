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

"""Helper functions for InternVL vision embeddings tests."""

from collections.abc import Sequence

import numpy as np
import torch
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVisionConfig,
)
from internvl_impl.modeling_intern_vit import (
    InternVisionEmbeddings as HFInternVisionEmbeddings,
)
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from max.pipelines.architectures.internvl.model_config import InternVLConfig
from max.pipelines.architectures.internvl.tokenizer import (
    extract_patches_from_image,
)
from torch.utils.dlpack import from_dlpack

from utils.weight_converter import convert_hf_to_max_weights


@torch.no_grad()
def generate_torch_outputs(
    pixel_values: torch.Tensor,
    hf_config: HFInternVisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using the vLLM implementation."""

    # Create the reference model
    ref_model = (
        HFInternVisionEmbeddings(
            hf_config,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights
    ref_model.load_state_dict(embeddings_weights, strict=True)
    ref_model.eval()

    # Forward pass
    with torch.no_grad():
        output = ref_model(pixel_values)

    return output


def patchify_pixel_values(
    pixel_values: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Converts NCHW pixel values to the patched layout MAX expects, on CPU."""
    # numpy has no bfloat16, so go through float32.
    pixel_values_np = pixel_values.permute(0, 2, 3, 1).cpu().float().numpy()
    patches = np.stack(
        [
            extract_patches_from_image(img, patch_size=patch_size)
            for img in pixel_values_np
        ]
    )
    return torch.from_numpy(patches).to(dtype=torch.bfloat16).contiguous()


def build_max_embeddings(
    internvl_config: InternVLConfig,
    embeddings_weights: dict[str, torch.Tensor],
    input_shape: Sequence[int],
    dtype: DType,
) -> Model:
    """Compiles the MAX InternVisionEmbeddings graph.

    Nothing here touches the device, so `precompile_mefs` can record the
    compile on a CPU worker.
    """
    device_ref = DeviceRef.GPU()
    max_weights = convert_hf_to_max_weights(embeddings_weights)

    embeddings = InternVisionEmbeddings(internvl_config, device=device_ref)
    embeddings.load_state_dict(
        state_dict=max_weights,
        strict=True,
    )

    session = InferenceSession(devices=[Accelerator()])

    input_type = TensorType(dtype, shape=input_shape, device=device_ref)

    with Graph("InternVisionEmbeddings", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = embeddings(x.tensor)
        graph.output(output)

    return session.load(graph, weights_registry=embeddings.state_dict())


def execute_max_embeddings(
    compiled: Model, patched_pixel_values: torch.Tensor, device: Device
) -> torch.Tensor:
    """Runs the compiled MAX InternVisionEmbeddings on patched pixel values."""
    result = compiled.execute(
        Buffer.from_dlpack(patched_pixel_values).to(device)
    )
    return from_dlpack(result[0])
