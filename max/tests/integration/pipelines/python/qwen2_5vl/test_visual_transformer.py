# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    generate_attention_mask,
    get_rope_index,
    get_window_index,
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionWindowSdpaAttention,
)
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6

torch.manual_seed(0)


def torch_mrope_pos_ids_3d(
    grid_thw: torch.tensor, spatial_merge_size: int
) -> torch.tensor:
    """Calculate the 3D rope index based on image and video's temporal, height and width in LLM."""
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    return pos_ids


def torch_get_window_index(
    grid_thw: torch.tensor,
    window_size: int,
    spatial_merge_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> tuple[torch.tensor, list]:
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h, llm_grid_w = (
            grid_h // spatial_merge_size,
            grid_w // spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = (
            seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        )
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
    window_index = torch.cat(window_index, dim=0)

    return window_index, cu_window_seqlens


def torch_generate_attention_mask(
    grid_thw: torch.tensor, seq_length: int, cu_window_seqlens: list
) -> tuple[torch.tensor, torch.tensor]:
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=grid_thw.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    attention_mask_cu_seqlens = torch.zeros(
        [1, seq_length, seq_length], device=grid_thw.device, dtype=torch.bool
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask_cu_seqlens[
            ...,
            cu_seqlens[i - 1] : cu_seqlens[i],
            cu_seqlens[i - 1] : cu_seqlens[i],
        ] = True

    attention_mask_cu_window_seqlens = torch.zeros(
        [1, seq_length, seq_length], device=grid_thw.device, dtype=torch.bool
    )
    for i in range(1, len(cu_window_seqlens)):
        attention_mask_cu_window_seqlens[
            ...,
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
        ] = True

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    attention_mask_cu_seqlens = torch.zeros(
        (1, seq_length, seq_length)
    ).masked_fill_(attention_mask_cu_seqlens.logical_not(), fill_val)
    attention_mask_cu_window_seqlens = torch.zeros(
        (1, seq_length, seq_length)
    ).masked_fill_(attention_mask_cu_window_seqlens.logical_not(), fill_val)

    return attention_mask_cu_seqlens, attention_mask_cu_window_seqlens


def torch_get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    tokens_per_second: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        assert attention_mask is not None
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],  # type: ignore
                        image_grid_thw[image_index][1],  # type: ignore
                        image_grid_thw[image_index][2],  # type: ignore
                    )
                    second_per_grid_t = 0.0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],  # type: ignore
                        video_grid_thw[video_index][1],  # type: ignore
                        video_grid_thw[video_index][2],  # type: ignore
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(
                    -1, llm_grid_h * llm_grid_w
                )

                time_tensor = (
                    expanded_range * second_per_grid_t * tokens_per_second
                )

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas,
            device=input_ids.device,
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:  # either no text tokens or no images/videos
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0)
                .expand(3, -1, -1)
                .to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = (
                max_position_ids + 1 - attention_mask.shape[-1]
            )
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)  # type: ignore
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)  # type: ignore
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],  # type: ignore
                device=input_ids.device,  # type: ignore
                dtype=input_ids.dtype,  # type: ignore
            )

        return position_ids, mrope_position_deltas


def test_pos_ids():
    image_grid_thw = torch.tensor([[1, 98, 146], [1, 22, 28]])
    spatial_merge_size = 2
    expected_pos_ids = torch_mrope_pos_ids_3d(
        image_grid_thw, spatial_merge_size
    )
    actual_pos_ids = mrope_pos_ids_3d(
        image_grid_thw.detach().numpy(), spatial_merge_size
    )
    np.testing.assert_array_equal(
        actual_pos_ids,
        expected_pos_ids.detach().numpy(),
    )


def test_window_index():
    image_grid_thw = torch.tensor([[1, 98, 146], [1, 22, 28]])
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    expected_window_index, expected_cu_window_seqlens = torch_get_window_index(
        image_grid_thw,
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )
    actual_window_index, actual_cu_window_seqlens = get_window_index(
        image_grid_thw.detach().numpy(),
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )
    np.testing.assert_array_equal(
        actual_window_index,
        expected_window_index.detach().numpy(),
    )

    np.testing.assert_array_equal(
        actual_cu_window_seqlens,
        np.array(expected_cu_window_seqlens),
    )


def test_generate_attention_mask():
    image_grid_thw = torch.tensor([[1, 98, 146], [1, 22, 28]])
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14
    seq_length = 14308
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    _, cu_window_seqlens = torch_get_window_index(
        image_grid_thw,
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )
    actual_attention_mask_full, actual_attention_mask_window = (
        generate_attention_mask(
            image_grid_thw.detach().numpy(),
            seq_length,
            np.array(cu_window_seqlens),
        )
    )
    attention_mask_full, attention_mask_window = torch_generate_attention_mask(
        image_grid_thw, seq_length, cu_window_seqlens
    )

    np.testing.assert_allclose(
        attention_mask_full.detach().numpy(),
        actual_attention_mask_full,
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )

    np.testing.assert_allclose(
        attention_mask_window.detach().numpy(),
        actual_attention_mask_window,
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


@pytest.mark.skip(reason="Currently failing with error 1e-2. Still debugging.")
def test_scaled_dot_product_attention_vision_mask():
    # batch_size, Target seq_len, embed_dim
    q = torch.rand(16, 14308, 80, dtype=torch.float32)
    k = torch.rand(16, 14308, 80, dtype=torch.float32)
    v = torch.rand(16, 14308, 80, dtype=torch.float32)
    # batch_size, target seq_len, source seq_len
    attention_mask = torch.bernoulli(torch.full((1, 14308, 14308), 0.5)).to(
        torch.bool
    )
    dim = 1280
    n_heads = 16
    expected_attn_output = F.scaled_dot_product_attention(
        q, k, v, attention_mask, dropout_p=0.0
    )

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    max_attention_mask = torch.zeros(
        attention_mask.shape, dtype=torch.float32
    ).masked_fill_(attention_mask.logical_not(), fill_val)

    session = InferenceSession()
    with Graph(
        "visual",
        input_types=[
            TensorType(DType.float32, (q.shape), device=DeviceRef.CPU()),
            TensorType(DType.float32, (k.shape), device=DeviceRef.CPU()),
            TensorType(DType.float32, (v.shape), device=DeviceRef.CPU()),
            TensorType(
                DType.float32, (attention_mask.shape), device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        attn_output = VisionWindowSdpaAttention.scaled_dot_product_attention(
            xq=graph.inputs[0].tensor,
            xk=graph.inputs[1].tensor,
            xv=graph.inputs[2].tensor,
            attention_mask=graph.inputs[3].tensor,
            dim=dim,
            n_heads=n_heads,
        )

        graph.output(attn_output)
        compiled = session.load(graph)
        max_graph_output = compiled.execute(q, k, v, max_attention_mask)[0]
        assert isinstance(max_graph_output, Tensor)

        # Compare results.
        np.testing.assert_allclose(
            max_graph_output.to_numpy(),
            expected_attn_output.detach().numpy(),
            equal_nan=True,
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )


def test_get_rope_index():
    """tests get_rope_index for Qwen2.5VL
    hyper-paramewters used are from Qwen2.5VL config at
    (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json)
    """
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    tokens_per_second = 2

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {
                    "type": "image",
                    "image": "https://download.samplelib.com/jpeg/sample-city-park-400x300.jpg",
                },
                {
                    "type": "video",
                    "video": "https://download.samplelib.com/mp4/sample-5s.mp4",
                },
                # {"type": "video", "video": "https://download.samplelib.com/mp4/sample-10s.mp4"},
                {"type": "text", "text": "Describe this image"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=inputs[0],
        videos=inputs[1],
        padding=True,
        return_tensors="pt",
    )

    expected_position_ids, expected_mrope_position_deltas = (
        torch_get_rope_index(
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            tokens_per_second,
            input_ids=inputs["input_ids"],  # type: ignore
            image_grid_thw=inputs["image_grid_thw"],  # type: ignore
            video_grid_thw=inputs["video_grid_thw"],  # type: ignore
            second_per_grid_ts=inputs["second_per_grid_ts"],  # type: ignore
            attention_mask=inputs["attention_mask"],  # type: ignore
        )
    )

    actual_position_ids, actual_mrope_position_deltas = get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second,
        input_ids=inputs["input_ids"].numpy(),  # type: ignore
        image_grid_thw=inputs["image_grid_thw"].numpy(),  # type: ignore
        video_grid_thw=inputs["video_grid_thw"].numpy(),  # type: ignore
        second_per_grid_ts=np.array(inputs["second_per_grid_ts"]),  # type: ignore
        attention_mask=inputs["attention_mask"].numpy(),  # type: ignore
    )

    np.testing.assert_array_equal(
        actual_position_ids,
        expected_position_ids.detach().numpy(),
    )

    np.testing.assert_array_equal(
        actual_mrope_position_deltas,
        expected_mrope_position_deltas.detach().numpy(),
    )
