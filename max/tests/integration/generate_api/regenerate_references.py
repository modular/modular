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
"""Regenerates the generate API tests' Hugging Face references on a GPU.

Run it on each GPU kind, for example:

    ./bazelw test --config=remote-b200 \\
        --test_arg=--revision=<sha from hf-repo-lock.tsv> \\
        //max/tests/integration/generate_api:regenerate_references_gemma4_text

The references live in the ``GENERATE_API_DATA`` tarball in S3. The script
prints how far the new reference is from the current one, so a change in the
Hugging Face stack shows up before it lands. It writes a new tarball into the
target's ``test.outputs``, the current one with this run's reference
replaced, and prints the command that uploads it and the new
``GENERATE_API_DATA``. It never uploads anything itself. To regenerate more
than one reference, upload each new tarball and pass its S3 path as
``--base`` to the next run.

FLUX.2-dev's references are older diffusers images in S3
(``FLUX2_GOLDEN_TAR``). Today's transformers 5.12.1 and diffusers 0.37.0 do
not reproduce them, so ``--run flux2_dev`` only renders today's images into
its output directory and prints how far they are from the S3 images.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import diffusers
import numpy as np
import numpy.typing as npt
import torch
import transformers
from _generate_api import (
    CONVERSATIONS,
    DATA_TARBALL_NAME,
    FLUX2_DEV,
    FLUX2_GUIDANCE,
    FLUX2_HEIGHT,
    FLUX2_REQUESTS,
    FLUX2_REVISION,
    FLUX2_SEED,
    FLUX2_STEPS,
    FLUX2_WIDTH,
    GEMMA4,
    GENERATE_API_DATA,
    TEXT_PROMPTS,
    TEXT_STEPS,
    TOP_K,
    Reference,
    Row,
    Step,
    data_dir,
    dump_reference,
    load_flux2_references,
    step_metrics,
    text_messages,
    top_k_step,
    write_data_tarball,
)
from huggingface_hub import snapshot_download
from PIL import Image
from test_common.test_data import MockPixelGenerationRequest
from test_common.torch_utils import run_image_generation

RUNS = ("gemma4_text", "gemma4_multimodal", "flux2_dev")

ATTN_IMPLEMENTATION = "sdpa"


def _gpu_dir() -> str:
    name = torch.cuda.get_device_name(0)
    for kind in ("B200", "MI355"):
        if kind in name:
            return kind.lower()
    raise ValueError(f"The generate API tests have no references for {name}.")


def _greedy(
    model: transformers.PreTrainedModel,
    inputs: transformers.BatchEncoding,
    full_logits: list[npt.NDArray[np.float32]],
) -> tuple[list[int], list[Step]]:
    output = model.generate(
        **inputs,
        max_new_tokens=TEXT_STEPS,
        do_sample=False,
        top_k=None,
        top_p=None,
        output_logits=True,
        return_dict_in_generate=True,
    )
    prompt_ids = [int(i) for i in inputs["input_ids"][0].tolist()]
    steps = []
    for logits in output.logits:
        row = logits[0].float().cpu().numpy()
        full_logits.append(row)
        steps.append(top_k_step(row))
    new_tokens = output.sequences[0][len(prompt_ids) :].tolist()
    assert new_tokens == [step["token"] for step in steps], (
        "generate picked tokens other than the argmax of its logits."
    )
    return prompt_ids, steps


def _rows(
    run: str,
    model: transformers.PreTrainedModel,
    processor: transformers.ProcessorMixin,
    full: list[npt.NDArray[np.float32]],
) -> list[Row]:
    device = model.device
    rows: list[Row] = []
    if run == "gemma4_text":
        for prompt in TEXT_PROMPTS:
            inputs = processor.tokenizer.apply_chat_template(
                text_messages(prompt),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            logits: list[npt.NDArray[np.float32]] = []
            prompt_ids, steps = _greedy(model, inputs, logits)
            full.append(np.stack(logits))
            rows.append(
                {"prompt": prompt, "prompt_ids": prompt_ids, "steps": steps}
            )
        return rows
    for conversation in CONVERSATIONS:
        inputs = processor.apply_chat_template(
            conversation.messages(),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device, dtype=torch.bfloat16)
        logits = []
        prompt_ids, steps = _greedy(model, inputs, logits)
        full.append(np.stack(logits))
        rows.append(
            {
                "image": conversation.image,
                "question": conversation.question,
                "prompt_ids": prompt_ids,
                "steps": steps,
            }
        )
    return rows


def _compare_with_checked_in(
    old: Reference | None,
    reference: Reference,
    full: list[npt.NDArray[np.float32]],
) -> None:
    """Prints how far the new reference is from the current one."""
    if old is None:
        print("No current reference to compare with.")
        return
    if old["revision"] != reference["revision"]:
        print(
            f"WARNING: the revision changed from {old['revision']} to"
            f" {reference['revision']}."
        )
    worst = 0.0
    for index, (old_row, row, logits) in enumerate(
        zip(old["rows"], reference["rows"], full, strict=True)
    ):
        if old_row["prompt_ids"] != row["prompt_ids"]:
            print(f"WARNING: row {index}'s prompt ids changed.")
            continue
        for step, step_logits in zip(old_row["steps"], logits, strict=True):
            worst = max(worst, step_metrics(step, step_logits).kl)
    print(f"Max KL from the current reference: {worst:.3e}")


def _flux2_dev(out: Path, revision: str | None) -> None:
    """Renders today's diffusers images into ``out`` and compares them with
    the golden images the test reads from S3."""
    pipeline = diffusers.DiffusionPipeline.from_pretrained(
        FLUX2_DEV,
        revision=revision or FLUX2_REVISION,
        torch_dtype=torch.bfloat16,
    )
    requests = [
        MockPixelGenerationRequest.from_prompt(
            prompt=request.prompt,
            seed=FLUX2_SEED,
            height=FLUX2_HEIGHT,
            width=FLUX2_WIDTH,
            num_inference_steps=FLUX2_STEPS,
            guidance_scale=FLUX2_GUIDANCE,
        )
        for request in FLUX2_REQUESTS
    ]
    with torch.inference_mode():
        results = run_image_generation(
            pipeline=pipeline,
            device=torch.device("cuda:0"),
            requests=requests,
            num_steps=FLUX2_STEPS,
        )
    references = load_flux2_references()
    for request, result in zip(FLUX2_REQUESTS, results, strict=True):
        pixels = np.clip(np.asarray(result["images"]), 0.0, 1.0)
        Image.fromarray((pixels * 255).round().astype(np.uint8)).save(
            out / f"{request.name}.png"
        )
        difference = np.abs(pixels - references[request.name]).mean() * 255
        print(
            f"{request.name}: mean absolute difference from the golden image"
            f" {difference:.2f}/255"
        )


def _write_tarball(
    out: Path,
    base: Path,
    base_s3_path: str,
    reference_name: str,
    reference: Reference,
) -> None:
    """Writes the base tarball's files, with ``reference_name`` replaced, to
    a new tarball and prints how to upload it."""
    files = {
        path.relative_to(base).as_posix(): path.read_bytes()
        for path in base.rglob("*")
        if path.is_file()
    }
    files[reference_name] = dump_reference(reference).encode()
    sha = write_data_tarball(files, out / DATA_TARBALL_NAME)
    name, version = base_s3_path.split("/")[-4:-2]
    new_version = int(version) + 1
    print(f"Wrote {DATA_TARBALL_NAME} with sha256 {sha}. To use it, upload it:")
    print(
        f"    ./utils/upload-public-bazel-artifact.sh {name} {new_version}"
        f" {DATA_TARBALL_NAME}"
    )
    print("and set GENERATE_API_DATA in _generate_api.py to")
    print(
        "    s3://modular-bazel-artifacts-public/artifacts/"
        f"{name}/{new_version}/{sha}/{DATA_TARBALL_NAME}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "The checkpoint revision. Defaults to the current reference's"
            " revision, or to the cache's main when there is none."
        ),
    )
    parser.add_argument(
        "--base",
        default=GENERATE_API_DATA,
        help="The S3 path of the data tarball to start from.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dump-full-logits",
        action="store_true",
        help="Also save every step's full logits as an .npz file.",
    )
    args = parser.parse_args()
    out = args.output_dir or Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if args.run == "flux2_dev":
        _flux2_dev(out, args.revision)
        return
    base = data_dir(args.base)
    reference_name = f"references/{_gpu_dir()}/{args.run}.json"
    current = base / reference_name
    old: Reference | None = (
        json.loads(current.read_text()) if current.exists() else None
    )
    main_revision = Path(snapshot_download(GEMMA4, local_files_only=True)).name
    revision = args.revision or (old["revision"] if old else main_revision)
    if revision != main_revision:
        print(
            f"WARNING: generating at {revision}, but the cache's main is"
            f" {main_revision}."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        GEMMA4, revision=revision
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        GEMMA4,
        revision=revision,
        dtype=torch.bfloat16,
        device_map=torch.device("cuda:0"),
        attn_implementation=ATTN_IMPLEMENTATION,
    ).eval()
    full: list[npt.NDArray[np.float32]] = []
    with torch.inference_mode():
        rows = _rows(args.run, model, processor, full)
    reference: Reference = {
        "model": GEMMA4,
        "revision": revision,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "top_k": TOP_K,
        "rows": rows,
    }
    _compare_with_checked_in(old, reference, full)
    (out / f"{args.run}.json").write_text(dump_reference(reference))
    _write_tarball(out, base, args.base, reference_name, reference)
    if (
        args.dump_full_logits
        or os.environ.get("GENERATE_API_DUMP_LOGITS") == "1"
    ):
        np.savez_compressed(out / f"{args.run}_full_logits.npz", *full)
    for row in rows:
        tokens = [step["token"] for step in row["steps"]]
        print(repr(processor.decode(tokens, skip_special_tokens=True)))
    print(f"Wrote the {args.run} references to {out}")


if __name__ == "__main__":
    main()
