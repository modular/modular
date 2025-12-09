# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json
from pathlib import Path
from typing import Any

import click
from eval.run import eval_vllm


@click.command()
@click.argument("eval_function")  # Accept eval_vllm as positional argument
@click.option("--model-name", required=True, help="Model name to evaluate")
@click.option("--url", required=True, help="vLLM server URL")
@click.option(
    "--eval-name",
    required=True,
    help="Evaluation task name(s), comma-separated",
)
@click.option(
    "--output-dir", type=click.Path(), required=True, help="Output directory"
)
def main(
    eval_function: str,
    model_name: str,
    url: str,
    eval_name: str,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)  # Convert string to Path object
    # Split comma-separated tasks and strip whitespace
    tasks = [task.strip() for task in eval_name.split(",")]

    for task in tasks:
        eval_vllm(model_name, url, task, output_path)

    # Consolidate all task results into a single JSON file
    consolidated_results: dict[str, Any] = {}

    consolidated_results["model_name"] = model_name

    for task in tasks:
        task_file = output_path / f"{task}.json"
        if task_file.exists():
            try:
                with task_file.open("r") as f:
                    task_results = json.load(f)
                consolidated_results[task] = task_results
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not read results for task '{task}': {e}")
                consolidated_results[task] = {
                    "error": f"Failed to load results: {e}"
                }
        else:
            print(
                f"Warning: Results file not found for task '{task}' at {task_file}"
            )
            consolidated_results[task] = {"error": "Results file not found"}

    # Create a safe filename from the model name
    safe_model_name = (
        model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    )
    consolidated_file = output_path / f"{safe_model_name}.json"

    # Write consolidated results
    with consolidated_file.open("w") as f:
        json.dump(consolidated_results, f, indent=2)

    print(f"Consolidated results written to: {consolidated_file}")
    print(f"Tasks included: {list(consolidated_results.keys())}")

    # Remove original individual JSON files
    for task in tasks:
        task_file = output_path / f"{task}.json"
        if task_file.exists():
            try:
                task_file.unlink()
                print(f"Removed individual file: {task_file}")
            except OSError as e:
                print(f"Warning: Could not remove file '{task_file}': {e}")
        else:
            print(f"Individual file not found (skipping removal): {task_file}")


if __name__ == "__main__":
    main()
