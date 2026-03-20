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
# test_verify_pipelines.py
import os

import click
from verify_pipelines import main as verify_main


@click.command()
@click.option("--pipeline", required=True, help="Pipeline to verify")
@click.option(
    "--override-pipeline-golden-location",
    "override_pipeline_golden_location",
    required=True,
    default=None,
    help="Override pregenerated_golden_path for a pipeline. Format: PIPELINE_NAME:/path/to/golden.tar.gz",
)
@click.option("--devices", default="gpu", help="Devices to run on")
@click.option("--no-aws", "no_aws", is_flag=True, default=False)
@click.option(
    "--find-tolerances", "find_tolerances", is_flag=True, default=False
)
def main(
    pipeline: str,
    devices: str,
    override_pipeline_golden_location: str,
    no_aws: bool,
    find_tolerances: bool,
) -> None:
    test_undeclared_outputs_dir = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR")
    if test_undeclared_outputs_dir is not None:
        print(f"TEST_UNDECLARED_OUTPUTS_DIR: {test_undeclared_outputs_dir}")

    report_txt_path = os.path.join(
        str(test_undeclared_outputs_dir), "report.txt"
    )
    verdicts_json_path = os.path.join(
        str(test_undeclared_outputs_dir), "verdicts.json"
    )

    args = [
        "--pipeline",
        pipeline,
        "--devices",
        devices,
        "--override-pipeline-golden-location",
        override_pipeline_golden_location,
        "--report",
        report_txt_path,
        "--store-verdicts-json",
        verdicts_json_path,
    ]
    if find_tolerances:
        args.append("--find-tolerances")
    verify_main(args, standalone_mode=True)


if __name__ == "__main__":
    main()
