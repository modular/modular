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
import click
from verify_pipelines import main as verify_main


@click.command()
@click.option("--pipeline", required=True, help="Pipeline to verify")
@click.option("--devices", default="gpu", help="Devices to run on")
@click.option("--no-aws", "no_aws", is_flag=True, default=False)
@click.option(
    "--find-tolerances", "find_tolerances", is_flag=True, default=False
)
def main(
    pipeline: str, devices: str, no_aws: bool, find_tolerances: bool
) -> None:
    args = ["--pipeline", pipeline, "--devices", devices]
    if no_aws:
        args.append("--no-aws")
    if find_tolerances:
        args.append("--find-tolerances")

    verify_main(args, standalone_mode=True)


if __name__ == "__main__":
    main()
