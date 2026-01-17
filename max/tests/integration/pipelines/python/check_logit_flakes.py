# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
import enum
import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


class VerificationStatus(enum.Enum):
    OK = "ok"
    INVALID = "invalid"
    ERROR = "error"
    FLAKE = "flake"
    INFRA = "infra"


class OverallStatus(enum.Enum):
    OK = "ok"
    INFRA_ERROR = "infra-error"
    VERIFICATION_FLAKE = "verification-flake"
    VERIFICATION_FAILURE = "verification-failure"


def determine_overall_status(
    verdicts: dict[str, dict[str, VerificationStatus]],
) -> OverallStatus:
    """Determine the overall verification status from individual model verdicts.

    Args:
        verdicts: Dictionary mapping file names to dictionaries of model names
                  and their verification statuses.

    Returns:
        The overall status based on the priority rules.
    """
    # If the top-level dict is empty, that's an infra error
    if not verdicts:
        logger.error(
            "INFRA_ERROR: Top-level verdicts dictionary is empty. "
            "No verification results were found."
        )
        return OverallStatus.INFRA_ERROR

    # Collect all statuses for analysis
    error_or_invalid_statuses = []
    flake_statuses = []
    infra_statuses = []
    ok_statuses = []

    for filename, models in verdicts.items():
        for model_name, status in models.items():
            match status:
                case VerificationStatus.ERROR | VerificationStatus.INVALID:
                    error_or_invalid_statuses.append(
                        (filename, model_name, status)
                    )
                case VerificationStatus.INFRA:
                    infra_statuses.append((filename, model_name, status))
                case VerificationStatus.FLAKE:
                    flake_statuses.append((filename, model_name, status))
                case VerificationStatus.OK:
                    ok_statuses.append((filename, model_name, status))
                case _:
                    # This should never happen with proper typing, but just in case
                    logger.error(
                        f"INFRA_ERROR: Unknown status '{status}' for model '{model_name}' "
                        f"in file '{filename}'. This indicates a bug in the verification system."
                    )
                    return OverallStatus.INFRA_ERROR

    # If any VerificationStatus is error or invalid, that's a verification failure
    if error_or_invalid_statuses:
        logger.error(
            f"VERIFICATION_FAILURE: Found {len(error_or_invalid_statuses)} model(s) "
            f"with ERROR or INVALID status:"
        )
        for filename, model_name, status in error_or_invalid_statuses:
            logger.error(
                f"  - Model '{model_name}' in file '{filename}': {status.value}"
            )
        return OverallStatus.VERIFICATION_FAILURE

    # Check for empty second-level dicts; this indicates something went so
    # wrong that a verdict JSON could not be generated
    for filename, models in verdicts.items():
        if not models:
            logger.error(
                f"INFRA_ERROR: File '{filename}' has an empty models dictionary. "
                f"Expected at least one model verification result."
            )
            return OverallStatus.INFRA_ERROR

    # If any VerificationStatus is infra, that's an infra error
    if infra_statuses:
        logger.error(
            f"INFRA_ERROR: Found {len(infra_statuses)} model(s) with INFRA status:"
        )
        for filename, model_name, status in infra_statuses:
            logger.error(
                f"  - Model '{model_name}' in file '{filename}': {status.value}"
            )
        return OverallStatus.INFRA_ERROR

    # If any VerificationStatus is flake, that's a verification flake
    if flake_statuses:
        logger.warning(
            f"VERIFICATION_FLAKE: Found {len(flake_statuses)} model(s) with FLAKE status:"
        )
        for filename, model_name, status in flake_statuses:
            logger.warning(
                f"  - Model '{model_name}' in file '{filename}': {status.value}"
            )
        return OverallStatus.VERIFICATION_FLAKE

    # All statuses should be OK at this point
    if ok_statuses:
        logger.info(
            f"OK: All {len(ok_statuses)} model verification(s) passed successfully:"
        )
        for filename, model_name, status in ok_statuses:
            logger.info(
                f"  - Model '{model_name}' in file '{filename}': {status.value}"
            )
        return OverallStatus.OK

    # This should never be reached, but if somehow no statuses were found,
    # that's an infra error
    logger.error(
        "INFRA_ERROR: No verification statuses found in any model dictionaries. "
        "This should have been caught earlier by the empty dict check."
    )
    return OverallStatus.INFRA_ERROR


@click.command()
@click.option(
    "--verdicts-dir",
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=Path
    ),
    help="Directory containing JSON verdict files",
)
def main(verdicts_dir: Path) -> None:
    """Check logit verification results and determine overall status.

    Reads JSON files from the verdicts directory and determines the overall
    verification status based on individual model results.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Read all JSON files in the directory
    verdicts: dict[str, dict[str, VerificationStatus]] = {}
    json_files = list(verdicts_dir.glob("*.json"))

    logger.info(f"Found {len(json_files)} JSON file(s) in '{verdicts_dir}'")

    for json_file in json_files:
        logger.info(f"Reading file: {json_file.name}")
        try:
            with open(json_file) as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid format in file '{json_file.name}': "
                    f"Expected dict at top level, got {type(data)}"
                )

            # Parse the JSON structure and convert status strings to VerificationStatus
            models: dict[str, VerificationStatus] = {}
            for model_name, model_data in data.items():
                if not isinstance(model_data, dict):
                    raise ValueError(
                        f"Invalid format in file '{json_file.name}': "
                        f"Expected dict for model '{model_name}', got {type(model_data)}"
                    )

                status_str = model_data.get("status")
                if not status_str:
                    raise ValueError(
                        f"Missing 'status' field for model '{model_name}' "
                        f"in file '{json_file.name}'"
                    )

                try:
                    status = VerificationStatus(status_str)
                    models[model_name] = status
                except ValueError as e:
                    raise ValueError(
                        f"Unknown status value '{status_str}' for model '{model_name}' "
                        f"in file '{json_file.name}'"
                    ) from e

            verdicts[json_file.name] = models

        except json.JSONDecodeError:
            logger.exception(f"Failed to parse JSON file '{json_file.name}'")
            raise
        except Exception:
            logger.exception(f"Error reading file '{json_file.name}'")
            raise

    # Determine overall status
    overall_status = determine_overall_status(verdicts)

    # Print verdict to stdout (for parsing by CI systems)
    print(f"verdict={overall_status.value}")


if __name__ == "__main__":
    main()
