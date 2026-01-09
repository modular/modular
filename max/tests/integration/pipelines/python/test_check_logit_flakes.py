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
from check_logit_flakes import (
    OverallStatus,
    VerificationStatus,
    determine_overall_status,
)


def test_empty_top_level_dict() -> None:
    """Test that an empty top-level dict returns INFRA_ERROR."""
    verdicts: dict[str, dict[str, VerificationStatus]] = {}
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_empty_second_level_dict() -> None:
    """Test that an empty second-level dict returns INFRA_ERROR."""
    verdicts: dict[str, dict[str, VerificationStatus]] = {"file1.json": {}}
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_multiple_files_one_empty() -> None:
    """Test that any empty second-level dict returns INFRA_ERROR."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.OK},
        "file2.json": {},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_error_status() -> None:
    """Test that ERROR status returns VERIFICATION_FAILURE."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.ERROR},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_invalid_status() -> None:
    """Test that INVALID status returns VERIFICATION_FAILURE."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.INVALID},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_error_takes_precedence_over_ok() -> None:
    """Test that ERROR status takes precedence over OK."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.ERROR,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_error_takes_precedence_over_flake() -> None:
    """Test that ERROR status takes precedence over FLAKE."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.FLAKE,
            "model2": VerificationStatus.ERROR,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_error_takes_precedence_over_infra() -> None:
    """Test that ERROR status takes precedence over INFRA."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.INFRA,
            "model2": VerificationStatus.ERROR,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_flake_status() -> None:
    """Test that FLAKE status returns VERIFICATION_FLAKE."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.FLAKE},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FLAKE


def test_flake_takes_precedence_over_ok() -> None:
    """Test that FLAKE status takes precedence over OK."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.FLAKE,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FLAKE


def test_infra_takes_precedence_over_flake() -> None:
    """Test that INFRA status takes precedence over FLAKE."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.INFRA,
            "model2": VerificationStatus.FLAKE,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_infra_status() -> None:
    """Test that INFRA status returns INFRA_ERROR."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.INFRA},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_infra_takes_precedence_over_ok() -> None:
    """Test that INFRA status takes precedence over OK."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.INFRA,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_all_ok() -> None:
    """Test that all OK statuses return OK."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.OK},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.OK


def test_all_ok_multiple_models() -> None:
    """Test that all OK statuses across multiple models return OK."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.OK,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.OK


def test_all_ok_multiple_files() -> None:
    """Test that all OK statuses across multiple files return OK."""
    verdicts = {
        "file1.json": {"model1": VerificationStatus.OK},
        "file2.json": {"model2": VerificationStatus.OK},
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.OK


def test_priority_order_error_over_all() -> None:
    """Test complete priority order with ERROR at top."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.ERROR,
            "model3": VerificationStatus.FLAKE,
            "model4": VerificationStatus.INFRA,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_priority_order_infra_over_flake_and_ok() -> None:
    """Test priority order with INFRA over FLAKE and OK."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.FLAKE,
            "model3": VerificationStatus.INFRA,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_priority_order_infra_over_ok() -> None:
    """Test priority order with INFRA over OK."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.INFRA,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.INFRA_ERROR


def test_mixed_error_and_invalid() -> None:
    """Test that mixed ERROR and INVALID both return VERIFICATION_FAILURE."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.ERROR,
            "model2": VerificationStatus.INVALID,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FAILURE


def test_multiple_files_mixed_statuses() -> None:
    """Test complex scenario with multiple files and mixed statuses."""
    verdicts = {
        "file1.json": {
            "model1": VerificationStatus.OK,
            "model2": VerificationStatus.OK,
        },
        "file2.json": {
            "model3": VerificationStatus.OK,
        },
        "file3.json": {
            "model4": VerificationStatus.FLAKE,
        },
    }
    result = determine_overall_status(verdicts)
    assert result == OverallStatus.VERIFICATION_FLAKE
