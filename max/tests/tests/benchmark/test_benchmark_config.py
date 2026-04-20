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
"""Benchmark config utility functions unit tests"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from max.benchmark.benchmark_shared.config import (
    BaseBenchmarkConfig,
    ServingBenchmarkConfig,
)
from pydantic import Field


class TestServingSweepFields:
    """Test that sweep/workload-related fields exist on ServingBenchmarkConfig."""

    def test_sweep_fields_on_serving_config(self) -> None:
        """Test that sweep-related fields exist on ServingBenchmarkConfig with correct defaults."""
        config = ServingBenchmarkConfig()

        assert config.workload_config is None
        assert config.log_dir is None
        assert config.dry_run is False
        assert config.upload_results is False
        assert config.benchmark_sha is None
        assert config.cluster_information_path is None
        assert config.benchmark_config_name is None
        assert config.metadata == []
        assert config.latency_percentiles == "50,90,95,99"
        assert config.num_iters == 1
        assert config.num_prompts_multiplier is None
        assert config.flush_prefix_cache is True
        assert config.max_concurrency is None
        assert config.request_rate == "inf"

    def test_sweep_config_field_metadata(self) -> None:
        """Test that sweep-related fields have proper json_schema_extra metadata."""
        model_fields = ServingBenchmarkConfig.model_fields

        for name, field_info in model_fields.items():
            extra = field_info.json_schema_extra
            if extra is None or not isinstance(extra, dict):
                continue
            if name in ["workload_config"]:
                assert "group" in extra
                assert extra["group"] == "Workload Configuration"
            elif name in [
                "upload_results",
                "benchmark_sha",
                "cluster_information_path",
                "benchmark_config_name",
            ]:
                assert "group" in extra
                assert extra["group"] == "Result Upload Configuration"
            elif name in [
                "num_iters",
                "flush_prefix_cache",
                "num_prompts_multiplier",
            ]:
                assert "group" in extra
                assert extra["group"] == "Sweep Configuration"
            elif name in ["max_concurrency"]:
                assert "group" in extra
                assert extra["group"] == "Request Configuration"
            elif name in ["request_rate"]:
                assert "group" in extra
                assert extra["group"] == "Traffic Control"


# ===----------------------------------------------------------------------=== #
# ConfigFileModel / cyclopts config-file loading tests
# ===----------------------------------------------------------------------=== #


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestServingConfigFileLoading:
    """Tests for ``--config-file`` loading with ServingBenchmarkConfig.

    These verify that the cyclopts/ConfigFileModel approach works correctly
    after removal of the legacy argparse infrastructure.

    Config YAML format notes
    ------------------------
    ``ServingBenchmarkConfig`` uses ``section_name = "benchmark_config"`` as
    its default, but that default is NOT visible to the ``model_validator``
    because it runs before Pydantic applies defaults.  The section is only
    extracted when ``section_name`` is explicitly supplied at construction time.

    Two valid file formats therefore exist:

    * **Flat** (no section wrapper) — works without passing ``section_name``::

          model: myorg/model
          host: 10.0.0.1

    * **Sectioned** — requires ``section_name="benchmark_config"``::

          benchmark_config:
              model: myorg/model
              host: 10.0.0.1
    """

    # ------------------------------------------------------------------
    # Flat YAML (no section wrapper)
    # ------------------------------------------------------------------

    def test_flat_yaml_loads_model_and_host(self, tmp_path: Path) -> None:
        """Flat YAML values are applied to ServingBenchmarkConfig."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(cfg_path, {"model": "myorg/llama", "host": "10.0.0.1"})

        config = ServingBenchmarkConfig(config_file=str(cfg_path))
        assert config.model == "myorg/llama"
        assert config.host == "10.0.0.1"

    def test_flat_yaml_applies_port_and_backend(self, tmp_path: Path) -> None:
        """Numeric and string fields from flat YAML are applied correctly."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path, {"model": "myorg/llama", "port": 9000, "backend": "vllm"}
        )

        config = ServingBenchmarkConfig(config_file=str(cfg_path))
        assert config.port == 9000
        assert config.backend == "vllm"

    def test_flat_yaml_with_tempfile(self) -> None:
        """config_file works with a real NamedTemporaryFile."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"model": "tmp/model", "port": 8080}, f)
            tmp_name = f.name

        config = ServingBenchmarkConfig(config_file=tmp_name)
        assert config.model == "tmp/model"
        assert config.port == 8080

    # ------------------------------------------------------------------
    # Sectioned YAML (benchmark_config section, explicit section_name)
    # ------------------------------------------------------------------

    def test_sectioned_yaml_with_explicit_section_name(
        self, tmp_path: Path
    ) -> None:
        """Sectioned YAML is parsed correctly when section_name is explicit."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path,
            {
                "benchmark_config": {
                    "model": "myorg/llama",
                    "host": "10.0.0.1",
                    "port": 9000,
                },
                "other_section": {"ignored": True},
            },
        )

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path), section_name="benchmark_config"
        )
        assert config.model == "myorg/llama"
        assert config.host == "10.0.0.1"
        assert config.port == 9000

    def test_missing_section_raises_key_error(self, tmp_path: Path) -> None:
        """A KeyError is raised when the requested section is absent."""
        cfg_path = tmp_path / "config.yaml"
        _write_yaml(cfg_path, {"other_section": {"model": "x"}})

        with pytest.raises(KeyError, match="benchmark_config"):
            ServingBenchmarkConfig(
                config_file=str(cfg_path), section_name="benchmark_config"
            )

    def test_non_dict_yaml_raises_type_error(self, tmp_path: Path) -> None:
        """A TypeError is raised when the YAML root is not a mapping."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("- item1\n- item2\n")

        with pytest.raises(TypeError, match="mapping"):
            ServingBenchmarkConfig(config_file=str(cfg_path))

    # ------------------------------------------------------------------
    # Precedence: explicit kwargs beat config file
    # ------------------------------------------------------------------

    def test_explicit_kwarg_beats_flat_yaml(self, tmp_path: Path) -> None:
        """An explicitly supplied kwarg takes precedence over flat YAML."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path,
            {"model": "file/model", "host": "10.0.0.1", "port": 9000},
        )

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path), model="cli/model"
        )
        assert config.model == "cli/model"
        assert config.host == "10.0.0.1"  # still from file
        assert config.port == 9000  # still from file

    def test_multiple_overrides_beat_config_file(self, tmp_path: Path) -> None:
        """Multiple explicit kwargs all override the corresponding file values."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path, {"model": "file/model", "host": "1.2.3.4", "port": 9000}
        )

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path),
            model="cli/model",
            host="localhost",
        )
        assert config.model == "cli/model"
        assert config.host == "localhost"
        assert config.port == 9000  # not overridden → from file

    # ------------------------------------------------------------------
    # model_fields_set semantics
    # ------------------------------------------------------------------

    def test_model_fields_set_includes_both_cli_and_file_fields(
        self, tmp_path: Path
    ) -> None:
        """Both explicit kwargs and config-file fields appear in model_fields_set.

        ``ConfigFileModel.load_config_file`` is a ``model_validator(mode="before")``
        that injects file values into the input dict before Pydantic validation.
        Pydantic therefore treats them as explicitly supplied, so both the CLI
        kwarg (``model``) and the file value (``host``) end up in
        ``model_fields_set``.
        """
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(cfg_path, {"model": "file/model", "host": "10.0.0.1"})

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path), model="cli/model"
        )
        # Both the explicit kwarg and the file-sourced field are tracked.
        assert "model" in config.model_fields_set
        assert "host" in config.model_fields_set
        # The CLI value still wins over the file value.
        assert config.model == "cli/model"


# ===----------------------------------------------------------------------=== #
# Sweepable Type Tests
# ===----------------------------------------------------------------------=== #


class TestSweepableConfig(BaseBenchmarkConfig):
    """Test configuration class with sweepable_type fields for testing."""

    # Integer sweepable field
    max_concurrency: str | None = Field(
        default=None,
        description="Maximum concurrent requests. Can be a single integer, 'None', or comma-separated string for sweep configs.",
        json_schema_extra={
            "group": "Request Configuration",
            "group_description": "Parameters controlling request concurrency and processing",
            "sweepable_type": "int",
        },
    )

    # Float sweepable field
    request_rate: str | None = Field(
        default="inf",
        description="Requests per second. Can be a single float value or comma-separated string for sweep configs.",
        json_schema_extra={
            "group": "Traffic Control",
            "group_description": "Parameters controlling request rate and traffic patterns",
            "sweepable_type": "float",
        },
    )

    # Non-sweepable field for comparison
    model: str = Field(
        default="test-model",
        description="Model name (not sweepable).",
        json_schema_extra={
            "group": "Model Configuration",
            "group_description": "Model configuration parameters",
        },
    )


def parse_sweepable_values(
    value: str | None, sweepable_type: type
) -> list[int | float | None]:
    """
    Parse comma-separated values according to sweepable_type.

    This function mimics the parsing logic used in sweep_benchmark_serving.py
    for handling sweepable_type fields.

    Args:
        value: Comma-separated string of values to parse
        sweepable_type: Type to parse values as (int or float)

    Returns:
        List of parsed values

    Raises:
        ValueError: If value cannot be parsed as the specified type
    """
    if not value or value.lower() == "none":
        return []

    # Handle "inf" for float types
    if sweepable_type is float and value.lower() == "inf":
        return [float("inf")]

    try:
        # Split by comma and parse each value
        values = [x.strip() for x in value.split(",")]
        parsed_values: list[int | float | None] = []

        for val in values:
            if not val:  # Handle empty strings
                parsed_values.append(None)
            elif val.lower() == "none":
                parsed_values.append(None)
            elif val.lower() == "inf" and sweepable_type is float:
                parsed_values.append(float("inf"))
            else:
                # Handle float values that can be converted to int
                if sweepable_type is int and "." in val:
                    # Convert float string to int (truncates decimal part)
                    parsed_values.append(int(float(val)))
                else:
                    parsed_values.append(sweepable_type(val))

        return parsed_values
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Cannot parse '{value}' as {sweepable_type.__name__}: {e}"
        ) from e


class TestSweepableTypeParsing:
    """Test class for sweepable_type parsing functionality."""

    def test_parse_int_values_basic(self) -> None:
        """Test basic integer parsing for sweepable_type: int fields."""
        # Test single integer
        result = parse_sweepable_values("5", int)
        assert result == [5]

        # Test comma-separated integers
        result = parse_sweepable_values("1,2,3,4,5", int)
        assert result == [1, 2, 3, 4, 5]

        # Test with spaces
        result = parse_sweepable_values("1, 2, 3, 4, 5", int)
        assert result == [1, 2, 3, 4, 5]

    def test_parse_int_values_with_none(self) -> None:
        """Test integer parsing with None values."""
        # Test "None" string
        result = parse_sweepable_values("None", int)
        assert result == []

        # Test mixed with None
        result = parse_sweepable_values("1,None,3", int)
        assert result == [1, None, 3]

        # Test all None
        result = parse_sweepable_values("None,None,None", int)
        assert result == [None, None, None]

    def test_parse_float_values_basic(self) -> None:
        """Test basic float parsing for sweepable_type: float fields."""
        # Test single float
        result = parse_sweepable_values("1.5", float)
        assert result == [1.5]

        # Test comma-separated floats
        result = parse_sweepable_values("1.0,2.5,3.14,4.0", float)
        assert result == [1.0, 2.5, 3.14, 4.0]

        # Test with spaces
        result = parse_sweepable_values("1.0, 2.5, 3.14, 4.0", float)
        assert result == [1.0, 2.5, 3.14, 4.0]

    def test_parse_float_values_with_inf(self) -> None:
        """Test float parsing with infinity values."""
        # Test "inf" string
        result = parse_sweepable_values("inf", float)
        assert result == [float("inf")]

        # Test mixed with inf
        result = parse_sweepable_values("1.0,inf,3.0", float)
        assert result == [1.0, float("inf"), 3.0]

        # Test all inf
        result = parse_sweepable_values("inf,inf,inf", float)
        assert result == [float("inf"), float("inf"), float("inf")]

    def test_parse_float_values_with_none(self) -> None:
        """Test float parsing with None values."""
        # Test "None" string
        result = parse_sweepable_values("None", float)
        assert result == []

        # Test mixed with None
        result = parse_sweepable_values("1.0,None,3.0", float)
        assert result == [1.0, None, 3.0]

    def test_parse_empty_values(self) -> None:
        """Test parsing empty values."""
        # Test empty string
        result = parse_sweepable_values("", int)
        assert result == []

        result = parse_sweepable_values("", float)
        assert result == []

    def test_parse_single_value(self) -> None:
        """Test parsing single values without commas."""
        # Test single integer
        result = parse_sweepable_values("42", int)
        assert result == [42]

        # Test single float
        result = parse_sweepable_values("3.14", float)
        assert result == [3.14]

    def test_parse_whitespace_handling(self) -> None:
        """Test that whitespace is properly handled."""
        # Test leading/trailing spaces
        result = parse_sweepable_values(" 1 , 2 , 3 ", int)
        assert result == [1, 2, 3]

        # Test multiple spaces
        result = parse_sweepable_values("1,  2,   3", int)
        assert result == [1, 2, 3]

    def test_parse_invalid_values(self) -> None:
        """Test parsing invalid values raises appropriate errors."""
        # Test invalid integer
        with pytest.raises(ValueError, match="Cannot parse 'abc' as int"):
            parse_sweepable_values("abc", int)

        # Test invalid float
        with pytest.raises(ValueError, match="Cannot parse 'xyz' as float"):
            parse_sweepable_values("xyz", float)

        # Test mixed valid/invalid
        with pytest.raises(ValueError, match="Cannot parse '1,abc,3' as int"):
            parse_sweepable_values("1,abc,3", int)

    def test_parse_edge_cases(self) -> None:
        """Test edge cases in parsing."""
        # Test zero values
        result = parse_sweepable_values("0,0.0", int)
        assert result == [0, 0]

        result = parse_sweepable_values("0,0.0", float)
        assert result == [0.0, 0.0]

        # Test negative values
        result = parse_sweepable_values("-1,-2.5", int)
        assert result == [-1, -2]

        result = parse_sweepable_values("-1,-2.5", float)
        assert result == [-1.0, -2.5]

        # Test scientific notation
        result = parse_sweepable_values("1e2,2.5e-1", float)
        assert result == [100.0, 0.25]

    def test_parse_large_values(self) -> None:
        """Test parsing large values."""
        # Test large integers
        result = parse_sweepable_values("1000000,2000000", int)
        assert result == [1000000, 2000000]

        # Test large floats
        result = parse_sweepable_values("1e6,2e6", float)
        assert result == [1000000.0, 2000000.0]

    def test_parse_mixed_types_in_string(self) -> None:
        """Test that string contains mixed types that can be parsed."""
        # Test integers that can be parsed as floats
        result = parse_sweepable_values("1,2,3", float)
        assert result == [1.0, 2.0, 3.0]

        # Test floats that can be parsed as integers (truncated)
        result = parse_sweepable_values("1.0,2.0,3.0", int)
        assert result == [1, 2, 3]

    def test_parse_case_insensitive_none_inf(self) -> None:
        """Test that None and inf parsing is case insensitive."""
        # Test case insensitive None
        result = parse_sweepable_values("NONE", int)
        assert result == []

        result = parse_sweepable_values("none", int)
        assert result == []

        result = parse_sweepable_values("None", int)
        assert result == []

        # Test case insensitive inf
        result = parse_sweepable_values("INF", float)
        assert result == [float("inf")]

        result = parse_sweepable_values("inf", float)
        assert result == [float("inf")]

        result = parse_sweepable_values("Inf", float)
        assert result == [float("inf")]

    def test_parse_complex_combinations(self) -> None:
        """Test complex combinations of values."""
        # Test complex integer combination
        result = parse_sweepable_values("1,None,3,None,5", int)
        assert result == [1, None, 3, None, 5]

        # Test complex float combination
        result = parse_sweepable_values("1.0,inf,3.14,None,5.0", float)
        assert result == [1.0, float("inf"), 3.14, None, 5.0]

    def test_parse_duplicate_values(self) -> None:
        """Test parsing duplicate values."""
        # Test duplicate integers
        result = parse_sweepable_values("1,1,1", int)
        assert result == [1, 1, 1]

        # Test duplicate floats
        result = parse_sweepable_values("1.0,1.0,1.0", float)
        assert result == [1.0, 1.0, 1.0]

    def test_parse_single_comma(self) -> None:
        """Test parsing single comma (edge case)."""
        # Test single comma
        result = parse_sweepable_values(",", int)
        assert result == [None, None]  # Empty strings become None

        result = parse_sweepable_values(",", float)
        assert result == [None, None]

    def test_parse_trailing_comma(self) -> None:
        """Test parsing with trailing comma."""
        # Test trailing comma
        result = parse_sweepable_values("1,2,", int)
        assert result == [1, 2, None]  # Empty string becomes None

        result = parse_sweepable_values("1.0,2.0,", float)
        assert result == [1.0, 2.0, None]

    def test_parse_leading_comma(self) -> None:
        """Test parsing with leading comma."""
        # Test leading comma
        result = parse_sweepable_values(",1,2", int)
        assert result == [None, 1, 2]  # Empty string becomes None

        result = parse_sweepable_values(",1.0,2.0", float)
        assert result == [None, 1.0, 2.0]


_SWEEPABLE_TYPE_MAP: dict[str, type] = {"int": int, "float": float}


class TestSweepableTypeIntegration:
    """Integration tests for sweepable_type with actual config classes."""

    @staticmethod
    def _get_extra(cls: type[BaseBenchmarkConfig], name: str) -> dict[str, Any]:
        """Get json_schema_extra dict for a pydantic model field."""
        extra = cls.model_fields[name].json_schema_extra
        return extra if isinstance(extra, dict) else {}

    @staticmethod
    def _get_sweepable_type(
        cls: type[BaseBenchmarkConfig], name: str
    ) -> type | None:
        """Resolve the sweepable_type string to an actual Python type."""
        extra = cls.model_fields[name].json_schema_extra
        if not isinstance(extra, dict):
            return None
        raw = extra.get("sweepable_type")
        return _SWEEPABLE_TYPE_MAP.get(raw) if isinstance(raw, str) else None

    def test_config_field_metadata(self) -> None:
        """Test that config fields have correct sweepable_type metadata."""
        # Test max_concurrency field
        mc_extra = self._get_extra(TestSweepableConfig, "max_concurrency")
        assert "sweepable_type" in mc_extra
        assert mc_extra["sweepable_type"] == "int"

        # Test request_rate field
        rr_extra = self._get_extra(TestSweepableConfig, "request_rate")
        assert "sweepable_type" in rr_extra
        assert rr_extra["sweepable_type"] == "float"

        # Test model field (not sweepable)
        model_extra = self._get_extra(TestSweepableConfig, "model")
        assert "sweepable_type" not in model_extra

    def test_sweepable_type_parsing_integration(self) -> None:
        """Test integration of sweepable_type parsing with config values."""
        # Test with max_concurrency (int type)
        config = TestSweepableConfig(max_concurrency="1,2,3")

        if config.max_concurrency:
            sweepable_type = self._get_sweepable_type(
                TestSweepableConfig, "max_concurrency"
            )
            if sweepable_type:
                parsed_values = parse_sweepable_values(
                    config.max_concurrency, sweepable_type
                )
                assert parsed_values == [1, 2, 3]

        # Test with request_rate (float type)
        config = TestSweepableConfig(request_rate="1.0,2.5,inf")

        if config.request_rate:
            sweepable_type = self._get_sweepable_type(
                TestSweepableConfig, "request_rate"
            )
            if sweepable_type:
                parsed_values = parse_sweepable_values(
                    config.request_rate, sweepable_type
                )
                assert parsed_values == [1.0, 2.5, float("inf")]

    def test_sweepable_type_with_none_values(self) -> None:
        """Test sweepable_type parsing with None values in config."""
        config = TestSweepableConfig(max_concurrency=None)

        if config.max_concurrency:
            sweepable_type = self._get_sweepable_type(
                TestSweepableConfig, "max_concurrency"
            )
            if sweepable_type:
                parsed_values = parse_sweepable_values(
                    config.max_concurrency, sweepable_type
                )
                assert parsed_values == []
        else:
            assert True

    def test_sweepable_type_with_default_values(self) -> None:
        """Test sweepable_type parsing with default values."""
        config = TestSweepableConfig()

        sweepable_type = self._get_sweepable_type(
            TestSweepableConfig, "request_rate"
        )
        if sweepable_type:
            parsed_values = parse_sweepable_values(
                config.request_rate, sweepable_type
            )
            assert parsed_values == [float("inf")]

    def test_sweepable_type_error_handling(self) -> None:
        """Test error handling in sweepable_type parsing."""
        config = TestSweepableConfig(max_concurrency="1,abc,3")

        sweepable_type = self._get_sweepable_type(
            TestSweepableConfig, "max_concurrency"
        )
        if sweepable_type:
            with pytest.raises(
                ValueError, match="Cannot parse '1,abc,3' as int"
            ):
                parse_sweepable_values(config.max_concurrency, sweepable_type)

    def test_sweepable_type_with_real_serving_config(self) -> None:
        """Test sweepable_type parsing with actual ServingBenchmarkConfig fields."""
        config = ServingBenchmarkConfig(
            max_concurrency="1,2,4,8", request_rate="1.0,2.0,4.0,8.0"
        )

        # Test max_concurrency parsing
        sweepable_type = self._get_sweepable_type(
            ServingBenchmarkConfig, "max_concurrency"
        )
        if sweepable_type and config.max_concurrency:
            parsed_values = parse_sweepable_values(
                config.max_concurrency, sweepable_type
            )
            assert parsed_values == [1, 2, 4, 8]

        # Test request_rate parsing
        sweepable_type = self._get_sweepable_type(
            ServingBenchmarkConfig, "request_rate"
        )
        if sweepable_type and config.request_rate:
            parsed_values = parse_sweepable_values(
                config.request_rate, sweepable_type
            )
            assert parsed_values == [1.0, 2.0, 4.0, 8.0]


class TestSweepableTypeValidation:
    """Test validation and error handling for sweepable_type functionality."""

    def test_validate_sweepable_type_metadata(self) -> None:
        """Test that sweepable_type metadata is properly validated."""
        model_fields = TestSweepableConfig.model_fields

        # Check max_concurrency field
        mc_extra = model_fields["max_concurrency"].json_schema_extra
        assert isinstance(mc_extra, dict)
        sweepable_type = mc_extra.get("sweepable_type")
        assert sweepable_type in ["int", "float"]

        # Check request_rate field
        rr_extra = model_fields["request_rate"].json_schema_extra
        assert isinstance(rr_extra, dict)
        sweepable_type = rr_extra.get("sweepable_type")
        assert sweepable_type in ["int", "float"]

    def test_validate_parsed_values_types(self) -> None:
        """Test that parsed values have correct types."""
        # Test integer parsing returns integers
        result = parse_sweepable_values("1,2,3", int)
        assert all(isinstance(x, int) or x is None for x in result)

        # Test float parsing returns floats
        result = parse_sweepable_values("1.0,2.0,3.0", float)
        assert all(isinstance(x, float) or x is None for x in result)

    def test_validate_error_messages(self) -> None:
        """Test that error messages are informative."""
        # Test integer error message
        with pytest.raises(ValueError) as exc_info:
            parse_sweepable_values("abc", int)
        assert "Cannot parse 'abc' as int" in str(exc_info.value)

        # Test float error message
        with pytest.raises(ValueError) as exc_info:
            parse_sweepable_values("xyz", float)
        assert "Cannot parse 'xyz' as float" in str(exc_info.value)

    def test_validate_edge_case_handling(self) -> None:
        """Test that edge cases are handled correctly."""
        # Test empty string
        result = parse_sweepable_values("", int)
        assert result == []

        # Test single comma
        result = parse_sweepable_values(",", int)
        assert result == [None, None]

        # Test multiple commas
        result = parse_sweepable_values(",,,", int)
        assert result == [None, None, None, None]

    def test_validate_infinity_handling(self) -> None:
        """Test that infinity is handled correctly for float types."""
        # Test inf for float
        result = parse_sweepable_values("inf", float)
        assert result == [float("inf")]
        assert all(x == float("inf") for x in result)

        # Test inf should not work for int
        with pytest.raises(ValueError):
            parse_sweepable_values("inf", int)

    def test_validate_none_handling(self) -> None:
        """Test that None is handled correctly."""
        # Test None for both types
        result_int = parse_sweepable_values("None", int)
        result_float = parse_sweepable_values("None", float)

        assert result_int == []
        assert result_float == []

        # Test mixed None values
        result = parse_sweepable_values("1,None,3", int)
        assert result == [1, None, 3]
