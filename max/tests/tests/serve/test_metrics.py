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
import pickle
from unittest import mock

import pytest
from max.serve.config import Settings
from max.serve.telemetry import common, metrics
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal.instrument import (
    _ProxyGauge,
    _ProxyHistogram,
    _ProxyInstrument,
)

_meter = get_meter_provider().get_meter("testing")


def test_correct_metric_names() -> None:
    for name, inst in metrics.SERVE_METRICS.items():
        if isinstance(inst, _ProxyInstrument):
            assert name == inst._name
        else:
            assert name == inst.name


def test_max_measurement() -> None:
    m = metrics.MaxMeasurement("maxserve.itl", 1)
    m.commit()


def test_time_per_output_token_measurement() -> None:
    common.configure_metrics(Settings())
    assert "maxserve.time_per_output_token" in metrics.SERVE_METRICS
    m = metrics.MaxMeasurement("maxserve.time_per_output_token", 1.5)
    m.commit()  # Should not raise


def test_serialization() -> None:
    measurements = [
        metrics.MaxMeasurement("maxserve.itl", 1),
        metrics.MaxMeasurement("maxserve.itl", -3.4),
        metrics.MaxMeasurement("maxserve.itl", 1, attributes={"att1": "val"}),
    ]
    for m in measurements:
        b = pickle.dumps(m)
        m2 = pickle.loads(b)

        assert m.instrument_name == m2.instrument_name
        assert m.value == m2.value
        assert m.attributes == m2.attributes
        assert m.time_unix_nano == m2.time_unix_nano


def test_reject_unknown_metric() -> None:
    m = metrics.MaxMeasurement("bogus", 1)
    with pytest.raises(metrics.UnknownMetric):
        m.commit()


def test_instrument_called() -> None:
    common.configure_metrics(Settings())
    itl = metrics.SERVE_METRICS["maxserve.itl"]
    assert isinstance(itl, _ProxyInstrument)
    assert itl._real_instrument is not None
    with mock.patch.object(
        itl._real_instrument, "_measurement_consumer"
    ) as mock_consumer:
        # make _real_instrument None & verify that the measurement does _not_ get consumed
        with mock.patch.object(itl, "_real_instrument", None):
            metrics.MaxMeasurement("maxserve.itl", 1).commit()
            assert mock_consumer.consume_measurement.call_count == 0

        # put things back together and verify that it does get consumed
        metrics.MaxMeasurement("maxserve.itl", 1).commit()
        # make sure the consumer got called
        assert mock_consumer.consume_measurement.call_count == 1


def test_model_load_time_with_component_attribute() -> None:
    """Pins down the ``component`` tag on the model_load_time histogram.

    The model worker records the per-phase startup breakdown (build, compile,
    init, ...) on the same histogram as the untagged model-load aggregate,
    split by the ``component`` tag.
    """
    common.configure_metrics(Settings())
    assert "maxserve.model_load_time" in metrics.SERVE_METRICS

    # Untagged aggregate.
    metrics.MaxMeasurement("maxserve.model_load_time", 1234.5).commit()

    # Per-phase records, tagged by component.
    for component in ("build", "compile", "total"):
        metrics.MaxMeasurement(
            "maxserve.model_load_time",
            100.0,
            attributes={"component": component},
        ).commit()  # Should not raise


def test_batch_execution_time_with_attributes() -> None:
    """Test that batch_execution_time metric works with batch_type attribute."""
    common.configure_metrics(Settings())

    # Test with CE (prefill) batch type
    m_ce = metrics.MaxMeasurement(
        "maxserve.batch_execution_time", 100.5, attributes={"batch_type": "CE"}
    )
    m_ce.commit()  # Should not raise

    # Test with TG (decode) batch type
    m_tg = metrics.MaxMeasurement(
        "maxserve.batch_execution_time", 50.2, attributes={"batch_type": "TG"}
    )
    m_tg.commit()  # Should not raise


def test_tokens_per_request_histograms() -> None:
    """Test that per-request token histogram metrics can be recorded."""
    common.configure_metrics(Settings())

    # Verify metrics exist in SERVE_METRICS
    assert "maxserve.input_tokens_per_request" in metrics.SERVE_METRICS
    assert "maxserve.output_tokens_per_request" in metrics.SERVE_METRICS

    # Test recording input tokens per request
    m_input = metrics.MaxMeasurement("maxserve.input_tokens_per_request", 256)
    m_input.commit()  # Should not raise

    # Test recording output tokens per request
    m_output = metrics.MaxMeasurement("maxserve.output_tokens_per_request", 128)
    m_output.commit()  # Should not raise


def _is_histogram(inst: object) -> bool:
    """True if a SERVE_METRICS instrument is a Histogram.

    Entries in SERVE_METRICS are created at import time (before any real meter
    provider is configured), so they are always proxy instruments.
    """
    return isinstance(inst, _ProxyHistogram)


def test_all_histograms_have_explicit_buckets() -> None:
    """Every histogram must have tuned bucket boundaries.

    The default latency-ms buckets are wrong for non-latency histograms
    (percentages, token counts, throughput, ...), so common.py assigns
    per-metric buckets by exact instrument name. Guard against a new histogram
    being added without a matching bucket View (which would silently fall back
    to the SDK default buckets), and against stale map entries.
    """
    histogram_names = {
        name
        for name, inst in metrics.SERVE_METRICS.items()
        if _is_histogram(inst)
    }
    mapped = set(common.HISTOGRAM_BUCKETS_BY_METRIC)

    missing = histogram_names - mapped
    assert not missing, (
        f"Histograms missing explicit bucket Views: {sorted(missing)}"
    )

    stale = mapped - histogram_names
    assert not stale, (
        f"HISTOGRAM_BUCKETS_BY_METRIC references non-histograms: {sorted(stale)}"
    )


def test_block_level_metrics_are_gauges() -> None:
    """num_used_blocks / num_total_blocks are instantaneous levels (gauges).

    Emitting an absolute level into a counter would sum the levels into a
    meaningless running total, so these must be gauges (LastValue).
    """
    for name in (
        "maxserve.cache.num_used_blocks",
        "maxserve.cache.num_total_blocks",
    ):
        inst = metrics.SERVE_METRICS[name]
        assert isinstance(inst, _ProxyGauge), (
            f"{name} should be a gauge, got {type(inst)}"
        )


def test_disk_block_counters_record() -> None:
    """The disk-tier block transfer counters exist and record without raising."""
    common.configure_metrics(Settings())
    for name in (
        "maxserve.cache.disk_blocks_read",
        "maxserve.cache.disk_blocks_written",
    ):
        assert name in metrics.SERVE_METRICS
        metrics.MaxMeasurement(name, 7).commit()  # Should not raise


def test_batch_metrics_with_batch_type_attribute() -> None:
    """Pins down the ``batch_type`` label on the new graduated batch histograms.
    ``maxserve.batch_prompt_throughput`` is representative of the batch-level
    instruments declared in this PR; the per-instrument plumbing is identical
    so a single test guards the whole class.
    """
    common.configure_metrics(Settings())
    assert "maxserve.batch_prompt_throughput" in metrics.SERVE_METRICS
    metrics.MaxMeasurement(
        "maxserve.batch_prompt_throughput",
        9100.0,
        attributes={"batch_type": "CE"},
    ).commit()
    metrics.MaxMeasurement(
        "maxserve.batch_prompt_throughput",
        4.9,
        attributes={"batch_type": "TG"},
    ).commit()
