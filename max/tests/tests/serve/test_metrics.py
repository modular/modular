# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import pickle

import pytest
from max.serve.telemetry import metrics
from opentelemetry.metrics import get_meter_provider  # type: ignore

_meter = get_meter_provider().get_meter("testing")


def test_correct_metric_names():
    for name, inst in metrics.SERVE_METRICS.items():
        assert name == inst.name


def test_max_measurement():
    m = metrics.MaxMeasurement("maxserve.itl", 1)
    m.commit()


def test_serialization():
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


def test_reject_unknown_metric():
    m = metrics.MaxMeasurement("bogus", 1)
    with pytest.raises(metrics.UnknownMetric):
        m.commit()
