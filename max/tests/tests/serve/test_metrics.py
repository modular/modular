# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import pickle
from unittest import mock

import pytest
from max.serve.config import Settings
from max.serve.telemetry import common, metrics
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal.instrument import _ProxyInstrument

_meter = get_meter_provider().get_meter("testing")


def test_correct_metric_names():
    for name, inst in metrics.SERVE_METRICS.items():
        if isinstance(inst, _ProxyInstrument):
            assert name == inst._name
        else:
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


def test_instrument_called():
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
