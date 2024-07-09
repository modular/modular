# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: requests
# RUN: %mojo -debug-level full %s

from python import Python
from sys import argv
from testing import assert_true, assert_false
from time import sleep


from max.engine import InferenceSession, InputSpec
from max.tensor import TensorSpec
from max.serve.metrics import (
    TelemetryContext,
    PrometheusMetricsEndPoint,
    Instrument,
    Counter,
    Gauge,
    Histogram,
)


fn test_custom_prometheus[
    T: Instrument, Fn: fn (inout T) capturing -> None
](
    instrument_name: String,
    desc: String,
    units: String,
    expected_msg: List[String],
    custom_attributes: Dict[String, String] = Dict[String, String](),
) raises:
    """Creates a prometheus end-point for custom metrics and creates a custom
    instrument. Operations are performed on the instrument, and we query the
    end-point to make sure that the metric value is as expected.
    """
    var session = InferenceSession()
    var tctx = TelemetryContext(session)
    var end_point = "localhost:9464"
    var metrics_end_point = PrometheusMetricsEndPoint(end_point)
    var initialized = tctx.init_custom_metrics_prometheus_endpoint(
        metrics_end_point
    )
    assert_true(initialized)

    # note otel converts . to _, so we just use a simple name below
    # var instrument_name = "modular_test_gauge_" + str(rand_val)
    var gauge = tctx.create_instrument[T](
        instrument_name, desc, units, custom_attributes
    )
    # Call the user supplied function.
    Fn(gauge)
    tctx.flush()

    var requests = Python.import_module("requests")
    var sess = requests.session()
    var resp = sess.get("http://" + end_point + "/metrics")
    var text = resp.text
    sess.close()
    # print(text)
    for msg in expected_msg:
        assert_true(
            msg[] in str(text),
            String("Could not find entry: ")
            + msg[]
            + " in "
            + desc
            + " with seed ",
        )

    # test second initialization
    var another_end_point = "localhost:9939"
    var another_metrics_end_point = PrometheusMetricsEndPoint(another_end_point)
    var second_init = tctx.init_custom_metrics_prometheus_endpoint(
        another_metrics_end_point
    )
    assert_false(second_init)
    _ = tctx.clearCustomMetricsPrometheusEndpoint()
    _ = tctx^
    _ = session^


fn dictToAttributeString(custom_attributes: Dict[String, String]) -> String:
    var attribute_string = String()
    var first = True
    for e in custom_attributes.items():
        if not first:
            attribute_string = attribute_string + ","
        attribute_string = attribute_string + e[].key + '="' + e[].value + '"'
        first = False
    attribute_string = "{" + attribute_string + "}"
    return attribute_string


fn test_custom_metrics() raises:
    random.seed()
    var rand_val = random.random_ui64(1, 200)

    var custom_attributes = Dict[String, String]()
    custom_attributes["foo_1"] = "bar"
    custom_attributes["foo_2"] = "bar2"

    var attribute_string = dictToAttributeString(custom_attributes)

    # COUNTERS
    @parameter
    fn CounterTest[T: DType](inout counter: Counter[T]):
        var inc = Int64(1).cast[T]()
        for _ in range(rand_val):
            counter.add(inc)

    var instrument_name = "test_modular_counter"
    var expected_msg = List[String](
        instrument_name
        + "_foos_total"
        + attribute_string
        + " "
        + str(rand_val),
    )
    test_custom_prometheus[Counter[DType.uint64], CounterTest[DType.uint64]](
        instrument_name,
        "uint test counter",
        "foos",
        expected_msg,
        custom_attributes,
    )

    rand_val = random.random_ui64(1, 200)
    instrument_name = "test_modular_counter"
    expected_msg = List[String](
        instrument_name
        + "_foos_total"
        + attribute_string
        + " "
        + str(rand_val),
    )
    test_custom_prometheus[Counter[DType.float64], CounterTest[DType.float64]](
        instrument_name,
        "float64 test counter",
        "foos",
        expected_msg,
        custom_attributes,
    )
    # HISTOGRAMS

    @parameter
    fn HistogramTest[T: DType](inout histogram: Histogram[T]):
        for i in range(rand_val):
            var v = Int64(i).cast[T]()
            histogram.record(v)

    instrument_name = "test_modular_histogram"
    expected_msg = List[String](
        instrument_name + '_foos_bucket{le="0"} 1',
        instrument_name + '_foos_bucket{le="+Inf"} ' + str(rand_val),
    )
    test_custom_prometheus[
        Histogram[DType.uint64], HistogramTest[DType.uint64]
    ](
        instrument_name,
        "uint test histogram",
        "foos",
        expected_msg,
    )

    rand_val = random.random_ui64(1, 200)
    instrument_name = "test_modular_histogram"
    expected_msg = List[String](
        instrument_name + '_foos_bucket{le="0"} 1',
        instrument_name + '_foos_bucket{le="+Inf"} ' + str(rand_val),
    )
    test_custom_prometheus[
        Histogram[DType.float64], HistogramTest[DType.float64]
    ](
        instrument_name,
        "float64 test histogram",
        "foos",
        expected_msg,
    )

    # GAUGES

    # For gauges, the seed needs to be greater than 2 since the formula used
    # >     (rand_val * (rand_val - 1) / 2) - rand_val
    # for verification would lead to an invalid value for rand_val<=2.
    rand_val = random.random_ui64(3, 200)

    @parameter
    fn GaugeTest[T: DType](inout gauge: Gauge[T]):
        # we want to test both adds and subs, so we add i, and then sub 1.
        for i in range(rand_val):
            var v = Int64(i).cast[T]()
            gauge.add(v)
        # sub 1
        for _ in range(rand_val):
            var v = Int64(-1).cast[T]()
            gauge.add(v)

    var instrument_name_s = str("test_modular_gauge_") + str(rand_val)
    expected_msg = List[String](
        instrument_name_s
        + "_foos "
        + str((rand_val * (rand_val - 1) / 2) - rand_val),
    )
    test_custom_prometheus[Gauge[DType.int64], GaugeTest[DType.int64]](
        instrument_name_s,
        "test gauge int",
        "foos",
        expected_msg,
    )

    rand_val = random.random_ui64(3, 200)
    instrument_name_s = str("test_modular_gauge") + str(rand_val)
    expected_msg = List[String](
        instrument_name_s
        + "_foos "
        + str((rand_val * (rand_val - 1) / 2) - rand_val),
    )
    test_custom_prometheus[Gauge[DType.float64], GaugeTest[DType.float64]](
        instrument_name_s,
        "float64 test gauge",
        "foos",
        expected_msg,
    )


def main():
    test_custom_metrics()
