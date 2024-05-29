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
from tensor import TensorSpec
from max.serve.metrics import TelemetryContext, PrometheusMetricsEndPoint


fn test_custom_counter_prometheus[T: DType]() raises:
    """Creates a prometheus end-point for custom metrics and creates a custom
    counter. Counter value is incremented, and we query the end-point to make
    sure that the metric value is as expected.
    """
    random.seed()
    var rand_val = random.random_ui64(1, 200)
    var session = InferenceSession()
    var tctx = TelemetryContext(session)
    var end_point = "localhost:9464"
    var metrics_end_point = PrometheusMetricsEndPoint(end_point)
    var initialized = tctx.init_custom_metrics_prometheus_endpoint(
        metrics_end_point
    )
    assert_true(initialized)

    # note otel converts . to _, so we just use a simple name below
    var counter_name = "modular_test_counter_" + str(rand_val)
    var counter = tctx.create_counter[T](
        counter_name, "test counter from mojo", "foos"
    )
    var inc = Int64(1).cast[T]()
    for _ in range(rand_val):
        counter.add(inc)
    tctx.flush()

    var requests = Python.import_module("requests")
    var sess = requests.session()
    var resp = sess.get("http://" + end_point + "/metrics")
    var text = resp.text
    sess.close()

    var expected_msg = List[String](
        counter_name + "_foos_total " + str(rand_val),
    )
    for msg in expected_msg:
        assert_true(
            msg[] in str(text), String("Could not find entry: ") + msg[]
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


fn test_custom_histogram_prometheus[T: DType]() raises:
    """Creates a prometheus end-point for custom metrics and creates a custom
    counter. Counter value is incremented, and we query the end-point to make
    sure that the metric value is as expected.
    """
    random.seed()
    var rand_val = random.random_ui64(1, 200)
    var session = InferenceSession()
    var tctx = TelemetryContext(session)
    var end_point = "localhost:9464"
    var metrics_end_point = PrometheusMetricsEndPoint(end_point)
    var initialized = tctx.init_custom_metrics_prometheus_endpoint(
        metrics_end_point
    )
    assert_true(initialized)

    # note otel converts . to _, so we just use a simple name below
    var counter_name = "modular_test_histogram_" + str(rand_val)
    var counter = tctx.create_histogram[T](
        counter_name, "test histogram from mojo", "foos"
    )
    for i in range(rand_val):
        var v = Int64(i).cast[T]()
        counter.record(v)
    tctx.flush()

    var requests = Python.import_module("requests")
    var sess = requests.session()
    var resp = sess.get("http://" + end_point + "/metrics")
    var text = resp.text
    sess.close()

    # check both upper and lower bounds
    var expected_msg = List[String](
        counter_name + '_foos_bucket{le="0"} 1',
        counter_name + '_foos_bucket{le="+Inf"} ' + str(rand_val),
    )
    # print(text)
    for msg in expected_msg:
        assert_true(
            msg[] in str(text), String("Could not find entry: ") + msg[]
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


fn main() raises:
    test_custom_counter_prometheus[DType.uint64]()
    test_custom_counter_prometheus[DType.float64]()
    test_custom_histogram_prometheus[DType.uint64]()
    test_custom_histogram_prometheus[DType.float64]()
