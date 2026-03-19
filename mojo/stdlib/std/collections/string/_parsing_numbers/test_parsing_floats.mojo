def test_float64_parsing_accuracy() raises:
	var parsed = Float64("4.4501363245856945e-308")
	var expected = 4.4501363245856945e-308
	assert parsed == expected, "Float64 parsing failed: got {} expected {}".format(parsed, expected)
