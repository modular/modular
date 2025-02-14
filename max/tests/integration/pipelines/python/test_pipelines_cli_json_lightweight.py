# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json

import pytest
from max.entrypoints import pipelines


def test_pipelines_list_json(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(["list", "--json"])
    captured = capsys.readouterr()

    # Parse the JSON output
    output = json.loads(captured.out)

    # Verify the schema structure
    assert "architectures" in output
    assert isinstance(output["architectures"], dict)

    # Check structure of an architecture entry
    for arch_name, arch_data in output["architectures"].items():
        assert isinstance(arch_name, str)
        assert isinstance(arch_data, dict)
        assert "example_repo_ids" in arch_data
        assert isinstance(arch_data["example_repo_ids"], list)
        assert "supported_encodings" in arch_data
        assert isinstance(arch_data["supported_encodings"], list)

        # Check encoding entries
        for encoding in arch_data["supported_encodings"]:
            assert isinstance(encoding, dict)
            assert "encoding" in encoding
            assert "supported_kv_cache_strategies" in encoding
            assert isinstance(encoding["supported_kv_cache_strategies"], list)
