# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import socket

import pytest
from max.serve.api_server import (
    validate_port_is_free,
)


def test_setting_throws_occupied_port() -> None:
    # Ensure port occupied
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", 8000))
        except OSError as e:
            # It's okay if it's occupied already for some reason.
            pass
        with pytest.raises(ValueError):
            validate_port_is_free(8000)
