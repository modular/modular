# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import socket

import pytest
from max.serve.config import Settings


def test_setting_throws_occupied_port():
    # Ensure port occupied
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", 8000))
        except OSError as e:
            # It's okay if it's occupied already for some reason.
            pass
        with pytest.raises(ValueError):
            socket_settings = Settings(host="0.0.0.0", port=8000)
