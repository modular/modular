# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import functools

from lm_eval.__main__ import cli_evaluate


def monkeypatch_api_timeout() -> None:
    """Disable timeouts when running API models with num_concurrent>1."""

    import aiohttp
    from lm_eval.models import api_models

    old_client_session = getattr(api_models, "ClientSession")  # noqa: B009

    @functools.wraps(old_client_session)
    def wrapped_client_session(*args, **kwargs):
        kwargs["timeout"] = aiohttp.ClientTimeout(total=None)
        return old_client_session(*args, **kwargs)

    setattr(api_models, "ClientSession", wrapped_client_session)  # noqa: B010


monkeypatch_api_timeout()

if __name__ == "__main__":
    cli_evaluate()
