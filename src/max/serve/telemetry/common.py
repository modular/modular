# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import os
import platform
import uuid

from opentelemetry.sdk.resources import Resource


otelBaseUrl = "https://telemetry.modular.com:443"


def _getCloudProvider() -> str:
    providers = ["amazon", "google", "microsoft", "oracle"]
    path = "/sys/class/dmi/id/"
    if os.path.isdir(path):
        for idFile in os.listdir(path):
            try:
                with open(idFile, "r") as file:
                    contents = file.read().lower()
                    for provider in providers:
                        if provider in contents:
                            return provider
            except Exception:
                pass
    return ""


resource = Resource.create(
    {
        "event.domain": "serve",
        "telemetry.session": uuid.uuid4().hex,
        "enduser.id": "",
        "os.type": platform.system(),
        "os.version": platform.release(),
        "cpu.description": platform.processor(),
        "cpu.arch": platform.architecture()[0],
        "system.cloud": _getCloudProvider(),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)
