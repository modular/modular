# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""
MAX serving in Python prototype. Main API server thing.
"""

import argparse
import uvicorn

from fastapi import FastAPI

from max.serve.config import APIType, get_settings
from max.serve.router import kserve_routes, openai_routes

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
}


def create_app():
    settings = get_settings()
    prefix = lambda api_type: (
        str(api_type) if len(settings.api_types) > 1 else ""
    )
    app = FastAPI()
    for api_type in settings.api_types:
        app.include_router(ROUTES[api_type].router, prefix=prefix(api_type))
    return app


def main():
    parser = argparse.ArgumentParser()
    app = create_app()
    uvicorn.run(app)


if __name__ == "__main__":
    main()
