# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Placeholder file for any configs (runtime, models, pipelines, etc)
"""

from enum import StrEnum, auto
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class APIType(StrEnum):
    KSERVE = auto()
    OPENAI = auto()


class RunnerType(StrEnum):
    PYTORCH = auto()
    TOKEN_GEN = auto()


class Settings(BaseSettings):
    api_types: list[APIType] = Field(
        description="List of exposed API types.", default=[]
    )
    runner_type: RunnerType = Field(
        description="Type of execution runner.", default=RunnerType.PYTORCH
    )


def api_prefix(settings: Settings, api_type: APIType):
    return "/" + str(api_type) if len(settings.api_types) > 1 else ""
