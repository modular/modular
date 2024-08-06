# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Placeholder file for any configs (runtime, models, pipelines, etc)
"""

from functools import lru_cache
from enum import StrEnum, auto

from pydantic import Field
from pydantic_settings import BaseSettings


class APIType(StrEnum):
    KSERVE = auto()
    OPENAI = auto()


class RunnerType(StrEnum):
    PYTORCH = auto()


class Settings(BaseSettings):
    api_types: list[APIType] = Field(
        description="List of exposed API types.", default=[]
    )
    runner_type: RunnerType = Field(
        description="Type of execution runner.", default=RunnerType.PYTORCH
    )


def get_settings():
    return Settings()
