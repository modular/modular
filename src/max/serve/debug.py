# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import functools
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable, ClassVar, Union

from fastapi import FastAPI, Request
from pydantic import Field
from pydantic_settings import BaseSettings
from pyinstrument import Profiler
from pyinstrument.renderers.base import FrameRenderer
from pyinstrument.renderers.console import ConsoleRenderer
from pyinstrument.renderers.html import HTMLRenderer
from pyinstrument.renderers.jsonrenderer import JSONRenderer
from pyinstrument.renderers.speedscope import SpeedscopeRenderer

logger = logging.getLogger("max.serve")


class DebugSettings(BaseSettings):
    profiling_enabled: bool = Field(
        description="Enable pyinstrument profiling.", default=False
    )


@dataclass
class ProfileFormatMetadata:
    label: str
    extension: str
    renderer_cls: type[FrameRenderer]


class ProfileFormat(ProfileFormatMetadata, Enum):
    TEXT = ("text", "txt", ConsoleRenderer)
    JSON = ("json", "json", JSONRenderer)
    HTML = ("html", "html", HTMLRenderer)
    SPEEDSCOPE = ("speedscope", "trace", SpeedscopeRenderer)

    @lru_cache
    @staticmethod
    def members():
        return {member.label: member for member in ProfileFormat}

    @classmethod
    def _missing_(cls, value):
        members = cls.members()
        if isinstance(value, str) and value in members:
            return members[value]
        return ProfileFormat.HTML


@dataclass
class ProfileSession:
    DEFAULT_INTERVAL_SECS: ClassVar[float] = 0.001

    request_id: Union[str, None] = None  # Empty when not profiling.

    interval: float = DEFAULT_INTERVAL_SECS
    profile_format: ProfileFormat = field(
        default_factory=lambda: ProfileFormat.HTML
    )

    @classmethod
    def default_profiler(cls):
        # "disabled" is actually the only async_mode that records coroutine
        # frames across all (vs. just one) event loops.
        return Profiler(
            interval=cls.DEFAULT_INTERVAL_SECS, async_mode="disabled"
        )


PROFILE_SESSION_VAR = ContextVar("profile_session", default=ProfileSession())


def profile_in_session():
    return PROFILE_SESSION_VAR.get().request_id != None


def write_profile(profiler: Profiler, session: ProfileSession):
    request_id = session.request_id
    profile_format = session.profile_format

    filename = f"profile.{request_id}.{profile_format.extension}"
    logger.info("Writing profile: %s", filename)
    with open(filename, "w") as out:
        out.write(profiler.output(renderer=profile_format.renderer_cls()))


async def profile_call(profiler: Profiler, call: Callable):
    session = PROFILE_SESSION_VAR.get()
    if not session.request_id:
        # Not currently profiling.
        return await call()

    profiler._interval = session.interval
    profiler.start()
    result = await call()
    profiler.stop()

    write_profile(profiler, session)
    return result


def register_debug(app: FastAPI, settings: DebugSettings):
    if settings.profiling_enabled:
        profiler = ProfileSession.default_profiler()

        @app.middleware("http")
        async def profile_session(request: Request, call_next: Callable):
            params = request.query_params
            profiling = params.get("profile", False)

            if profiling:
                session = PROFILE_SESSION_VAR.get()
                session.request_id = request.state.request_id
                session.profile_format = ProfileFormat(  # type: ignore
                    params.get("profile_format", "html")
                )
                result = await profile_call(
                    profiler, functools.partial(call_next, request)
                )
                session.request_id = None
                return result
            else:
                return await call_next(request)
