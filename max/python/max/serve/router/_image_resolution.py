# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Resolve and validate image references from chat-completion requests.

Turns the ``image_url`` / ``video_url`` references in an OpenAI request into raw
image bytes (from ``http(s):``, ``data:``, or ``file:`` URIs) and fully decodes
them once for validation. Kept in its own module so the image-resolution
concern stays focused and out of the much larger ``openai_routes`` handler.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import logging
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol
from urllib.parse import unquote, urlparse

import aiofiles
import httpx
from httpx import (
    AsyncClient,
    HTTPStatusError,
    Response,
    Timeout,
    TimeoutException,
    TransportError,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.tokenizer import ALLOWED_IMAGE_FORMATS
from max.serve.config import Settings
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch
from max.support.human_readable_formatter import to_human_readable_bytes
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl, ValidationError

logger = logging.getLogger("max.serve")

# Which ``maxserve.media.*`` ``source`` a reference's scheme resolves under.
# ``url`` rather than ``fetch``: the orchestrator tier tags the same split
# ``url``, and one dashboard has to read both.
_MEDIA_SOURCE_BY_SCHEME = {
    "http": "url",
    "https": "url",
    "data": "inline",
    "file": "file",
}

# The bounded value set of the ``format`` label. ``Image.open`` is pinned to
# ``ALLOWED_IMAGE_FORMATS``, and PIL reports ``MPO`` for a multi-picture JPEG
# opened through the JPEG factory, so this list plus ``unknown`` is the whole
# set -- nothing here is attacker-controllable. Lower-cased to match the
# spelling the orchestrator and Mach use for the same label.
_METRIC_IMAGE_FORMATS = frozenset(
    fmt.lower() for fmt in (*ALLOWED_IMAGE_FORMATS, "MPO")
)


@dataclass(frozen=True, slots=True)
class _ImageFact:
    """What one image cost at admission, for the ``maxserve.media.*`` family.

    Collected rather than recorded in place because the decode runs in a
    worker thread and the metric clients are not all thread-safe; the frame
    that owns the list records them on the event loop. ``pixels`` is absent
    only when the header itself would not parse, and ``decode_ms`` only when
    the pixel decode was skipped or never reached.
    """

    image_format: str
    pixels: int | None = None
    decode_ms: float | None = None
    rejection: str | None = None


def _record_fact(facts: list[_ImageFact] | None, fact: _ImageFact) -> None:
    """Append ``fact`` when a caller asked for facts, and nothing otherwise."""
    if facts is not None:
        facts.append(fact)


def emit_media_facts(facts: list[_ImageFact]) -> None:
    """Record one request's per-image media facts, on the event loop.

    Shared by both routers: the chat path collects facts around its batch
    decode, the responses path around the single image it inlines. A route
    that decoded an image without recording its facts leaves the per-format
    decode cost computed over a denominator that silently excludes it.
    """
    for fact in facts:
        if fact.pixels is not None:
            METRICS.media_image_size(fact.pixels, fact.image_format)
        if fact.decode_ms is not None:
            METRICS.media_image_decodes(fact.image_format)
            METRICS.media_image_decode_ms(fact.decode_ms, fact.image_format)
        if fact.rejection is not None:
            METRICS.media_rejections(fact.rejection)


def _metric_image_format(image: Image.Image) -> str:
    """The ``format`` label for an opened image, held to the bounded set."""
    name = (image.format or "").lower()
    return name if name in _METRIC_IMAGE_FORMATS else "unknown"


def _reject_media(
    reason: str, message: str, *, error: type[ValueError] = InputError
) -> NoReturn:
    """Count a media rejection under ``reason``, then raise it as a 400.

    Every bucket in the taxonomy is raised on the event loop, so the counter
    is recorded at the raise rather than recovered from the message text by a
    handler further up. ``error`` selects the exception the caller already
    raised, since the route maps ``InputError`` and bare ``ValueError`` to
    different response bodies.
    """
    METRICS.media_rejections(reason)
    raise error(message) from None


# SSRF protection for client-supplied media URLs: validate the host, reject
# internal/reserved addresses, and pin to the resolved IP. See
# ``_validate_and_pin``.

_ALLOWED_SCHEMES = ("http", "https")

# Explicit cloud-metadata endpoints. Most are already caught by the link-local /
# private predicates below, but list them so the intent is auditable and so a
# predicate gap can never silently expose them.
_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure IMDS
        ipaddress.ip_address("169.254.170.2"),  # AWS ECS task metadata
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDS over IPv6
    }
)

# Non-routable ranges that ``ipaddress``'s predicates miss/misreport across
# Python versions: 100.64.0.0/10 (RFC6598 CGNAT, the default in-cluster pod/node
# CIDR on EKS/GKE) and fec0::/10 (deprecated IPv6 site-local).
_EXTRA_BLOCKED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fec0::/10"),
)

# Cap on redirect hops we follow, each re-validated. Media fetches need only a
# hop or two; keep it low to bound re-validation work and egress.
_MAX_REDIRECTS = 4


def _ip_is_blocked(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Whether an address must never be fetched from (an SSRF target).

    Conservative by construction: blocks the union of private, loopback,
    link-local, reserved, multicast and unspecified ranges plus the explicit
    metadata IPs, and unwraps IPv4-in-IPv6 forms (``::ffff:a.b.c.d``, 6to4,
    teredo) so an embedded internal address cannot slip through.
    """
    # Unwrap IPv4-mapped IPv6 (e.g. ``::ffff:169.254.169.254``) and recheck the
    # embedded v4 address against every rule.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and _ip_is_blocked(mapped):
        return True
    # 6to4 (``2002::/16``) and Teredo (``2001::/32``) embed a v4 address too.
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None and _ip_is_blocked(sixtofour):
        return True
    teredo = getattr(ip, "teredo", None)
    if teredo is not None:
        # ``teredo`` is a (server, client) pair of v4 addresses.
        if any(_ip_is_blocked(part) for part in teredo):
            return True
    if ip in _METADATA_IPS:
        return True
    if any(ip in net for net in _EXTRA_BLOCKED_NETWORKS):
        return True
    # ``is_global`` is the catch-all; the explicit predicates below are
    # belt-and-suspenders for versions that misreport it.
    if not ip.is_global:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _parse_allowed_hosts(
    entries: list[str],
) -> tuple[list[ipaddress.IPv4Network | ipaddress.IPv6Network], set[str]]:
    """Split the configured allowlist into IP networks and hostnames.

    An entry that parses as an IP/CIDR becomes a network; everything else is a
    case-insensitive hostname.
    """
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    names: set[str] = set()
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            # Normalize the FQDN trailing-dot form so an allowlisted
            # "minio.internal" also matches the "minio.internal." a client may
            # send (and vice versa).
            names.add(entry.rstrip(".").lower())
    return nets, names


def _host_allowlisted(
    host: str,
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    names: set[str],
) -> bool:
    """Whether an otherwise-blocked host is explicitly permitted.

    A hostname match permits the host regardless of the address it resolves to
    (the operator is vouching for that name); a CIDR/IP allowlist permits a host
    only when *every* resolved address falls within an allowlisted range (so a
    host that mixes an allowed and a disallowed answer is still rejected).
    """
    if host.rstrip(".").lower() in names:
        return True
    return bool(
        nets and ips and all(any(ip in net for net in nets) for ip in ips)
    )


async def _resolve_host(host: str, port: int) -> list[str]:
    """Resolve a hostname to its IP strings. Isolated for testability.

    Returns the literal IP strings from ``getaddrinfo`` (A and AAAA). A host that
    is already an IP literal resolves to itself.
    """
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    # sockaddr[0] is typed str | int because typeshed includes the AF_PACKET
    # form; SOCK_STREAM lookups only yield AF_INET/AF_INET6, where it is a str.
    addrs: list[str] = []
    for info in infos:
        addr = info[4][0]
        assert isinstance(addr, str)
        addrs.append(addr)
    return addrs


async def _validate_and_pin(
    url: str,
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    names: set[str],
) -> tuple[str, str, str, httpx.URL]:
    """Validate a media URL and resolve it to a pinned connection target.

    Returns ``(pinned_url, host_header, sni_hostname, parsed)`` where
    ``pinned_url`` has its host replaced by a single validated IP literal so
    httpx connects there without re-resolving (no rebinding window), while
    ``host_header`` / ``sni_hostname`` carry the real hostname so the Host header
    is correct and TLS still verifies against the hostname; ``parsed`` is the
    parsed input URL, returned so the caller need not re-parse it.

    Raises ``InputError`` (mapped to a 400) for a disallowed scheme, a host that
    will not resolve, or a host that resolves to an internal/reserved/metadata
    address and is not allowlisted.
    """
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        # A malformed URL (e.g. a dotted-octal host or unbalanced IPv6 bracket,
        # possibly arriving via a redirect Location) is a client error, not a
        # server fault -- surface a 400, never an opaque 500.
        _reject_media("malformed_url", "malformed media URL")
    scheme = parsed.scheme
    if scheme not in _ALLOWED_SCHEMES:
        _reject_media("bad_scheme", f"unsupported media URL scheme '{scheme}'")
    host = parsed.host
    if not host:
        _reject_media("no_host", "media URL has no host")
    default_port = 443 if scheme == "https" else 80
    port = parsed.port or default_port

    try:
        ip_strings = await _resolve_host(host, port)
    except socket.gaierror:
        _reject_media(
            "dns_failure", f"could not resolve media URL host '{host}'"
        )
    if not ip_strings:
        _reject_media(
            "dns_failure", f"could not resolve media URL host '{host}'"
        )

    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw_ip in ip_strings:
        # getaddrinfo can hand back a scoped v6 literal ("fe80::1%eth0"); drop
        # the zone id before parsing.
        try:
            ips.append(ipaddress.ip_address(raw_ip.split("%", 1)[0]))
        except ValueError:
            _reject_media(
                "dns_failure",
                f"media URL host '{host}' resolved to an unparseable address",
            )

    if not _host_allowlisted(host, ips, nets, names):
        if any(_ip_is_blocked(ip) for ip in ips):
            _reject_media(
                "ssrf_blocked",
                "media URL host resolves to a disallowed"
                " (internal/reserved) address",
            )

    # Pin to a single validated (or allowlisted) IP, chosen deterministically.
    pinned_ip_obj = min(ips, key=lambda ip: (ip.version != 4, ip.packed))
    pinned_url = str(parsed.copy_with(host=str(pinned_ip_obj), port=port))
    host_header = host if port == default_port else f"{host}:{port}"
    return pinned_url, host_header, host, parsed


# ``data:`` payloads at or below this size (in bytes of the base64 string) are
# decoded inline; larger ones are offloaded to a worker thread. Base64 decoding
# is pure-Python CPU work that blocks the event loop, so a multi-MB payload
# decoded inline stalls every other in-flight request (the TTFT culprit in
# CENG-640). The threshold keeps the common small-thumbnail case off the thread
# pool while pushing the expensive large-payload case off the loop.
_DATA_URI_OFFLOAD_THRESHOLD = 256 * 1024

# Read remote media in bounded chunks so a streamed download can be aborted the
# moment it crosses the size cap, instead of buffering the whole (potentially
# huge) body before checking its length.
_HTTP_CHUNK_SIZE = 256 * 1024

# Explicit fetch timeouts. httpx's default is a 5s timeout on *every* operation,
# which is far too aggressive for media downloads: a large video (or a slow
# CDN) routinely needs more than 5s, and any stall longer than that raises a
# ``ReadTimeout`` that — left unhandled — surfaces as an opaque HTTP 500 (and
# the client then retries the whole generation). The read timeout below is
# per-chunk (each ``_HTTP_CHUNK_SIZE`` read must complete within it), not a cap
# on total download time; total size is already bounded by the byte cap. Pick
# values generous enough for slow-but-steady transfers while still failing a
# truly stalled connection in bounded time.
_FETCH_TIMEOUT = Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

# Some media hosts (e.g. Wikimedia, Google Cloud Storage) reject requests that
# carry a default library User-Agent (httpx sends ``python-httpx/...``) with an
# HTTP 403, which turned valid user-supplied image/video URLs into fetch
# failures. Present a common browser User-Agent (and a permissive Accept) so
# fetching from such hosts succeeds.
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def _estimated_decoded_bytes(image: Image.Image) -> int:
    """Estimate an image's decoded (in-memory) size in bytes from its header.

    ``width * height * channels`` for the common 8-bit-per-channel modes -- a
    close proxy for the pixel buffer ``image.load()`` will allocate, computed
    from the header alone so it is known before that allocation happens.
    """
    return image.width * image.height * len(image.getbands())


def decode_and_validate_images(
    images: list[bytes],
    max_decoded_bytes: int | None = None,
    skip_decode: Sequence[bool] | None = None,
    *,
    facts: list[_ImageFact] | None = None,
) -> list[Image.Image | None]:
    # Fully decode each image so empty, non-image, or truncated/streamed
    # content (e.g. animated or content-negotiated WebP) fails here as a clean
    # 400 instead of reaching the model worker and crashing it with an
    # unhandled PIL error or OSError (HTTP 500). ``Image.open`` is lazy -- it
    # only parses the header -- so a header-valid but undecodable image slips
    # through and later blows up in the tokenizer's ``to_rgb(...)`` ->
    # ``.convert("RGB")`` decode. ``image.load()`` forces that same pixel
    # decode now, while we can still turn the failure into a 400.
    #
    # The decoded images are returned and carried on the request
    # (``TextGenerationRequest.decoded_images``) so the tokenizer reuses them
    # instead of decoding the same bytes a second time. We therefore do not
    # close the images we return (no ``with`` block): ``load()`` has already
    # pulled the pixels into memory and the caller owns the decoded image.
    #
    # ``skip_decode`` marks images whose preprocessed tensor the tokenizer
    # already holds, so nothing downstream reads their pixels. Only ``load()``
    # is skipped -- the format allowlist and the decoded-size estimate are
    # cheap header reads and stay unconditional, so a skipped image is held to
    # the same limits as any other. Its slot is ``None``;
    # ``TextGenerationRequest.images_for_processing`` falls back to the raw
    # bytes per index, and ``open_image`` applies the same allowlist and the
    # same 400 if they do have to be decoded after all.
    #
    # ``facts`` is an out parameter: one :class:`_ImageFact` is appended per
    # image *before* any raise, so a caller whose third image is undecodable
    # still sees images 1-2 and the rejection. The caller records them, on the
    # event loop, because this function runs in a worker thread.
    if skip_decode is not None and len(skip_decode) != len(images):
        raise ValueError(
            f"skip_decode has {len(skip_decode)} entries but there are "
            f"{len(images)} image(s); it must be aligned with ``images``."
        )
    decoded: list[Image.Image | None] = []
    for index, image_bytes in enumerate(images):
        # The codec window: the header parse through the full pixel decode,
        # and nothing past it. The per-architecture processor runs downstream
        # of this function on an already-decoded buffer, and its cost follows
        # the pixel count rather than the container, so widening the window
        # to include it would rank codecs by the sizes of the images that
        # happen to use them.
        decode = StopWatch()
        try:
            image = Image.open(
                io.BytesIO(image_bytes), formats=ALLOWED_IMAGE_FORMATS
            )
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as e:
            _record_fact(facts, _ImageFact("unknown", rejection="undecodable"))
            raise InputError("invalid or unreadable image content") from e
        image_format = _metric_image_format(image)
        pixels = image.width * image.height
        # Bound decoded memory with the same knob that bounds the fetch --
        # ``max_media_bytes``. An image's *encoded* size (already charged to the
        # request budget when it was fetched) says nothing about how much it
        # decodes to: a small on-the-wire PNG can expand to hundreds of MB (a
        # decompression bomb). Estimating that from the header lets us reject an
        # oversized image before ``load()`` allocates its pixel buffer, instead
        # of maintaining a separate pixel-count limit (MXSERV-389, finding 6).
        #
        # This has to follow ``Image.open``, cheap as it is: the estimate reads
        # the header dimensions, and parsing the header is exactly what
        # ``Image.open`` does. It still precedes the only expensive step
        # (``load()``, which allocates the buffer), which is the property that
        # matters -- a bomb is refused before the allocation, not after.
        if max_decoded_bytes is not None:
            estimated = _estimated_decoded_bytes(image)
            if estimated > max_decoded_bytes:
                _record_fact(
                    facts,
                    _ImageFact(
                        image_format, pixels, rejection="decode_too_large"
                    ),
                )
                raise InputError(
                    "image decodes to "
                    f"{to_human_readable_bytes(estimated)}, exceeding the "
                    "maximum media size of "
                    f"{to_human_readable_bytes(max_decoded_bytes)} per request "
                    f"by {to_human_readable_bytes(estimated - max_decoded_bytes)}"
                )
        if skip_decode is not None and skip_decode[index]:
            # Past both cheap checks, so this image is as validated as any
            # other; what is skipped is only the allocation. Closed rather
            # than carried: handing the tokenizer a lazy, un-loaded image
            # would just move the decode to whenever it first touches a pixel.
            image.close()
            decoded.append(None)
            # Format and pixels are in hand either way -- the skip is decided
            # after the header read -- so only the decode sample is missing.
            _record_fact(facts, _ImageFact(image_format, pixels))
            continue
        try:
            image.load()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as e:
            _record_fact(
                facts, _ImageFact(image_format, pixels, rejection="undecodable")
            )
            raise InputError("invalid or unreadable image content") from e
        _record_fact(
            facts,
            _ImageFact(image_format, pixels, decode_ms=decode.elapsed_ms),
        )
        decoded.append(image)
    return decoded


class _MediaByteBudget:
    """A shared, decreasing byte budget for all media resolved in one request.

    The size limit on client-supplied media is a single per-request aggregate,
    not a per-item cap: the total of every fetched and decoded image and video
    must stay within :attr:`Settings.max_media_bytes`. One budget is created per
    request and shared across every media item, so a request cannot pull in
    more than that many bytes of media regardless of how it splits them.

    This is deliberately *not* :attr:`Settings.max_request_bytes`, which bounds
    the request body. A body of a few hundred bytes can name a dozen URLs the
    server then fetches on the client's behalf, so the body limit says nothing
    about how much media a request pulls in; keeping the two separate also means
    raising the body limit to admit large inline base64 does not silently widen
    what a decompression bomb is allowed to decode to.

    The chat path resolves a request's media concurrently, but asyncio runs
    those coroutines on a single thread and ``remaining`` is only ever charged
    between await points (never inside an ``asyncio.to_thread`` worker), so the
    mutation needs no lock. ``limit`` of ``None`` means unbounded (the operator
    set ``max_media_bytes`` to 0).
    """

    __slots__ = ("limit", "remaining")

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.remaining = limit

    def exceeds(self, additional: int) -> bool:
        """Whether ``additional`` bytes alone would overrun what is left."""
        return self.remaining is not None and additional > self.remaining

    def charge(self, n: int, media_kind: str) -> None:
        """Charge ``n`` received bytes, rejecting once the budget is spent.

        Raises:
            InputError: If the running media total exceeds the request limit.
        """
        if self.remaining is None:
            return
        if self.exceeds(n):
            self.reject(media_kind, n)
        self.remaining -= n

    def reject(self, media_kind: str, additional: int) -> NoReturn:
        """Reject the request as over its aggregate media-size budget.

        Args:
            media_kind: ``"image"`` or ``"video"``, for the error wording.
            additional: The bytes that would not fit, used to report by how
                much the request overran its budget.
        """
        assert self.limit is not None and self.remaining is not None
        over = additional - self.remaining
        _reject_media(
            "over_budget",
            f"{media_kind} media exceeds the maximum media size of "
            f"{to_human_readable_bytes(self.limit)} per request by "
            f"{to_human_readable_bytes(over)}",
        )


def _max_media_bytes(settings: Settings | None) -> int | None:
    """The per-request media-memory limit, or ``None`` when disabled/unset.

    Tolerates ``settings`` being ``None`` or a test double whose attribute is
    not a positive int, in which case the limit simply does not apply.
    """
    limit = getattr(settings, "max_media_bytes", None)
    return limit if isinstance(limit, int) and limit > 0 else None


def _request_media_budget(settings: Settings | None) -> _MediaByteBudget:
    """Build the per-request media budget from ``Settings.max_media_bytes``."""
    return _MediaByteBudget(_max_media_bytes(settings))


def _clean_data_uri_base64(data_uri: str) -> str:
    """Extract and normalize the base64 payload of a ``data:`` URI.

    Tolerates the two ways real clients (and the OpenRouter image relay)
    routinely deviate from canonical base64: stripped ``=`` padding and the
    URL-safe alphabet (``-``/``_``).
    """
    parts = data_uri.split(",", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("data URI has no base64 payload")
    # Some clients wrap long payloads across lines; strip any whitespace.
    b64 = "".join(parts[1].split())
    # Re-add stripped padding (base64 length must be a multiple of 4).
    b64 += "=" * (-len(b64) % 4)
    return b64


def _decode_base64(b64: str) -> bytes:
    decoder = (
        base64.urlsafe_b64decode
        if ("-" in b64 or "_" in b64)
        else base64.b64decode
    )
    return decoder(b64)


def _decode_data_uri_base64(data_uri: str) -> bytes:
    """Decode the base64 payload of a ``data:`` image URI."""
    return _decode_base64(_clean_data_uri_base64(data_uri))


class MediaRef(Protocol):
    """Structural type for a resolvable media reference."""

    @property
    def scheme(self) -> str: ...

    def unicode_string(self) -> str: ...

    def __str__(self) -> str: ...


class DataUrl:
    """Cheap ``data:`` URL ref that skips pydantic's expensive URL parse."""

    __slots__ = ("_raw",)
    scheme = "data"

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def unicode_string(self) -> str:
        return self._raw

    def __str__(self) -> str:
        return self._raw


def make_media_ref(url: str) -> MediaRef:
    """Build a media reference, skipping pydantic validation for ``data:`` URIs.

    ``AnyUrl`` validation of a large base64 ``data:`` payload is pure overhead
    (there is nothing network-shaped to validate) and blocks the event loop, so
    those are wrapped in the lightweight :class:`DataUrl` instead.
    """
    if url[:5].lower() == "data:":
        return DataUrl(url)
    try:
        return AnyUrl(url)
    except ValidationError:
        # The only place a client URL that is not URL-shaped is refused, and
        # pydantic's own error counts nothing -- leaving the most ordinary
        # client bug missing from the refusal rate. Truncated because the
        # value is client-controlled and otherwise unbounded.
        _reject_media("malformed_url", f"malformed media URL '{url[:200]}'")


async def _read_streamed_response(
    response: Response, media_kind: str, budget: _MediaByteBudget
) -> bytes:
    """Stream a final (non-redirect) response body under the request budget.

    Charges the shared per-request media budget as bytes arrive, so a fetch is
    aborted the moment the request's cumulative media crosses the limit. The
    ``Content-Length`` fast path rejects an over-budget body before any bytes
    are read; streaming then covers a missing or lying ``Content-Length`` and,
    because the budget is shared, bounds the total buffered media across every
    media item in the request rather than each one independently.

    Args:
        response: An open streaming response.
        media_kind: ``"image"`` or ``"video"`` (used in the error message).
        budget: The request's shared media-byte budget.

    Returns:
        The full response body.

    Raises:
        InputError: If the request's cumulative media exceeds the budget.
        HTTPStatusError: If the response status is 4xx or 5xx.
    """
    response.raise_for_status()
    # Fast path: reject up front when the server advertises a size that alone
    # overruns the remaining budget, so we never start streaming the body.
    advertised = response.headers.get("content-length")
    if advertised is not None and advertised.isdigit():
        if budget.exceeds(int(advertised)):
            budget.reject(media_kind, int(advertised))
    chunks: list[bytes] = []
    async for chunk in response.aiter_bytes(_HTTP_CHUNK_SIZE):
        budget.charge(len(chunk), media_kind)
        chunks.append(chunk)
    return b"".join(chunks)


async def _fetch_validated(
    image_ref: MediaRef,
    settings: Settings,
    media_kind: str,
    budget: _MediaByteBudget,
    client: AsyncClient,
) -> bytes:
    """SSRF-safe streaming fetch: validate + pin, following redirects manually so
    every hop is re-validated. Streams the final body under the byte cap and
    returns it, or raises ``HTTPStatusError`` (non-2xx -> caller maps) /
    ``InputError``.
    """
    nets, names = _parse_allowed_hosts(settings.media_url_allowed_hosts)
    current = str(image_ref)
    for _hop in range(_MAX_REDIRECTS + 1):
        pinned_url, host_header, sni_hostname, parsed = await _validate_and_pin(
            current, nets, names
        )
        # Only override Host/SNI when the original URL used a hostname. If the
        # client supplied an IP-literal URL, let httpx derive Host/SNI from the
        # (pinned) URL to avoid emitting an invalid IPv6 Host header.
        headers: dict[str, str] = {}
        try:
            ipaddress.ip_address(parsed.host)
        except ValueError:
            headers["Host"] = host_header

        extensions: dict[str, str] = {}
        if parsed.scheme == "https":
            try:
                ipaddress.ip_address(parsed.host)
            except ValueError:
                extensions["sni_hostname"] = sni_hostname
        try:
            async with client.stream(
                "GET",
                pinned_url,
                headers=headers,
                extensions=extensions,
                follow_redirects=False,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        _reject_media(
                            "bad_redirect",
                            f"media url '{image_ref}' returned a redirect"
                            " with no location",
                        )
                    # Resolve a relative redirect against the original hostname
                    # URL (not the pinned-IP one), then loop to re-validate it.
                    try:
                        next_url = str(parsed.join(location))
                    except httpx.InvalidURL:
                        _reject_media(
                            "malformed_url",
                            f"media url '{image_ref}' redirected to a"
                            " malformed location",
                        )
                    current = next_url
                    continue
                return await _read_streamed_response(
                    response, media_kind, budget
                )
        except httpx.RemoteProtocolError as e:
            # A malformed redirect Location that httpx rejects while opening the
            # stream is a client-visible 400, not a transient error.
            if "location" in str(e).lower():
                _reject_media(
                    "malformed_url",
                    f"media url '{image_ref}' returned a malformed redirect"
                    " location",
                )
            raise
    _reject_media(
        "too_many_redirects",
        f"too many redirects fetching media url '{image_ref}'",
    )


async def _fetch_url_bytes(
    image_ref: MediaRef,
    settings: Settings,
    media_kind: str,
    budget: _MediaByteBudget,
) -> bytes:
    """Fetch an http(s) media reference into raw bytes under the request budget.

    When SSRF protection is enabled (the default) the host is validated and the
    connection pinned to the resolved IP, following redirects manually and
    re-validating each hop; the break-glass path (protection disabled) restores
    the legacy fetch that lets httpx follow redirects itself. Both paths stream
    the body while charging the shared request budget and share the same error
    mapping.
    """
    # TODO: Evaluate creating a single AsyncClient for the app.
    async with AsyncClient(
        headers=_FETCH_HEADERS, timeout=_FETCH_TIMEOUT
    ) as client:
        try:
            if settings.media_url_ssrf_protection_enabled:
                return await _fetch_validated(
                    image_ref, settings, media_kind, budget, client
                )
            # Break-glass: guard disabled -> legacy unvalidated fetch that lets
            # httpx follow redirects itself.
            async with client.stream(
                "GET", str(image_ref), follow_redirects=True
            ) as response:
                return await _read_streamed_response(
                    response, media_kind, budget
                )
        except HTTPStatusError as e:
            _reject_media(
                "fetch_status",
                f"Failed to fetch {media_kind}: HTTP {e.response.status_code}",
                error=ValueError,
            )
        except TimeoutException:
            # A slow/stalled download must not surface as an opaque 500
            # (which the client then retries). Turn it into a clean input
            # error attributable to the unreachable/slow media source.
            _reject_media(
                "fetch_timeout",
                f"timed out fetching {media_kind} from its URL; the source "
                "may be too slow or the file too large",
            )
        except TransportError as e:
            # Connection reset / DNS / network failure mid-fetch: same
            # treatment as a timeout, a clean input error rather than a 500.
            _reject_media(
                "fetch_transport",
                f"failed to fetch {media_kind} from its URL "
                f"({type(e).__name__})",
            )


async def resolve_image_from_url(
    image_ref: MediaRef,
    settings: Settings,
    budget: _MediaByteBudget | None = None,
    media_kind: str | None = None,
) -> bytes:
    """Resolve a media reference into raw bytes under the request media budget.

    The size limit is a single per-request aggregate, not a per-item cap: all
    fetched and decoded media for a request must fit within
    :attr:`Settings.max_media_bytes`. Callers that resolve several items for one
    request pass a shared :class:`_MediaByteBudget` so the total is bounded;
    when ``budget`` is ``None`` a fresh one is built from ``settings`` (a single
    standalone item). An ``http(s)`` download is aborted as soon as the
    cumulative bytes cross the budget; a ``data:`` payload is charged once
    decoded (it is already bounded by the body-size middleware, which caps the
    request body).

    ``media_kind`` (``"image"``/``"video"``) only selects the error wording; it
    falls back to :attr:`Settings.media_kind` and then ``"image"``.
    """
    if budget is None:
        budget = _request_media_budget(settings)
    if media_kind is None:
        settings_kind = getattr(settings, "media_kind", None)
        media_kind = (
            settings_kind if isinstance(settings_kind, str) else "image"
        )

    # One measured window around all three scheme arms, so ``source`` is the
    # only thing that distinguishes them and a fetch cannot be timed while an
    # inline decode is not. Only a resolved item is counted; a rejected one is
    # already counted by ``maxserve.media.rejections``.
    resolve = StopWatch()

    if image_ref.scheme == "http" or image_ref.scheme == "https":
        images_bytes = await _fetch_url_bytes(
            image_ref, settings, media_kind, budget
        )
        logger.debug(
            "ResolvedImageUrl: %s -> %d bytes", image_ref, len(images_bytes)
        )
    elif image_ref.scheme == "data":
        data_uri = image_ref.unicode_string()
        # Decode off the event loop for large payloads: base64 decoding is
        # CPU-bound pure-Python work that otherwise stalls every concurrent
        # request (CENG-640). Small thumbnails decode inline to skip the
        # thread-pool hop. The budget is charged here on the event loop -- never
        # inside the worker thread -- once the decoded size is known.
        #
        # Caught out here rather than at the raise: the large-payload arm
        # runs in a worker thread, and the metric clients are not all
        # thread-safe. ``binascii.Error`` is a ``ValueError``, so this covers
        # both a missing payload and one that is not base64 at all.
        try:
            if len(data_uri) > _DATA_URI_OFFLOAD_THRESHOLD:
                images_bytes = await asyncio.to_thread(
                    _decode_data_uri_base64, data_uri
                )
            else:
                images_bytes = _decode_data_uri_base64(data_uri)
        except ValueError:
            _reject_media(
                "bad_data_uri",
                "inline media is not a decodable base64 data URI",
            )
        budget.charge(len(images_bytes), media_kind)
        logger.debug(
            "ResolvedImageB64: %s -> %d bytes",
            str(image_ref)[:16],
            len(images_bytes),
        )
    elif image_ref.scheme == "file":
        if settings is None:
            raise ValueError("Settings required for file URI resolution")

        # Parse the file URI.
        parsed = urlparse(str(image_ref))

        # Check host - only allow empty or localhost.
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            _reject_media(
                "bad_file_uri",
                f"File URI with remote host '{parsed.netloc}' is not supported",
                error=ValueError,
            )

        # Extract and decode the path.
        file_path = Path(unquote(parsed.path))

        # Validate against allowed roots.
        allowed_roots = [Path(root) for root in settings.allowed_image_roots]
        if not allowed_roots:
            _reject_media(
                "bad_file_uri",
                "File URI access denied: no allowed roots configured",
                error=ValueError,
            )

        # Canonicalize the path (resolving symlinks and ``..``) so containment
        # is decided against the real target rather than the requested
        # spelling. Resolve non-strictly: a nonexistent path must not raise
        # here, because the allowed-roots check has to run *before* any
        # existence or type probe. Statting an attacker-chosen absolute path
        # that lies outside the roots would turn this into a filesystem
        # existence/type oracle.
        try:
            resolved_path = file_path.resolve()
        except (OSError, RuntimeError):
            _reject_media(
                "bad_file_uri",
                f"Invalid file path: {file_path}",
                error=ValueError,
            )

        # Check if path is within allowed roots before touching the filesystem.
        path_allowed = False
        for root in allowed_roots:
            try:
                resolved_path.relative_to(root)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            _reject_media(
                "bad_file_uri",
                f"Path forbidden: {resolved_path} is outside allowed roots",
                error=ValueError,
            )

        # Only now that the path is contained within an allowed root do we
        # probe the filesystem for existence and type.
        if not resolved_path.exists():
            _reject_media(
                "bad_file_uri",
                f"File not found: {file_path}",
                error=ValueError,
            )

        if resolved_path.is_dir():
            _reject_media(
                "bad_file_uri",
                f"Path is a directory: {resolved_path}",
                error=ValueError,
            )

        # Read the file with its own local-file size limit.
        max_local_bytes = settings.max_local_image_bytes

        async with aiofiles.open(resolved_path, "rb") as f:
            images_bytes = await f.read(max_local_bytes + 1)
            if len(images_bytes) > max_local_bytes:
                _reject_media(
                    "bad_file_uri",
                    f"File exceeds size limit of {max_local_bytes} bytes",
                    error=ValueError,
                )
        budget.charge(len(images_bytes), media_kind)
        logger.debug(
            "ResolvedFileUri: %s -> %d bytes", resolved_path, len(images_bytes)
        )
    else:
        _reject_media(
            "bad_scheme", f"Invalid image ref '{image_ref}'", error=ValueError
        )

    source = _MEDIA_SOURCE_BY_SCHEME[image_ref.scheme]
    METRICS.media_items(media_kind, source)
    METRICS.media_resolve_time(resolve.elapsed_ms, source)
    METRICS.media_item_size(len(images_bytes), media_kind)
    return images_bytes


def _sniff_image_mime(image: Image.Image) -> str:
    """Return the MIME type of a decoded image, from the format PIL reports.

    Sniffing beats guessing from the URL: a ``.jpg`` URL that a host
    content-negotiates to WebP would otherwise be re-encoded under a MIME type
    that contradicts its own payload.
    """
    image_format = image.format
    if not image_format:
        raise InputError("invalid or unreadable image content")
    # MPO is a multi-picture JPEG; every other format PIL names maps directly.
    if image_format == "MPO":
        return "image/jpeg"
    return f"image/{image_format.lower()}"


def _encode_data_uri(
    image_bytes: bytes,
    max_decoded_bytes: int | None,
    facts: list[_ImageFact] | None = None,
) -> str:
    """Fully validate image bytes, then inline them as a base64 ``data:`` URI.

    Runs the same full-pixel decode as the chat path (see
    :func:`decode_and_validate_images`), so truncated or otherwise undecodable
    content fails here as a clean 400 rather than reaching the tokenizer's own
    decode and surfacing as a 500, and bounds decoded memory by the same
    ``max_media_bytes`` knob. Unlike the chat path, nothing downstream reuses
    this decode -- the responses path re-decodes from the ``data:`` URI it is
    handed -- so the image is released as soon as its format is read.
    """
    image = decode_and_validate_images(
        [image_bytes], max_decoded_bytes, facts=facts
    )[0]
    # No ``skip_decode`` here, so the one slot always holds a decoded image.
    assert image is not None
    try:
        mime = _sniff_image_mime(image)
    except InputError:
        # A decoded image PIL will not name. The decode itself already
        # recorded its own fact, so this adds the refusal rather than
        # replacing it.
        _record_fact(facts, _ImageFact("unknown", rejection="undecodable"))
        raise
    finally:
        image.close()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def fetch_media_data_uri(url: str, settings: Settings) -> str:
    """Fetch a web media URL and return it inlined as a base64 ``data:`` URI.

    The ``/v1/responses`` input schema accepts ``data:`` URIs only (see
    ``InputImageContent.validate_data_uri_only``), so a client-supplied
    ``http(s)`` image must be fetched and inlined before the body validates.
    Routing that through :func:`resolve_image_from_url` keeps the responses path
    on the same fetch as chat completions: one size limit, one error mapping,
    one place where host validation lives, and the same full-decode validation,
    rather than a second downloader that has to be hardened separately.

    Args:
        url: The client-supplied media URL.
        settings: Server settings, supplying the per-request media budget.

    Returns:
        A ``data:<mime>;base64,<payload>`` URI.

    Raises:
        InputError: If the media is oversized, unfetchable, or not a decodable
            image.
    """
    image_bytes = await resolve_image_from_url(
        make_media_ref(url), settings, media_kind="image"
    )
    # The same media knob that bounds the fetch also bounds how much this image
    # may decode to. This is the fixed limit, not a second budget: the
    # decoded-size check is a stateless per-image comparison against it.
    max_decoded_bytes = _max_media_bytes(settings)
    # Decoding for validation and base64-encoding are both CPU-bound, so a
    # multi-MB image goes to a worker thread for the same reason the ``data:``
    # decode does.
    #
    # The list is allocated here, in the frame that also catches, for the
    # same reason the chat path allocates its own: the decode raises out of
    # the worker thread with no return value, and the handler above cannot
    # see a local of the function that raised. Recording happens back on the
    # event loop because the metric clients are not all thread-safe.
    facts: list[_ImageFact] = []
    try:
        if len(image_bytes) > _DATA_URI_OFFLOAD_THRESHOLD:
            data_uri = await asyncio.to_thread(
                _encode_data_uri, image_bytes, max_decoded_bytes, facts
            )
        else:
            data_uri = _encode_data_uri(image_bytes, max_decoded_bytes, facts)
    except Exception:
        emit_media_facts(facts)
        raise
    emit_media_facts(facts)
    return data_uri
