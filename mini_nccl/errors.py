"""Exception types.

Distributed failures are only useful if they say *which* rank stopped
holding up its end, so every error here carries the local rank's view of
what it was waiting for.
"""

from __future__ import annotations


class MiniNcclError(Exception):
    """Base class for all mini-nccl failures."""


class CollectiveTimeoutError(MiniNcclError, TimeoutError):
    """A peer did not deliver its share of a collective in time.

    Raised instead of blocking forever, which is what makes a desynchronized
    or dead rank diagnosable rather than a silent hang.
    """


class PeerClosedError(MiniNcclError, ConnectionError):
    """A peer's connection dropped mid-message (usually a crashed rank)."""


class RendezvousError(MiniNcclError, TimeoutError):
    """Ranks failed to establish the full connection mesh."""
