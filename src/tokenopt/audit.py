"""Hash-chained audit log.

Every optimization decision (policy applied, budget enforced, cache hit,
promotion) is appended as a JSON line carrying the SHA-256 hash of the
previous entry. Any tampering breaks the chain, which ``verify()`` detects.
The log is append-only: entries are never edited or deleted in place.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _entry_hash(prev_hash: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev_hash}|{canonical}".encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    event: str
    tenant: str
    details: dict
    prev_hash: str
    entry_hash: str


@dataclass
class AuditLog:
    entries: list[AuditEntry] = field(default_factory=list)

    def append(self, event: str, tenant: str, details: dict | None = None) -> AuditEntry:
        prev = self.entries[-1].entry_hash if self.entries else "GENESIS"
        payload = {
            "seq": len(self.entries) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "tenant": tenant,
            "details": details or {},
            "prev_hash": prev,
        }
        entry = AuditEntry(
            seq=payload["seq"],
            timestamp=payload["timestamp"],
            event=event,
            tenant=tenant,
            details=details or {},
            prev_hash=prev,
            entry_hash=_entry_hash(prev, payload),
        )
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, int]:
        """Verify the chain. Returns (ok, first_bad_seq or -1)."""
        prev = "GENESIS"
        for entry in self.entries:
            payload = {
                "seq": entry.seq,
                "timestamp": entry.timestamp,
                "event": entry.event,
                "tenant": entry.tenant,
                "details": entry.details,
                "prev_hash": entry.prev_hash,
            }
            if entry.prev_hash != prev:
                return False, entry.seq
            if entry.entry_hash != _entry_hash(prev, payload):
                return False, entry.seq
            prev = entry.entry_hash
        return True, -1

    def to_jsonl(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(json.dumps({
                "seq": e.seq, "timestamp": e.timestamp, "event": e.event,
                "tenant": e.tenant, "details": e.details,
                "prev_hash": e.prev_hash, "entry_hash": e.entry_hash,
            }, sort_keys=True))
        return "\n".join(lines) + ("\n" if lines else "")
