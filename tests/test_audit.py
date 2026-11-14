"""Tests: hash-chained audit log."""
from tokenopt.audit import AuditLog


def test_append_and_verify_clean_chain():
    log = AuditLog()
    log.append("optimize", "acme", {"request_id": "r1"})
    log.append("promote", "acme", {"policy": "balanced@v1"})
    ok, bad = log.verify()
    assert ok is True and bad == -1


def test_empty_log_verifies():
    assert AuditLog().verify() == (True, -1)


def test_tampering_breaks_chain():
    log = AuditLog()
    log.append("optimize", "acme", {"request_id": "r1"})
    log.append("optimize", "acme", {"request_id": "r2"})
    log.entries[0].details["request_id"] = "forged"
    ok, bad = log.verify()
    assert ok is False and bad == 1


def test_hash_linking():
    log = AuditLog()
    first = log.append("optimize", "acme", {})
    second = log.append("optimize", "acme", {})
    assert first.prev_hash == "GENESIS"
    assert second.prev_hash == first.entry_hash
    assert first.seq == 1 and second.seq == 2


def test_jsonl_round_trips():
    log = AuditLog()
    log.append("optimize", "acme", {"request_id": "r1"})
    lines = log.to_jsonl().strip().splitlines()
    assert len(lines) == 1
    assert '"event": "optimize"' in lines[0]
