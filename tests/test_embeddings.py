"""Tests: hashed embeddings and similarity."""
import pytest

from tokenopt.embeddings import DIM, cosine, embed, similarity


def test_embedding_is_deterministic():
    assert embed("refund policy") == embed("refund policy")


def test_embedding_is_normalized():
    vec = embed("some words here")
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0)


def test_empty_text_embedding_is_zero():
    assert embed("") == [0.0] * DIM


def test_identical_texts_similarity_one():
    assert similarity("refund policy 30 days", "refund policy 30 days") == pytest.approx(1.0)


def test_unrelated_texts_similarity_low():
    assert similarity(
        "refund policy returns exchange window customer support",
        "quantum entanglement photon superposition laboratory experiment",
    ) < 0.3


def test_partial_overlap_between_zero_and_one():
    sim = similarity("refund policy returns window", "refund policy exchanges window")
    assert 0.0 < sim < 1.0


def test_cosine_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])
