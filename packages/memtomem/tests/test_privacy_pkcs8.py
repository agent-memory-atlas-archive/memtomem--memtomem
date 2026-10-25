"""PKCS#8 private keys are caught by the write guard (GHSA-mf67-ghgq-cv54 sync).

The PEM rule only knew algorithm-labelled headers (``BEGIN RSA PRIVATE KEY``),
so a PKCS#8 key — what OpenSSL 3 ``genrsa``/``genpkey`` write by default, and the
only PEM form an Ed25519 key has — passed every ingress guard and was stored,
chunked and embedded. Keys are generated per run, so no key material sits in
the tree.
"""

from __future__ import annotations

import datetime
import json
import re
from functools import cache
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization as ser
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

from memtomem import privacy
from memtomem.indexing.engine import PrivacyRejection
from memtomem.server.context import AppContext
from memtomem.server.tools.memory_crud import mem_add
from memtomem.storage.source_holds import REDACTION_BLOCKED_HOLD

from helpers import StubCtx

_ENCRYPTIONS = {
    "plain": ser.NoEncryption(),
    "encrypted": ser.BestAvailableEncryption(b"disposable"),
}


@cache
def _key(kind: str):
    if kind == "rsa":
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)
    if kind == "ec":
        return ec.generate_private_key(ec.SECP256R1())
    return ed25519.Ed25519PrivateKey.generate()


@cache
def _pkcs8(kind: str, encryption: str) -> str:
    return (
        _key(kind)
        .private_bytes(ser.Encoding.PEM, ser.PrivateFormat.PKCS8, _ENCRYPTIONS[encryption])
        .decode()
    )


_KEYS = [(k, e) for k in ("rsa", "ec", "ed25519") for e in _ENCRYPTIONS]
_KEY_IDS = [f"{k}-{e}" for k, e in _KEYS]


def _assert_loadable(pem: str, encryption: str) -> None:
    password = b"disposable" if encryption == "encrypted" else None
    ser.load_pem_private_key(pem.encode(), password=password)


@pytest.fixture(autouse=True)
def _reset_counters():
    privacy.reset_for_tests()
    yield
    privacy.reset_for_tests()


@pytest.mark.parametrize(("kind", "encryption"), _KEYS, ids=_KEY_IDS)
def test_a_generated_pkcs8_key_is_detected(kind, encryption):
    assert privacy.scan(_pkcs8(kind, encryption))


@pytest.mark.parametrize(
    ("prefix", "dump"),
    [
        # PyYAML plain style folds the long line inside the header.
        ("The service account credential for the staging deployment is ", yaml.safe_dump),
        # Non-ASCII text makes double-quoted style the default; it folds with
        # an escaped line break and an escaped leading space.
        ("스테이징 배포용 서비스 계정 키는 다음과 같다 ", yaml.safe_dump),
        ("key: ", json.dumps),
    ],
    ids=["yaml-plain", "yaml-double-quoted", "json"],
)
@pytest.mark.parametrize("encryption", list(_ENCRYPTIONS))
def test_a_serialized_key_is_detected_wherever_the_line_folds(prefix, dump, encryption):
    pem = _pkcs8("ed25519", encryption)
    for pad in range(0, 40):
        value = " " * pad + prefix + pem
        serialized = dump({"note": value})
        # The fixture must stay a real key: the serialized text loads back to
        # the exact PEM, so a miss here is a key the guard let through.
        loaded = (yaml.safe_load if dump is yaml.safe_dump else json.loads)(serialized)
        assert loaded["note"] == value
        _assert_loadable(loaded["note"][len(" " * pad + prefix) :], encryption)
        assert privacy.scan(serialized), f"missed at pad={pad}: {serialized[:120]!r}"


def test_labelled_and_openssh_keys_are_still_detected():
    key = _key("rsa")
    for fmt in (ser.PrivateFormat.TraditionalOpenSSL, ser.PrivateFormat.OpenSSH):
        pem = key.private_bytes(ser.Encoding.PEM, fmt, ser.NoEncryption()).decode()
        assert privacy.scan(pem)


def _certificate() -> str:
    key = _key("ec")
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "example.invalid")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime.datetime(2026, 1, 1))
        .not_valid_after(datetime.datetime(2027, 1, 1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(ser.Encoding.PEM).decode()


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            lambda: (
                _key("ed25519")
                .public_key()
                .public_bytes(ser.Encoding.PEM, ser.PublicFormat.SubjectPublicKeyInfo)
                .decode()
            ),
            id="public-key",
        ),
        pytest.param(_certificate, id="certificate"),
        pytest.param(lambda: "Rotate first, then begin private key rotation.", id="prose"),
        pytest.param(lambda: "The core scan misses BEGIN PRIVATE KEY headers.", id="prose-caps"),
    ],
)
def test_public_material_and_prose_are_not_flagged(text):
    assert privacy.scan(text()) == []


@pytest.mark.parametrize("scope", ["user", "project_local", "project_shared"])
def test_the_guard_blocks_a_pkcs8_key_in_every_scope(scope):
    result = privacy.enforce_write_guard(_pkcs8("ed25519", "plain"), surface="t", scope=scope)
    assert result.decision == "blocked"


def test_project_shared_refuses_the_bypass_valve():
    result = privacy.enforce_write_guard(
        _pkcs8("ed25519", "plain"), surface="t", scope="project_shared", force_unsafe=True
    )
    assert result.decision == "blocked_project_shared"


@pytest.mark.asyncio
async def test_mem_add_refuses_a_pkcs8_key(bm25_only_components):
    comp, mem_dir = bm25_only_components
    target = mem_dir / "deploy.md"
    result = await mem_add(  # type: ignore[arg-type]
        content="Staging deploy key:\n" + _pkcs8("rsa", "plain"),
        file=str(target),
        ctx=StubCtx(AppContext.from_components(comp)),
    )
    assert "privacy pattern" in result
    assert not target.exists()


@pytest.fixture
def engine(components):
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(side_effect=lambda texts, **_: [[0.1] * 1024 for _ in texts])
    embedder.dimension = 1024
    components.index_engine._embedder = embedder
    return components.index_engine


def _note(token: str, *, key: str | None = None) -> str:
    body = f"# Notes\n\n## Entry\n\n{token} remembered fact.\n\n"
    if key is not None:
        body += f"## Deploy\n\n{key}\n"
    return body


def _hold_reason(components, path: Path) -> str | None:
    from memtomem.storage.sqlite_helpers import norm_path

    row = (
        components.storage._get_db()
        .execute("SELECT reason FROM held_sources WHERE source_file=?", (norm_path(path),))
        .fetchone()
    )
    return row[0] if row else None


@pytest.fixture
def downstream(engine, tmp_path, monkeypatch):
    """Turn on hosted enrichment and summaries, with a counting LLM stub.

    Enrichment needs a token budget, so the config gets a minimal byte-level
    tokenizer, as in ``test_bounded_chunks``.
    """
    tokenizers = pytest.importorskip("tokenizers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    class LLM:
        generate = AsyncMock(return_value="Describes the entry.")

    monkeypatch.setattr(
        engine,
        "_config",
        engine._config.model_copy(
            update={
                "hard_max_chunk_tokens": 64,
                "chunk_tokenizer_path": str(tokenizer_path),
                "chunk_context_tokens": 256,
                "chunk_model_tokens": 384,
                "enrich_chunk_context": True,
                "auto_summarize": True,
            }
        ),
    )
    monkeypatch.setattr(engine, "_llm", LLM())
    return LLM.generate


@pytest.mark.asyncio
@pytest.mark.parametrize("with_key", [False, True], ids=["control", "key"])
async def test_indexing_a_key_sends_nothing_downstream(
    components, engine, downstream, memory_dir, with_key
):
    # The control proves the embedder and the LLM are reachable from this
    # index pass; with a key in the file, the guard stops it before either.
    path = memory_dir / "deploy.md"
    key = _pkcs8("ed25519", "plain") if with_key else None
    path.write_text(_note("tokPK1", key=key), encoding="utf-8")

    if with_key:
        with pytest.raises(PrivacyRejection):
            await engine.index_file(path)
        engine._embedder.embed_texts.assert_not_awaited()
        downstream.assert_not_called()
        assert len(await components.storage.bm25_search("tokPK1")) == 0
    else:
        assert not (await engine.index_file(path)).errors
        engine._embedder.embed_texts.assert_awaited()
        assert downstream.call_count >= 2, "expected an enrichment and a summary call"


@pytest.mark.asyncio
async def test_a_key_indexed_before_the_fix_is_held_on_reindex(
    components, engine, memory_dir, monkeypatch
):
    # The upgrade path: the file was indexed while the guard could not see
    # the key, so its chunks — key included — are stored with a receipt.
    # Upgrading alone re-scans nothing; the next index pass over the file
    # runs the guard ahead of the receipt check and holds it.
    path = memory_dir / "deploy.md"
    path.write_text(_note("tokPK2", key=_pkcs8("rsa", "encrypted")), encoding="utf-8")
    header = "-----BEGIN ENCRYPTED PRIVATE KEY-----"
    vulnerable = tuple(p for p in privacy.DEFAULT_PATTERNS if not re.search(p, header))
    assert len(vulnerable) == len(privacy.DEFAULT_PATTERNS) - 1
    with monkeypatch.context() as m:
        m.setattr(privacy, "DEFAULT_PATTERNS", vulnerable)
        assert not (await engine.index_file(path)).errors
    assert len(await components.storage.bm25_search("tokPK2")) == 1

    # Same bytes, so only the guard can tell this pass from the last one.
    with pytest.raises(PrivacyRejection) as caught:
        await engine.index_file(path)

    assert caught.value.held is True
    assert _hold_reason(components, path) == REDACTION_BLOCKED_HOLD
    assert len(await components.storage.bm25_search("tokPK2")) == 0
