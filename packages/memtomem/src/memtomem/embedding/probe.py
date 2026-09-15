"""Embed new document text to compare it against stored passages (#2461).

Similarity probes — conflict neighbours, formation evidence, duplicate checks —
hold *document* text, so they must be embedded the way stored rows were. Two
existing calls get that wrong in opposite directions under an asymmetric model
like E5: ``embed_query`` uses the query role, and ``embed_texts`` uses the right
role but refuses over-budget input because it is the ingress path. A probe
vector is never stored, so truncating it is acceptable where it is not for a
stored one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memtomem.errors import EmbeddingError

if TYPE_CHECKING:
    from memtomem.embedding.base import EmbeddingProvider


def _probe_capability(embedder: object):
    """Return the provider's ``embed_probe`` only when it genuinely defines one.

    ``Mock``/``AsyncMock`` fabricate any attribute on access, so a plain
    ``getattr`` would treat every test double as a probe-capable provider. Same
    rule as ``runtime.publish_onnx_batch_size``: a class method or an explicitly
    attached instance attribute counts.
    """
    class_method = getattr(type(embedder), "embed_probe", None)
    instance_dict = getattr(embedder, "__dict__", {})
    instance_method = instance_dict.get("embed_probe") if isinstance(instance_dict, dict) else None
    if not (callable(class_method) or callable(instance_method)):
        return None
    method = getattr(embedder, "embed_probe", None)
    return method if callable(method) else None


async def embed_document_probe(embedder: EmbeddingProvider, text: str) -> list[float]:
    """Embed ``text`` as a document for a similarity lookup that is never stored.

    Raises ``EmbeddingError`` on empty or whitespace-only input: a meaningless
    vector would still dense-search to *some* nearest row and present it as a
    neighbour.
    """
    if not text or not text.strip():
        raise EmbeddingError("Probe text cannot be empty")
    probe = _probe_capability(embedder)
    if probe is not None:
        return await probe(text)
    embeddings = await embedder.embed_texts([text])
    if not embeddings:
        raise EmbeddingError("No embeddings returned for probe")
    return embeddings[0]
