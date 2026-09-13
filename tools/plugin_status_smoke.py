#!/usr/bin/env python3
"""Exercise installed-wheel MCP status in isolated base or ONNX environments.

Run with the target venv's Python, never the workspace interpreter. ONNX checks
fetch only the pinned E5 tokenizer, not inference weights. All configuration,
cache, database and server registry files stay in a temporary directory.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


OPTIONAL_MODULES = ("huggingface_hub", "tokenizers", "fastembed", "onnxruntime")


def serve(root: Path) -> None:
    # Redirect home lookups before importing the installed server, without
    # changing the caller's HOME or any user configuration. Runtime ownership
    # uses an OS anchor separately, so give that its own temporary test anchor.
    original_expanduser = Path.expanduser

    def expanduser(path: Path) -> Path:
        if path.parts and path.parts[0] == "~":
            return root.joinpath(*path.parts[1:])
        return original_expanduser(path)

    with (
        patch.object(Path, "home", return_value=root),
        patch.object(Path, "expanduser", expanduser),
    ):
        import memtomem._runtime_paths as paths
        from memtomem.embedding.onnx import OnnxEmbedder

        def forbid_inference(*args, **kwargs):
            (root / "inference-attempted").touch()
            raise AssertionError("mem_status must not allocate an inference model")

        with (
            patch.object(paths, "runtime_dir", return_value=root / "runtime"),
            patch.object(paths, "candidate_runtime_dirs", return_value=[root / "runtime"]),
            patch.object(OnnxEmbedder, "_get_model", forbid_inference),
        ):
            from memtomem import _instance_registry as registry
            from memtomem.server import main

            # The registry imports path helpers by name. Check its actual bound
            # helpers before serving: status may garbage-collect old sentinels.
            assert registry.runtime_dir() == root / "runtime"
            assert registry.candidate_runtime_dirs() == [root / "runtime"]
            assert registry._candidate_registry_roots() == ([root / "runtime"], None)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            with patch.object(sys, "argv", ["memtomem-server"]):
                main()


async def check(profile: str, root: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    for module in OPTIONAL_MODULES:
        if profile == "onnx":
            assert importlib.util.find_spec(module) is not None, f"ONNX install lacks {module}"
        elif importlib.util.find_spec(module) is not None:
            raise AssertionError(f"base smoke requires an environment without {module}")

    provider = "onnx" if profile == "onnx" else "none"
    config_dir = root / ".memtomem"
    config_dir.mkdir()
    db = config_dir / "memtomem.db"
    config = {
        "embedding": {"provider": provider},
        "storage": {"sqlite_path": str(db)},
        "indexing": {"auto_discover": False, "memory_dirs": []},
        "warmup": {"enabled": False},
        "scheduler": {"enabled": False},
        "rerank": {"enabled": False},
    }
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    before = config_file.read_bytes()
    # Do not inherit developer config, project imports, or model cache paths.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("MEMTOMEM_", "_MEMTOMEM_", "HF_", "LANGFUSE_"))
        and k.upper() not in {"PYTHONPATH", "FASTEMBED_CACHE_PATH"}
    }
    env.update(
        MEMTOMEM_TOOL_MODE="core",
        MEMTOMEM_FASTEMBED_CACHE=str(root / "fastembed"),
        HF_HOME=str(root / "huggingface"),
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_XET="1",
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--serve", str(root)],
        cwd=root,
        env=env,
    )
    async with asyncio.timeout(180):
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert "mem_status" in {tool.name for tool in listing.tools}
                assert not db.exists(), "handshake must not initialize storage"
                result = await session.call_tool("mem_status", {})
                text = "\n".join(getattr(item, "text", "") for item in result.content)
                assert not result.is_error, text
                assert "Error:" not in text and "internal error" not in text, text
                assert f"{provider} /" in text, text
                assert "sqlite" in text and "Total chunks:" in text, text
                assert db.exists(), "status must initialize the isolated database"
                print(text)
    assert config_file.read_bytes() == before, "status rewrote the fixture configuration"
    assert not (root / "inference-attempted").exists(), "status attempted model inference"
    assert not list(root.rglob("*.onnx")), "status downloaded inference weights"
    if profile == "onnx":
        assert list((root / "fastembed").rglob("tokenizer.json")), "E5 tokenizer was not cached"
    print(f"plugin MCP status smoke OK ({profile}, isolated database, no inference)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("base", "onnx"), default="base")
    parser.add_argument("--serve", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
    else:
        with tempfile.TemporaryDirectory(prefix="memtomem-plugin-status-") as directory:
            asyncio.run(check(args.profile, Path(directory).resolve()))


if __name__ == "__main__":
    main()
