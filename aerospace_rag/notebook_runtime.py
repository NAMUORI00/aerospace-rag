from __future__ import annotations

import html
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Iterable, Mapping, Sequence

from .config import Settings
from .generation.transformers_backend import ensure_transformers_model
from .ingestion import SUPPORTED_SUFFIXES, iter_supported_files
from .models import QueryResponse, RetrievalHit, SourceRef


REQUIRED_NOTEBOOK_PACKAGES = {
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "torch": "torch",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "docling": "docling",
    "openpyxl": "openpyxl",
    "pypdf": "pypdf",
    "ipywidgets": "ipywidgets",
    "nbformat": "nbformat",
}


def current_working_dir() -> Path:
    try:
        return Path.cwd()
    except FileNotFoundError:
        cwd_target = Path("/content") if Path("/content").exists() else Path.home()
        os.chdir(cwd_target)
        return Path.cwd()


def is_project_root(path: Path) -> bool:
    return (path / "aerospace_rag").is_dir() and (path / "notebooks").is_dir()


def git_output(*args: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return default


def ensure_valid_cwd(default_colab_root: Path, repo_url: str, in_colab: bool) -> Path:
    if in_colab:
        os.chdir(default_colab_root.parent)
        print("Policy: Project source is cloned fresh; Google Drive is optional for data only.")
        print("Running git clone:", repo_url)
        if default_colab_root.exists():
            shutil.rmtree(default_colab_root)
        subprocess.check_call(["git", "clone", repo_url, str(default_colab_root)])
        project_root = default_colab_root if is_project_root(default_colab_root) else None
    else:
        cwd = current_working_dir()
        project_root = next((candidate for candidate in [cwd, cwd.parent] if is_project_root(candidate)), None)
    if project_root is None:
        raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다. Colab git clone 또는 로컬 프로젝트 경로를 확인하세요.")

    for module_name in list(sys.modules):
        if module_name == "aerospace_rag" or module_name.startswith("aerospace_rag."):
            del sys.modules[module_name]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    return project_root


def package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def ensure_dependencies(project_root: Path, in_colab: bool) -> dict[str, str]:
    _ = (project_root, in_colab)
    missing = [package for module, package in REQUIRED_NOTEBOOK_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    else:
        print("Core dependencies already installed")
    snapshot = {package: package_version(package) for package in REQUIRED_NOTEBOOK_PACKAGES.values()}
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return snapshot


def ollama_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OLLAMA_API_KEY"]
    return headers


def ollama_api_ok() -> bool:
    try:
        req = urllib.request.Request(
            os.environ["OLLAMA_BASE_URL"].rstrip("/") + "/api/tags",
            headers=ollama_headers(),
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def mark_ollama_unavailable(reason: str) -> dict[str, object]:
    print("Ollama unavailable; generation backend remains Ollama.")
    print('ask() will raise until Ollama is ready; set ANSWER_PROVIDER = "extractive" for no-LLM debugging.')
    print("Reason:", reason)
    return {"ready": False, "reason": reason}


def ensure_transformers_runtime(llm_needed: bool) -> dict[str, object]:
    if not llm_needed:
        return {"ready": False, "reason": "LLM not requested"}
    try:
        status = ensure_transformers_model(Settings.from_env())
    except Exception as exc:
        print("Transformers runtime unavailable.")
        print("Reason:", exc)
        return {"ready": False, "reason": str(exc)}
    print("Transformers ready:", status["model"], "device_map:", status["device_map"])
    return status


def ensure_ollama_runtime(llm_needed: bool, *, in_colab: bool, pull_model: bool = True) -> dict[str, object]:
    if not llm_needed:
        return {"ready": False, "reason": "LLM not requested"}
    model = os.environ["OLLAMA_MODEL"]
    base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
    if base_url == "https://ollama.com":
        if ollama_api_ok():
            print("Ollama cloud ready:", base_url, "model:", model)
            return {"ready": True, "model": model, "base_url": base_url}
        return mark_ollama_unavailable("Ollama cloud API did not respond on " + base_url)

    if not in_colab:
        print("Local runtime: ensure Ollama is running separately.")
        print("Expected:", base_url, "model:", model)
        return {"ready": False, "reason": "local runtime requires external Ollama"}

    if shutil.which("ollama") is None:
        print("Installing Ollama...")
        try:
            if shutil.which("zstd") is None:
                print("Installing Ollama prerequisite: zstd")
                subprocess.check_call(["apt-get", "install", "-y", "zstd"])
            subprocess.check_call("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
        except Exception as exc:
            return mark_ollama_unavailable(f"Ollama install failed: {exc}")

    if not ollama_api_ok():
        print("Starting Ollama server...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as exc:
            return mark_ollama_unavailable(f"Ollama server start failed: {exc}")
        for _ in range(60):
            if ollama_api_ok():
                break
            time.sleep(1)

    if not ollama_api_ok():
        return mark_ollama_unavailable("Ollama server did not become ready on " + base_url)

    if pull_model:
        print("Pulling Ollama model:", model)
        try:
            subprocess.check_call(["ollama", "pull", model])
        except Exception as exc:
            return mark_ollama_unavailable(f"Ollama model pull failed: {exc}")

    print("Ollama ready:", base_url, "model:", model)
    return {"ready": True, "model": model, "base_url": base_url}


def import_google_drive_data(
    *,
    enabled: bool,
    source_dir: str | Path,
    data_dir: Path,
    in_colab: bool,
) -> list[str]:
    if not enabled:
        print("Google Drive data import skipped. Set USE_GOOGLE_DRIVE_DATA = True to copy files from Drive.")
        return []
    if not in_colab:
        raise RuntimeError("Google Drive data import is only available in Colab.")

    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    source_root = Path(source_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Google Drive data folder not found: {source_root}")

    data_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source_path in sorted(source_root.rglob("*")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source_root)
        target_path = data_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied.append(relative.as_posix())

    print(f"Copied {len(copied)} files from Google Drive to {data_dir}")
    for name in copied[:20]:
        print("-", name)
    if len(copied) > 20:
        print(f"... and {len(copied) - 20} more")
    return copied


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_data_files(data_dir: Path) -> list[dict[str, object]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in iter_supported_files(data_dir):
        entry = {
            "name": path.relative_to(data_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        manifest.append(entry)
    return manifest


def _location_label(*, page: int | None = None, sheet: str | None = None, row: int | None = None) -> str:
    if page is not None:
        return f"p.{page}"
    if sheet and row is not None:
        return f"{sheet}:{row}"
    if sheet:
        return sheet
    if row is not None:
        return f"row {row}"
    return "table"


def _channel_text(channels: Mapping[str, float] | None) -> str:
    if not channels:
        return "-"
    return ", ".join(f"{name}={score:.3f}" for name, score in channels.items())


def _clean_inline_text(text: object, *, max_chars: int | None = None) -> str:
    value = " ".join(str(text).split())
    if max_chars is None or len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _clean_answer_line(text: object) -> str:
    value = str(text).strip()
    value = re.sub(r"^\s*#+\s+", "", value)
    value = re.sub(r"^\s*[-*+]\s+", "", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    return " ".join(value.split())


def _summarize_answer_for_table(answer: object, *, max_chars: int = 220) -> str:
    lines = [_clean_answer_line(line) for line in str(answer or "").splitlines()]
    meaningful = [line for line in lines if line]
    if not meaningful:
        return "-"
    summary_parts = [meaningful[0]]
    if summary_parts[0].endswith(":") and len(meaningful) > 1:
        summary_parts.append(meaningful[1])
    return _clean_inline_text(" ".join(summary_parts), max_chars=max_chars)


def _json_details(title: str, payload: object) -> str:
    return (
        f"<details>\n<summary>{html.escape(title)}</summary>\n\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```\n"
        "</details>"
    )


def format_retrieval_markdown(
    question: str,
    hits: Sequence[RetrievalHit],
    diagnostics: Mapping[str, object] | None = None,
) -> str:
    diagnostics = diagnostics or {}
    parts = [
        "### 검색 요약",
        f"- 질문: {question}",
        f"- 검색 결과 수: {len(hits)}",
    ]
    channels = diagnostics.get("channels")
    if isinstance(channels, Iterable) and not isinstance(channels, (str, bytes, dict)):
        parts.append(f"- 활성 채널: {', '.join(str(channel) for channel in channels)}")
    if hits:
        lines = ["### 상위 결과"]
        for idx, hit in enumerate(hits, start=1):
            lines.append(
                f"{idx}. `{hit.chunk.source_file}` ({_location_label(page=hit.chunk.page, sheet=hit.chunk.sheet, row=hit.chunk.row)}) "
                f"score={hit.score:.3f}"
            )
            lines.append(f"   - channels: {_channel_text(hit.channels)}")
            lines.append(f"   - excerpt: {_clean_inline_text(hit.chunk.text, max_chars=280)}")
        parts.append("\n".join(lines))
    if diagnostics:
        parts.append(_json_details("Retrieval diagnostics", diagnostics))
    return "\n\n".join(parts)


def format_answer_markdown(response: QueryResponse, *, max_sources: int = 3) -> str:
    answer = str(response.answer or "").strip() or "_답변이 비어 있습니다._"
    parts = ["### 답변", answer]
    if response.sources:
        lines = ["### 상위 근거"]
        for idx, source in enumerate(response.sources[:max_sources], start=1):
            lines.append(
                f"{idx}. `{source.source_file}` ({_location_label(page=source.page, sheet=source.sheet, row=source.row)}) "
                f"score={source.score:.3f}"
            )
            lines.append(f"   - channels: {_channel_text(source.channels)}")
        parts.append("\n".join(lines))
    if response.routing:
        parts.append(_json_details("Routing", response.routing))
    if response.diagnostics:
        parts.append(_json_details("Diagnostics", response.diagnostics))
    return "\n\n".join(parts)


def format_sources_markdown(sources: Sequence[SourceRef]) -> str:
    if not sources:
        return "### 근거 상세\n\n_근거가 없습니다._"
    parts = ["### 근거 상세"]
    for idx, source in enumerate(sources, start=1):
        excerpt_lines = str(source.excerpt or "").strip().splitlines() or ["(excerpt 없음)"]
        quoted_excerpt = "\n".join(f"> {line}" for line in excerpt_lines)
        parts.append(
            f"#### {idx}. `{source.source_file}` ({_location_label(page=source.page, sheet=source.sheet, row=source.row)})"
        )
        parts.append(f"- score: {source.score:.3f}")
        parts.append(f"- channels: {_channel_text(source.channels)}")
        parts.append(quoted_excerpt)
    return "\n\n".join(parts)


def build_response_row(question: str, response: QueryResponse, *, case: int | None = None) -> dict[str, object]:
    source_files: list[str] = []
    for source in response.sources:
        if source.source_file not in source_files:
            source_files.append(source.source_file)
    channels = response.diagnostics.get("channels")
    row: dict[str, object] = {
        "question": question,
        "summary": _summarize_answer_for_table(response.answer, max_chars=220),
        "provider": response.routing.get("provider") or response.diagnostics.get("provider") or "-",
        "source_count": len(response.sources),
        "channels": ", ".join(str(channel) for channel in channels) if isinstance(channels, list) else "-",
        "top_source": response.sources[0].source_file if response.sources else "-",
        "top_score": round(response.sources[0].score, 4) if response.sources else "-",
        "source_files": ", ".join(source_files) if source_files else "-",
    }
    if case is not None:
        row["case"] = case
    return row


NOTEBOOK_COLUMN_LABELS = {
    "case": "Case",
    "question": "Question",
    "summary": "Answer Summary",
    "provider": "Provider",
    "source_count": "Sources",
    "channels": "Channels",
    "top_source": "Top Source",
    "top_score": "Top Score",
    "source_files": "Source Files",
}


def format_results_table(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "<p><em>No results</em></p>"
    ordered_columns = list(columns or rows[0].keys())
    header_html = "".join(
        f"<th style='padding:10px; border-bottom:1px solid #d0d7de; text-align:left; background:#f6f8fa;'>"
        f"{html.escape(NOTEBOOK_COLUMN_LABELS.get(column, column.replace('_', ' ').title()))}</th>"
        for column in ordered_columns
    )
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for column in ordered_columns:
            value = row.get(column, "-")
            if isinstance(value, float):
                text = f"{value:.4f}"
            elif isinstance(value, (list, tuple, set)):
                text = ", ".join(str(item) for item in value)
            elif value is None or value == "":
                text = "-"
            else:
                text = str(value)
            safe_text = html.escape(text).replace("\n", "<br>")
            cells.append(
                "<td style='padding:10px; border-bottom:1px solid #eaecf0; vertical-align:top;'>"
                f"{safe_text}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div style='overflow-x:auto; margin:8px 0 16px;'>"
        "<table style='border-collapse:collapse; width:100%; font-size:14px; line-height:1.5;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _storage_short(value: object, *, max_chars: int = 90) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _storage_size(num: int | float) -> str:
    value = float(num or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _storage_artifact_stats(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return False, 0, 0
    if path.is_file():
        return True, 1, path.stat().st_size
    files = [child for child in path.rglob("*") if child.is_file()]
    return True, len(files), sum(child.stat().st_size for child in files)


def _storage_html_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
            + "</tr>"
        )
    return (
        "<style>"
        ".ragviz table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45;margin:8px 0 18px;}"
        ".ragviz th{background:#f6f8fa;border-bottom:1px solid #d0d7de;padding:8px;text-align:left;}"
        ".ragviz td{border-bottom:1px solid #eaecf0;padding:8px;vertical-align:top;}"
        ".ragviz code{background:#f6f8fa;padding:1px 4px;border-radius:4px;}"
        "</style><div class='ragviz'><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _storage_qdrant_sections(qdrant_dir: Path, collection_name: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    qdrant_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    if not qdrant_dir.exists():
        qdrant_rows.append(
            {
                "collection": collection_name,
                "collections_on_disk": "",
                "points_count": "",
                "vectors_count": "",
                "vector_config": "",
                "sparse_config": f"Qdrant directory not found: {qdrant_dir}",
            }
        )
        return qdrant_rows, point_rows

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(qdrant_dir))
        try:
            collections = [collection.name for collection in client.get_collections().collections]
            info = client.get_collection(collection_name)
            params = getattr(info.config, "params", None)
            qdrant_rows.append(
                {
                    "collection": collection_name,
                    "collections_on_disk": ", ".join(collections),
                    "points_count": getattr(info, "points_count", ""),
                    "vectors_count": getattr(info, "vectors_count", ""),
                    "vector_config": _storage_short(getattr(params, "vectors", None), max_chars=220),
                    "sparse_config": _storage_short(getattr(params, "sparse_vectors", None), max_chars=220),
                }
            )
            points, _ = client.scroll(collection_name=collection_name, limit=8, with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                location = "/".join(str(payload.get(key) or "") for key in ["page", "sheet", "row"]).strip("/")
                point_rows.append(
                    {
                        "point_id": str(point.id),
                        "chunk_id": _storage_short(payload.get("chunk_id"), max_chars=70),
                        "source_file": payload.get("source_file") or payload.get("source_doc") or "",
                        "modality": payload.get("modality", ""),
                        "page/sheet/row": location,
                        "payload_keys": ", ".join(sorted(payload.keys())[:14]),
                        "text_preview": _storage_short(payload.get("text"), max_chars=140),
                    }
                )
        finally:
            client.close()
    except Exception as exc:
        qdrant_rows.append(
            {
                "collection": collection_name,
                "collections_on_disk": "",
                "points_count": "",
                "vectors_count": "",
                "vector_config": "",
                "sparse_config": f"Qdrant inspect failed: {type(exc).__name__}: {exc}",
            }
        )
    return qdrant_rows, point_rows


def _storage_graph_payload(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        return {}
    return json.loads(graph_path.read_text(encoding="utf-8"))


def _storage_graph_rows(graph: Mapping[str, Any]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entity_to_chunks = {key: list(value) for key, value in dict(graph.get("entity_to_chunks") or {}).items()}
    entity_text = dict(graph.get("entity_text") or {})
    entity_types = dict(graph.get("entity_types") or {})
    relations = list(graph.get("relations") or [])
    chunks = dict(graph.get("chunks") or {})
    summary_rows = [
        {
            "schema_version": graph.get("schema_version", ""),
            "entities": len(entity_to_chunks),
            "relations": len(relations),
            "chunks": len(chunks),
            "entity_to_chunk_edges": sum(len(value) for value in entity_to_chunks.values()),
        }
    ]
    top_entity_ids = sorted(entity_to_chunks, key=lambda entity: (-len(entity_to_chunks[entity]), entity_text.get(entity, entity)))[:14]
    entity_rows = []
    for entity_id in top_entity_ids:
        entity_rows.append(
            {
                "entity_id": entity_id,
                "label": entity_text.get(entity_id, entity_id),
                "type": entity_types.get(entity_id, ""),
                "chunk_count": len(entity_to_chunks.get(entity_id, [])),
                "sample_chunks": ", ".join(
                    _storage_short(chunks.get(chunk_id, {}).get("source_file", chunk_id), max_chars=45)
                    for chunk_id in entity_to_chunks.get(entity_id, [])[:3]
                ),
            }
        )
    return summary_rows, entity_rows


def _storage_relationship_svg(graph: Mapping[str, Any]) -> str:
    entity_to_chunks = {key: list(value) for key, value in dict(graph.get("entity_to_chunks") or {}).items()}
    chunk_entities = {key: list(value) for key, value in dict(graph.get("chunk_entities") or {}).items()}
    entity_text = dict(graph.get("entity_text") or {})
    entity_types = dict(graph.get("entity_types") or {})
    relations = list(graph.get("relations") or [])
    chunks = dict(graph.get("chunks") or {})
    top_entity_ids = sorted(entity_to_chunks, key=lambda entity: (-len(entity_to_chunks[entity]), entity_text.get(entity, entity)))[:14]
    top_chunk_ids = sorted(
        chunk_entities,
        key=lambda chunk: (-len(chunk_entities[chunk]), chunks.get(chunk, {}).get("source_file", "")),
    )[:10]
    selected_entities = set(top_entity_ids)
    selected_chunks = set(top_chunk_ids)
    for entity_id in top_entity_ids:
        selected_chunks.update(entity_to_chunks.get(entity_id, [])[:2])

    width, height = 1160, 720
    artifact_nodes = [
        ("qdrant", 100, 110, "Qdrant\nvector+payload"),
        ("chunks", 100, 230, "chunks.jsonl\ncanonical payload"),
        ("bm25", 100, 350, "bm25.json\nkeyword docs"),
        ("graph", 100, 470, "graph_index.json\nentities+relations"),
    ]
    chunk_list = list(selected_chunks)[:12]
    entity_list = list(selected_entities)[:14]
    chunk_pos = {}
    entity_pos = {}
    for idx, chunk_id in enumerate(chunk_list):
        chunk_pos[chunk_id] = (470, 70 + idx * (600 / max(1, len(chunk_list) - 1)))
    for idx, entity_id in enumerate(entity_list):
        entity_pos[entity_id] = (890, 60 + idx * (620 / max(1, len(entity_list) - 1)))

    svg = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='720' "
        "style='background:#fbfbfd;border:1px solid #d0d7de;border-radius:8px;'>",
        "<defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='3' orient='auto'>"
        "<path d='M0,0 L0,6 L8,3 z' fill='#778'/></marker></defs>",
    ]
    for name, x, y, _label in artifact_nodes:
        target_x = 360 if name != "graph" else 760
        target_y = 340 if name != "graph" else 360
        svg.append(
            f"<line x1='{x + 95}' y1='{y}' x2='{target_x}' y2='{target_y}' "
            "stroke='#c8ccd4' stroke-width='2' marker-end='url(#arrow)'/>"
        )
    for entity_id in entity_list:
        ex, ey = entity_pos[entity_id]
        for chunk_id in entity_to_chunks.get(entity_id, []):
            if chunk_id in chunk_pos:
                cx, cy = chunk_pos[chunk_id]
                svg.append(
                    f"<line x1='{cx + 110}' y1='{cy}' x2='{ex - 105}' y2='{ey}' "
                    "stroke='#9ab' stroke-width='1.4' opacity='0.55'/>"
                )
    for relation in relations[:300]:
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        if source in entity_pos and target in entity_pos and source != target:
            x1, y1 = entity_pos[source]
            x2, y2 = entity_pos[target]
            svg.append(
                f"<path d='M{x1 + 80},{y1} C{x1 + 160},{y1} {x2 + 160},{y2} {x2 + 80},{y2}' "
                "stroke='#e09f3e' stroke-width='1.2' fill='none' opacity='0.45'/>"
            )
    for _name, x, y, label in artifact_nodes:
        title, subtitle = label.split("\n")
        svg.append(f"<rect x='{x - 80}' y='{y - 34}' width='185' height='68' rx='8' fill='#eef6ff' stroke='#6ea8fe'/>")
        svg.append(f"<text x='{x + 12}' y='{y - 8}' text-anchor='middle' font-size='13' font-weight='700' fill='#17324d'>{html.escape(title)}</text>")
        svg.append(f"<text x='{x + 12}' y='{y + 14}' text-anchor='middle' font-size='11' fill='#4f6173'>{html.escape(subtitle)}</text>")
    for chunk_id, (x, y) in chunk_pos.items():
        payload = dict(chunks.get(chunk_id, {}))
        label = payload.get("source_file") or chunk_id
        modality = payload.get("modality") or "chunk"
        svg.append(f"<rect x='{x - 115}' y='{y - 24}' width='230' height='48' rx='7' fill='#eefaf0' stroke='#6abf69'/>")
        svg.append(f"<text x='{x}' y='{y - 5}' text-anchor='middle' font-size='11' font-weight='700' fill='#153b1c'>{html.escape(_storage_short(label, max_chars=32))}</text>")
        svg.append(f"<text x='{x}' y='{y + 13}' text-anchor='middle' font-size='10' fill='#47704a'>{html.escape(modality)} | {html.escape(_storage_short(chunk_id, max_chars=30))}</text>")
    for entity_id, (x, y) in entity_pos.items():
        label = entity_text.get(entity_id, entity_id)
        entity_type = entity_types.get(entity_id, "entity")
        count = len(entity_to_chunks.get(entity_id, []))
        svg.append(f"<rect x='{x - 105}' y='{y - 24}' width='210' height='48' rx='24' fill='#fff4e6' stroke='#f0ad4e'/>")
        svg.append(f"<text x='{x}' y='{y - 5}' text-anchor='middle' font-size='11' font-weight='700' fill='#573b0a'>{html.escape(_storage_short(label, max_chars=28))}</text>")
        svg.append(f"<text x='{x}' y='{y + 13}' text-anchor='middle' font-size='10' fill='#84621f'>{html.escape(entity_type)} | chunks={count}</text>")
    svg.append("<text x='100' y='35' text-anchor='middle' font-size='14' font-weight='700' fill='#334'>Artifacts</text>")
    svg.append("<text x='470' y='35' text-anchor='middle' font-size='14' font-weight='700' fill='#334'>Chunks / Qdrant payload points</text>")
    svg.append("<text x='890' y='35' text-anchor='middle' font-size='14' font-weight='700' fill='#334'>Graph entities / relations</text>")
    svg.append("</svg>")
    return "".join(svg)


def format_storage_visualization(
    index_dir: str | Path,
    *,
    build_report: Mapping[str, object] | None = None,
    collection_name: str = "aerospace_chunks",
) -> list[dict[str, str]]:
    resolved_index_dir = Path(index_dir)
    build_report = build_report or {}
    resolved_collection = str(build_report.get("qdrant_collection") or collection_name)
    qdrant_dir = resolved_index_dir / "qdrant"
    graph_path = resolved_index_dir / "graph" / "graph_index.json"
    chunks_path = resolved_index_dir / "chunks.jsonl"
    bm25_path = resolved_index_dir / "bm25.json"
    artifacts = [
        ("qdrant", qdrant_dir, "dense/sparse vectors + chunk payload point storage"),
        ("chunks.jsonl", chunks_path, "canonical chunk payload store"),
        ("bm25.json", bm25_path, "tokenized sparse keyword index"),
        ("graph/graph_index.json", graph_path, "entity, relation, entity_to_chunks graph store"),
    ]
    artifact_rows = []
    for name, path, role in artifacts:
        exists, file_count, byte_count = _storage_artifact_stats(path)
        artifact_rows.append(
            {
                "artifact": name,
                "exists": exists,
                "files": file_count,
                "size": _storage_size(byte_count),
                "path": str(path),
                "role": role,
            }
        )

    qdrant_rows, point_rows = _storage_qdrant_sections(qdrant_dir, resolved_collection)
    graph = _storage_graph_payload(graph_path)
    graph_summary_rows, entity_rows = _storage_graph_rows(graph)
    return [
        {"type": "markdown", "content": "### Qdrant / Graph storage visualization"},
        {"type": "html", "content": _storage_html_table(artifact_rows, ["artifact", "exists", "files", "size", "path", "role"])},
        {"type": "markdown", "content": "#### Qdrant collection metadata / vector settings"},
        {
            "type": "html",
            "content": _storage_html_table(
                qdrant_rows,
                ["collection", "collections_on_disk", "points_count", "vectors_count", "vector_config", "sparse_config"],
            ),
        },
        {"type": "markdown", "content": "#### Qdrant point payload sample"},
        {
            "type": "html",
            "content": _storage_html_table(
                point_rows,
                ["point_id", "chunk_id", "source_file", "modality", "page/sheet/row", "payload_keys", "text_preview"],
            ),
        },
        {"type": "markdown", "content": "#### Graph index summary"},
        {
            "type": "html",
            "content": _storage_html_table(
                graph_summary_rows,
                ["schema_version", "entities", "relations", "chunks", "entity_to_chunk_edges"],
            ),
        },
        {"type": "markdown", "content": "#### Top graph entities"},
        {"type": "html", "content": _storage_html_table(entity_rows, ["entity_id", "label", "type", "chunk_count", "sample_chunks"])},
        {"type": "markdown", "content": "#### Storage relationship map"},
        {"type": "html", "content": _storage_relationship_svg(graph)},
    ]
