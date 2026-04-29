# Aerospace Colab RAG

항공우주 문서를 Colab 또는 로컬 Python에서 직접 호출해 쓰는 SmartFarm-style RAG 런타임입니다. `data/` 폴더의 지원 문서를 자동 수집하고, dense/sparse/image vector 검색, GraphRAG fallback, private overlay, DAT/QACT fusion을 제공합니다. 서버/컨테이너 배포 경계는 이 프로젝트 범위에서 제외합니다.

공개 repo에는 원본 데이터 파일을 포함하지 않습니다.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Docling, sentence-transformers, unstructured 기반 파싱/임베딩을 쓰려면 모델 extras를 추가 설치합니다.

```powershell
.venv\Scripts\pip install -r requirements-models.txt
```

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NAMUORI00/aerospace-rag/blob/main/notebooks/aerospace_rag_colab_ui.ipynb)

Open [notebooks/aerospace_rag_colab_ui.ipynb](notebooks/aerospace_rag_colab_ui.ipynb) in Colab and run cells top to bottom. The notebook clones this repo into `/content/aerospace-rag`, installs dependencies, prepares Ollama `gemma4:e2b`, asks you to place supported documents under `/content/aerospace-rag/data`, builds the local index, and prints a reproducibility report.

The notebook does not mount Google Drive. Put files directly into the Colab file panel under `aerospace-rag/data`.

## Data Files

The ingest step recursively scans `data/` and skips generated `data/index/` outputs.

Supported file types:

- `.pdf`
- `.docx`, `.pptx`
- `.xlsx`, `.xlsm`
- `.png`, `.jpg`, `.jpeg`, `.webp`
- `.txt`, `.md`

## Build Index

```powershell
.venv\Scripts\python -m aerospace_rag.cli.ingest --data-dir data
```

기본값은 `data/` 안의 지원 파일을 모두 읽습니다. 과거 샘플 5개 파일만 엄격히 검증하려면:

```powershell
.venv\Scripts\python -m aerospace_rag.cli.ingest --data-dir data --strict-expected
```

생성물:

- `data/index/qdrant`: Qdrant local persistent index or fallback dense/sparse/image vector file
- `data/index/falkordb/falkordb.db`: FalkorDBLite graph database when available
- `data/index/falkordb/graph_index.json`: graph retrieval helper index
- `data/index/bm25.json`: keyword retrieval index
- `data/index/chunks.jsonl`: normalized source chunks

## Query

```powershell
.venv\Scripts\python -m aerospace_rag.cli.query "H3 8호기 발사 실패 원인은?" --debug
```

Python direct-call API:

```python
from aerospace_rag.pipeline import build_index, ask

build_index(data_dir="data")
response = ask("위성영상 가격은 저장영상과 신규촬영에서 어떻게 다른가?", debug=True)
print(response.answer)
```

## LLM

생성 LLM은 Ollama로 고정합니다.

- `OLLAMA_BASE_URL`: default `http://127.0.0.1:11434`
- `OLLAMA_MODEL`: default `gemma4:e2b`
- `OLLAMA_API_KEY`: optional. Required only when `OLLAMA_BASE_URL=https://ollama.com`.

`ask()`는 항상 Ollama를 먼저 호출합니다. Ollama 서버 또는 모델 호출이 실패하면 외부 LLM으로 우회하지 않고 검색 근거를 extractive 형식으로 반환합니다. CLI의 `--provider extractive`는 LLM 없이 검색 결과만 확인하는 디버그 경로입니다.

Ollama Cloud 직접 호출 예:

```powershell
$env:OLLAMA_BASE_URL="https://ollama.com"
$env:OLLAMA_MODEL="gemma4:31b"
$env:OLLAMA_API_KEY="<your-ollama-api-key>"
```

## Runtime

- Ingest prefers Docling, falls back to unstructured, then dependency-light parsers for PDF/DOCX/PPTX/XLSX/image/text files.
- Dense retrieval uses `BAAI/bge-m3` through `sentence-transformers` when model extras are installed; otherwise it falls back to deterministic hash embeddings.
- Qdrant stores canonical payload keys such as `canonical_doc_id`, `canonical_chunk_id`, `doc_id`, and `chunk_id`.
- Vector retrieval exposes `dense_text`, `dense_image`, and `sparse` channels. If Qdrant is unavailable, the same channel API falls back to a local JSON vector store.
- GraphRAG tries FalkorDB path queries and falls back to `graph_index.json` with the same public/private scope rules.
- Knowledge extraction is deterministic by default. Set `EXTRACTOR_LLM_BACKEND=ollama` to use Ollama for entity/relation extraction, with deterministic fallback.
- Fusion uses QACT/DAT runtime profiles, candidate depth, evidence adjustment, and weighted RRF.
- Query diagnostics include `embedding_provider`, `embedding_model`, `channel_weights`, `weights_source`, `query_segment`, channel enablement, and private overlay status.

## Private Overlay

Private ingest updates SQLite overlay, vector index, chunks, and graph fallback together:

```powershell
.venv\Scripts\python -m aerospace_rag.cli.private_ingest "고객사 전용 H3 검토 메모" --farm-id tenant-a
.venv\Scripts\python -m aerospace_rag.cli.query "H3 검토 메모는?" --include-private --farm-id tenant-a --debug
```

Private 근거가 포함되어도 LLM 호출은 Ollama에서만 수행됩니다. 외부 OpenAI-compatible/vLLM provider 경로는 없습니다.

## Artifact Export / Import

직접 호출 환경에서도 인덱스 산출물 manifest를 만들 수 있습니다:

```powershell
.venv\Scripts\python -m aerospace_rag.artifacts.export --index-dir data/index --output-dir data/index/export
.venv\Scripts\python -m aerospace_rag.artifacts.importer --manifest data/index/export/artifact_manifest.json --index-dir data/index
```

Manifest에는 Qdrant, FalkorDB helper, BM25, chunks 파일 목록과 sha256이 들어갑니다.

기존 `aerospace_rag.ingest`, `aerospace_rag.query`, `aerospace_rag.private_ingest`, `aerospace_rag.artifact_export`, `aerospace_rag.artifact_import` 경로는 호환 래퍼로 유지됩니다.

## Package Layout

- `aerospace_rag.cli`: CLI entry points
- `aerospace_rag.ingestion`: document parsing and chunk creation
- `aerospace_rag.stores`: local index, vector, graph, and private overlay stores
- `aerospace_rag.retrieval`: BM25, embeddings, extraction, fusion, and runtime weights
- `aerospace_rag.generation`: answer generation providers
- `aerospace_rag.artifacts`: portable index artifact import/export

## FalkorDB Notes

`falkordblite` is used automatically on Linux/Colab because it can run FalkorDB from the Python environment without Docker. Its RedisLite dependency does not support native Windows, so Windows runs the same graph retrieval API with `graph_index.json` unless you point it at an existing FalkorDB server:

```powershell
$env:FALKORDB_HOST="localhost"
$env:FALKORDB_PORT="6379"
$env:FALKORDB_GRAPH="aerospace"
```
