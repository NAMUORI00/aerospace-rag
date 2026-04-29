# Aerospace Local RAG

Docker Compose 없이 Python 가상환경에서 실행하는 항공우주 RAG 프로젝트입니다. v1은 사용자가 `data/` 폴더에 수동 업로드한 5개 파일을 데이터셋으로 사용합니다. 공개 repo에는 원본 데이터 파일을 포함하지 않습니다.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NAMUORI00/aerospace-rag/blob/main/notebooks/aerospace_rag_colab_ui.ipynb)

The notebook is designed as a reproducibility walkthrough. It uses GitHub clone only for project setup in Colab and does not mount or reuse Google Drive.

```python
GITHUB_REPO_URL = "https://github.com/NAMUORI00/aerospace-rag.git"
```

Open [notebooks/aerospace_rag_colab_ui.ipynb](notebooks/aerospace_rag_colab_ui.ipynb) in Colab and run cells top to bottom. The notebook is split into environment, source clone, dependency, runtime, data, configuration, ingestion, indexing, retrieval-only, LLM answer, evidence, repeated questions, reproducibility report, and troubleshooting sections. It will move to `/content`, delete any existing `/content/aerospace-rag`, clone the repo into `/content/aerospace-rag`, install Python dependencies, start Ollama, pull `gemma4:e2b`, ask you to upload the 5 data files through the browser if missing, build the local Qdrant/BM25/FalkorDB-helper index, and print a reproducibility report with commit, package versions, data hashes, runtime config, and index paths.

## Data Files

The public repository keeps only `data/.gitkeep` and `data/README.md`. Copy or upload these files into `data/` before building the index:

- `251222_H3 8호기 발사 경과.pdf`
- `NASA awards Momentus contract for solar sail demonstration study(영).pdf`
- `위성영상가격.png`
- `인공위성_질문응답.xlsx`
- `해외정부 우주항공 현황.png`

For SmartFarm-like retrieval quality, install the embedding model extras:

```powershell
.venv\Scripts\pip install -r requirements-models.txt
```

The default runtime contract is documented in `.env.example`. The defaults are:

- Embedding: `BAAI/bge-m3`
- DAT mode: `hybrid`
- Temporary LLM: Ollama `gemma4:e2b`

## Build Index

```powershell
.venv\Scripts\python -m aerospace_rag.ingest --data-dir data
```

생성물:

- `data/index/qdrant`: Qdrant local persistent index
- `data/index/falkordb/falkordb.db`: FalkorDBLite graph database
- `data/index/falkordb/graph_index.json`: graph retrieval helper index
- `data/index/bm25.json`: keyword retrieval index
- `data/index/chunks.jsonl`: normalized source chunks

## Query

```powershell
.venv\Scripts\python -m aerospace_rag.query "H3 8호기 발사 실패 원인은?" --debug
```

Python API:

```python
from aerospace_rag.pipeline import build_index, ask

build_index(data_dir="data")
response = ask("위성영상 가격은 저장영상과 신규촬영에서 어떻게 다른가?", debug=True)
print(response.answer)
```

## LLM Providers

- `ollama`: 기본값. `OLLAMA_BASE_URL`의 Ollama `/api/chat` endpoint로 `OLLAMA_MODEL=gemma4:e2b`를 호출합니다.
- `openai_compatible`: OpenAI-compatible endpoint가 따로 있을 때만 사용합니다.
- `extractive`: 외부 모델 없이 검색 근거를 요약 형식으로 반환합니다.

검색 자체는 항상 Qdrant, BM25, FalkorDB graph channel을 사용합니다.

Colab에서 Ollama를 쓰려면 노트북의 Ollama 준비 셀이 `ollama` 설치, `ollama serve`, `ollama pull gemma4:e2b`를 처리합니다. 로컬에서는 Ollama를 먼저 실행해 둔 뒤 질의하세요.

## SmartFarm-Like Runtime

The Python-only runtime now mirrors the SmartFarm internals more closely:

- Qdrant stores SmartFarm-style canonical payload keys such as `canonical_doc_id`, `canonical_chunk_id`, `doc_id`, and `chunk_id`.
- Dense retrieval uses `BAAI/bge-m3` through `sentence-transformers` when model extras are installed; otherwise it falls back to deterministic hash embeddings so tests and notebook smoke runs still work.
- Fusion uses DAT-style query segmentation and channel diagnostics instead of a single fixed weight.
- Query diagnostics include `embedding_provider`, `embedding_model`, `channel_weights`, `weights_source`, `query_segment`, and channel enablement.

## FalkorDB Notes

`falkordblite` is used automatically on Linux/Colab because it can run FalkorDB from the Python environment without Docker. Its RedisLite dependency does not support native Windows, so Windows runs the same graph retrieval API with `graph_index.json` unless you point it at an existing FalkorDB server:

```powershell
$env:FALKORDB_HOST="localhost"
$env:FALKORDB_PORT="6379"
$env:FALKORDB_GRAPH="aerospace"
```
