# Aerospace Colab RAG

항공우주 문서를 Colab 또는 로컬 Python에서 직접 호출해 쓰는 RAG 런타임입니다. `data/` 폴더의 지원 문서를 자동 수집하고, Qdrant dense/sparse/image 검색, JSON graph-lite 검색, 고정 weighted RRF, Ollama 답변 생성을 제공합니다. 서버/컨테이너 배포와 외부 LLM provider 운영은 이 프로젝트 범위에서 제외합니다.

공개 repo에는 원본 데이터 파일을 포함하지 않습니다.

## Goal

이 repo의 목표는 기업 관계자가 GitHub에서 인증된 계정으로 바로 열 수 있는 Google Colab T4 프로토타입을 제공하는 것입니다. Colab 노트북은 항공우주 업무 문서를 업로드한 뒤 파싱, 청킹, Qdrant/BM25/graph-lite 검색, weighted RRF 결합, Ollama 답변 생성, 근거 확인까지 한 화면에서 재현하도록 구성되어 있습니다.

기대 출력은 다음과 같습니다.

- `data/index/qdrant`, `data/index/graph/graph_index.json`, `data/index/bm25.json`, `data/index/chunks.jsonl` 인덱스 산출물
- 검색 단독 검증에서 채널별 근거와 diagnostics
- LLM 답변, 상위 근거, 반복 질문 결과표
- 실제 업무 파일용 질문 3개 이상의 답변/출처/검색 채널 표시

## Lineage

이 프로젝트는 [NAMUORI00/smartfarm-workspace](https://github.com/NAMUORI00/smartfarm-workspace)의 Dense/Sparse/Graph 3채널 RAG 아이디어를 항공우주 문서와 Colab T4 데모 환경에 맞게 줄인 포팅판입니다.

- 유지한 개념: dense 검색, sparse/BM25 검색, graph 기반 보강, weighted RRF, 근거 중심 답변, 재현 가능한 실행 흐름
- 대체한 부분: Docker compose, submodule 워크스페이스, FalkorDB 운영 구조를 단일 Python 패키지와 JSON graph-lite 인덱스로 축소
- 범위 밖: 장기 운영 서버, 컨테이너 배포, 외부 LLM provider 확장, 기업 내부 데이터 영구 저장

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

기본 임베딩과 `.docx`/`.pptx` 파싱을 쓰려면 모델 extras를 추가 설치합니다.

```powershell
.venv\Scripts\pip install -r requirements-models.txt
```

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NAMUORI00/aerospace-rag/blob/main/notebooks/aerospace_rag_colab_ui.ipynb)

Open [notebooks/aerospace_rag_colab_ui.ipynb](notebooks/aerospace_rag_colab_ui.ipynb) in Colab and run cells top to bottom. The notebook clones this repo into `/content/aerospace-rag`, installs dependencies, prepares Ollama `gemma4:e4b`, asks you to place supported documents under `/content/aerospace-rag/data`, builds the local index, and runs retrieval/answer checks. Its indexing default is strict Ollama extraction with one-hour timeout limits, JSON Schema structured output, one Ollama repair attempt for malformed JSON, and no local fallback.

By default, put files directly into the Colab file panel under `aerospace-rag/data`.

Colab T4 execution checklist:

1. Runtime type: GPU, T4.
2. Run every cell from the top.
3. Upload supported files into `/content/aerospace-rag/data` when the data-prep section asks for them.
4. Confirm the index section reports file/chunk counts and the four index artifacts above.
5. Confirm the answer sections show answer text, source files, scores, and diagnostics.

If a run fails:

- Missing package: rerun the dependency cell.
- Ollama unavailable: rerun the Ollama runtime cell, or use `ANSWER_PROVIDER = "extractive"` only for no-LLM retrieval debugging.
- No data found: upload files under `/content/aerospace-rag/data`, not Drive or the notebook root.
- Model pull timeout: keep the T4 runtime active and rerun the Ollama cell before rebuilding the index.

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

- `data/index/qdrant`: Qdrant local persistent index
- `data/index/graph/graph_index.json`: graph-lite retrieval helper index
- `data/index/bm25.json`: keyword retrieval index
- `data/index/chunks.jsonl`: normalized source chunks

## Query

```powershell
.venv\Scripts\python -m aerospace_rag.cli.query "H3 8호기 발사 실패 원인은?" --debug
```

Python direct-call API:

```python
from aerospace_rag import ask, build_index

build_index(data_dir="data")
response = ask("위성영상 가격은 저장영상과 신규촬영에서 어떻게 다른가?", debug=True)
print(response.answer)
```

## LLM

생성 LLM은 Ollama로 고정합니다.

- `OLLAMA_BASE_URL`: default `http://127.0.0.1:11434`
- `OLLAMA_MODEL`: default `gemma4:e4b`
- `OLLAMA_API_KEY`: optional. Required only when `OLLAMA_BASE_URL=https://ollama.com`.

`ask()`는 기본적으로 Ollama를 호출합니다. Ollama 서버 또는 모델 호출이 실패하면 자동으로 extractive 답변으로 우회하지 않고 명확한 오류를 냅니다. CLI의 `--provider extractive` 또는 `ask(..., provider="extractive")`는 LLM 없이 검색 결과만 확인하는 명시적 디버그 경로입니다.

Ollama Cloud 직접 호출 예:

```powershell
$env:OLLAMA_BASE_URL="https://ollama.com"
$env:OLLAMA_MODEL="gemma4:31b"
$env:OLLAMA_API_KEY="<your-ollama-api-key>"
```

## Runtime

- PDF/XLSX/image/text ingest uses direct lightweight parsers. `.docx`/`.pptx` requires Docling and fails clearly if it is not installed.
- Dense retrieval uses `BAAI/bge-m3` through `sentence-transformers` by default. `AEROSPACE_EMBED_BACKEND=hash` is an explicit debug mode, not an automatic fallback.
- Qdrant stores canonical payload keys such as `canonical_doc_id`, `canonical_chunk_id`, `doc_id`, and `chunk_id`.
- Vector retrieval exposes `dense_text`, `dense_image`, and `sparse` channels. `AEROSPACE_VECTOR_BACKEND=json` is an explicit debug mode for tests and lightweight smoke runs.
- Graph retrieval is graph-lite only and reads `data/index/graph/graph_index.json`.
- Knowledge extraction uses Ollama by default. Colab sets `EXTRACTOR_LLM_BACKEND=ollama`, `OLLAMA_EXTRACT_TIMEOUT_SECONDS=3600`, `OLLAMA_GENERATE_TIMEOUT_SECONDS=3600`, `OLLAMA_EXTRACT_RETRIES=1`, `OLLAMA_EXTRACT_REPAIR_RETRIES=1`, and generation limits. Extraction requests use JSON Schema structured output, and there is no automatic local fallback in the Colab flow.
- Fusion uses fixed query-segment weights, evidence adjustment, and weighted RRF.
- Query diagnostics include `embedding_provider`, `embedding_model`, `channel_weights`, `weights_source`, `query_segment`, and channel enablement.

## Artifact Export / Import

직접 호출 환경에서도 인덱스 산출물 manifest를 만들 수 있습니다:

```powershell
.venv\Scripts\python -m aerospace_rag.artifacts.export --index-dir data/index --output-dir data/index/export
.venv\Scripts\python -m aerospace_rag.artifacts.importer --manifest data/index/export/artifact_manifest.json --index-dir data/index
```

Manifest에는 Qdrant, graph-lite, BM25, chunks 파일 목록과 sha256이 들어갑니다.

## Package Layout

- `aerospace_rag.cli`: CLI entry points
- `aerospace_rag.ingestion`: document parsing and chunk creation
- `aerospace_rag.stores`: local index, vector, and graph-lite stores
- `aerospace_rag.retrieval`: BM25, embeddings, extraction, fusion, and channel weights
- `aerospace_rag.generation`: answer generation providers
- `aerospace_rag.artifacts`: portable index artifact import/export
