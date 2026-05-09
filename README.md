# Aerospace Colab RAG

항공우주 업무 문서를 Google Colab 또는 로컬 Python 런타임에서 직접 다루기 위한 RAG 구현입니다. 구현의 중심은 서버 운영이 아니라, 업로드된 문서를 한 번의 런타임 안에서 파싱하고, 로컬 인덱스를 만들고, 여러 검색 채널을 결합한 뒤, 근거가 붙은 답변을 생성하는 재현 가능한 프로토타입입니다.

공개 저장소에는 원본 업무 데이터가 포함되지 않습니다. 코드는 임의의 지원 문서를 대상으로 동작하도록 구성되어 있고, 과거 검증용 파일 이름은 strict ingest 모드의 계약으로만 남아 있습니다.

## 구현 목표

이 프로젝트는 [NAMUORI00/smartfarm-workspace](https://github.com/NAMUORI00/smartfarm-workspace)의 Dense/Sparse/Graph 3채널 RAG 아이디어를 항공우주 문서와 Colab T4 데모 환경에 맞게 축소한 포팅판입니다.

유지한 핵심 개념은 dense 검색, sparse/BM25 검색, graph 기반 보강, weighted RRF 결합, 근거 중심 답변, 진단 가능한 실행 흐름입니다. 반대로 Docker compose, 장기 실행 서버, submodule 워크스페이스, FalkorDB 운영 구조는 제거하고 단일 Python 패키지와 파일 기반 인덱스로 단순화했습니다.

구현 범위는 다음으로 제한됩니다.

- 문서 파싱과 청킹
- Qdrant 기반 dense/sparse/image vector index
- JSON graph-lite index
- BM25 keyword index
- self-calibrated fusion profile과 weighted RRF
- Ollama 기반 답변 생성
- Colab 노트북에서의 실행 흐름 오케스트레이션
- 인덱스 산출물 export/import manifest

범위 밖 항목은 장기 운영 서버, 컨테이너 배포, 외부 LLM provider 확장, 기업 내부 데이터 영구 저장입니다.

## 전체 구조

런타임은 `build_index()`와 `ask()` 두 공개 API를 중심으로 움직입니다.

`build_index()`는 `data/` 같은 입력 디렉터리에서 지원 파일을 읽고 `Chunk` 목록으로 정규화합니다. 이후 동일한 chunk 집합을 BM25, Qdrant, graph-lite 저장소에 각각 기록하고, fusion 모드가 활성화되어 있으면 현재 인덱스 내용으로 고정 가중치 프로파일을 자체 보정합니다.

`ask()`는 이미 만들어진 로컬 인덱스를 읽어 질문을 검색합니다. Qdrant dense text, Qdrant sparse, Qdrant dense image, BM25, graph-lite 결과를 모은 뒤 weighted RRF로 합치고, 상위 chunk를 Ollama 또는 명시적 extractive debug provider에 전달해 답변과 근거를 반환합니다.

## 데이터 모델

모든 입력 문서는 `Chunk`로 변환됩니다. `Chunk`는 검색과 답변 생성이 공유하는 최소 단위이며, 다음 정보를 함께 유지합니다.

- `chunk_id`: 파일과 위치에서 파생된 안정적인 chunk 식별자
- `text`: 검색과 답변에 사용할 정규화 텍스트
- `source_file`: 원본 파일명
- `modality`: `text`, `table`, `image`, `qa`, `formula` 등 chunk 성격
- `page`, `sheet`, `row`: 원본 위치 정보
- `metadata`: title, category, keywords, asset/table/image/formula reference 같은 보조 정보

Qdrant payload와 `chunks.jsonl`에는 `canonical_doc_id`, `canonical_chunk_id`, `doc_id`, `tier` 같은 표준 키도 같이 기록됩니다. 검색 경로가 달라도 최종 근거 표시는 같은 chunk payload를 기준으로 맞춰지도록 하기 위한 계약입니다.

## Ingestion

`aerospace_rag.ingestion`은 입력 파일을 발견하고 `DocumentParser`를 통해 chunk를 만듭니다.

지원 파일 형식은 `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.xlsm`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.txt`, `.md`입니다. 생성된 인덱스 산출물이 다시 ingest되지 않도록 `data/index`, notebook checkpoint, `__pycache__` 디렉터리는 건너뜁니다.

파싱 전략은 파일 형식별로 다릅니다.

- PDF는 `pypdf`로 페이지 텍스트를 추출합니다.
- XLSX/XLSM은 `openpyxl`로 행 단위 QA 또는 table chunk를 만듭니다.
- DOCX/PPTX는 Docling을 사용하며, 설치되어 있지 않으면 명확한 오류를 냅니다.
- 이미지 파일은 기본적으로 image chunk가 되며, 검증된 표 이미지에는 코드에 포함된 table text override를 적용합니다.
- TXT/MD는 텍스트를 직접 읽고 token window 단위로 나눕니다.

strict ingest 모드는 과거 데모 데이터셋의 파일명 존재 여부를 확인하는 계약 테스트용 경로입니다. 일반 ingest 모드는 디렉터리 안의 지원 파일을 모두 대상으로 삼습니다.

## Indexing

`LocalIndex.build()`는 하나의 chunk 목록으로 세 가지 검색 저장소를 만듭니다.

- `chunks.jsonl`: 최종 검색 결과를 원본 `Chunk` 객체로 복원하기 위한 normalized chunk log
- `bm25.json`: token 기반 keyword retrieval index
- `qdrant/`: dense text, sparse, dense image vector를 담는 Qdrant local collection
- `graph/graph_index.json`: entity, relation, neighbor, chunk mapping을 담는 graph-lite index

Qdrant가 기본 vector backend입니다. 테스트와 경량 smoke run을 위해 `json` vector backend도 존재하지만, 이는 명시적인 debug 경로입니다. 기본 embedding backend는 `sentence-transformers`의 `BAAI/bge-m3`이며, hash embedding도 명시적인 debug 모드로만 사용됩니다.

## Retrieval

검색은 `LocalIndex.search()`에서 한 번에 조합됩니다. 먼저 `collect_channel_scores()`가 채널별 후보를 수집합니다.

- `vector_dense_text`: 질문 embedding과 chunk text embedding의 dense 유사도
- `vector_sparse`: Qdrant sparse vector와 BM25 keyword score의 결합 채널
- `vector_image`: image basis를 포함한 dense image 검색 채널
- `graph`: entity match와 relation neighbor 확장을 쓰는 graph-lite 채널

이후 채널별 후보 수를 바탕으로 fusion weight를 결정하고 `weighted_rrf()`로 순위를 합칩니다. 최종 `RetrievalHit`에는 통합 점수와 채널별 기여도가 함께 들어갑니다.

runtime profile을 쓰지 못하는 경우에는 query segment별 static weight로 돌아가며, 이때 query token과 chunk token overlap을 이용한 작은 lexical rerank bonus가 적용될 수 있습니다.

## Fusion Profile

인덱스 생성 후 fusion 모드가 `static`, `off`, `disabled`가 아니면 `write_self_calibrated_fusion_profile()`이 실행됩니다.

이 함수는 indexed chunk에서 pseudo query를 만들고, 미리 정의된 weight grid를 여러 후보로 평가합니다. metric은 `mrr@k`이며, 가장 좋은 후보를 다음 파일로 고정 저장합니다.

- `fusion_weights.runtime.json`
- `fusion_profile_meta.runtime.json`

런타임 검색은 이 profile을 동적으로 학습하지 않습니다. 저장된 profile을 읽어 fixed weight로 사용하고, profile이 없거나 잘못되었을 때만 static fallback을 사용합니다. 채널에 evidence가 없거나 너무 적으면 해당 채널 weight를 줄이거나 0으로 만든 뒤 다시 정규화합니다.

## Graph-Lite

`GraphStore`는 외부 graph database를 사용하지 않습니다. `KnowledgeExtractor`가 chunk에서 entity와 relation을 추출하고, 그 결과를 JSON 파일로 저장합니다.

검색 시에는 질문 token이 entity label 또는 canonical id와 겹치는지 확인합니다. 직접 매칭된 entity가 연결된 chunk에 점수를 주고, relation neighbor를 따라 인접 entity chunk에도 confidence 기반 보조 점수를 부여합니다.

이 설계는 운영 graph database의 기능을 대체하려는 것이 아니라, Colab 데모 안에서 graph retrieval의 효과를 재현하기 위한 경량 구현입니다.

## Generation

답변 생성 provider는 의도적으로 좁게 유지됩니다.

- `ollama`: 기본 답변 생성 경로
- `extractive`: LLM 없이 검색 결과만 확인하는 명시적 debug 경로

Ollama 호출이 실패해도 자동으로 extractive 답변으로 우회하지 않습니다. 이 정책은 실패를 숨기지 않고, 실제 생성 모델이 준비되지 않은 상태를 명확히 드러내기 위한 것입니다.

프롬프트는 상위 근거만 사용하도록 구성됩니다. table chunk가 포함되면 같은 행의 값을 비교하고 열 순서를 유지하도록 추가 지시를 넣습니다.

## GPT Pro Cross-Check

선택적으로 OpenAI GPT Pro 모델을 답변 검증 단계에 사용할 수 있습니다. 기본값은 꺼짐이며, 켜면 `ask()`가 Ollama 또는 extractive 답변을 만든 뒤 검색 근거와 답변을 `gpt-5.5-pro`에 보내 `diagnostics.cross_check`에 판정, confidence, unsupported claim, 보완 제안을 남깁니다.

필요한 환경 변수는 다음과 같습니다.

- `GPT_PRO_CROSS_CHECK_ENABLED=1`
- `OPENAI_API_KEY=<OpenAI API key>`
- `GPT_PRO_CROSS_CHECK_MODEL=gpt-5.5-pro` (기본값)
- `GPT_PRO_CROSS_CHECK_REASONING_EFFORT=high` (기본값)
- `GPT_PRO_CROSS_CHECK_TIMEOUT_SECONDS=600` (기본값)

CLI에서는 `--gpt-pro-cross-check` 옵션으로 같은 검증을 켤 수 있습니다. OpenAI API key가 없으면 cross-check는 `skipped` 상태로 기록되고 기존 RAG 답변 생성은 계속됩니다.

## Colab Notebook

`notebooks/aerospace_rag_colab_ui.ipynb`는 패키지 코드를 감싸는 실행 오케스트레이터입니다. 노트북 자체가 별도의 RAG 구현을 갖는 것이 아니라, repository clone, dependency 준비, 데이터 업로드 위치 확인, index build, retrieval check, answer check, 반복 질문 표시를 순서대로 연결합니다.

Colab 흐름의 기본 의도는 엄격합니다. extraction은 Ollama backend를 사용하고, JSON Schema structured output과 repair retry를 적용하며, 자동 local fallback을 켜지 않습니다. 이 설정은 Colab 데모가 실제 LLM extraction 경로를 검증하도록 하기 위한 것입니다.

## Artifact Layer

`aerospace_rag.artifacts`는 생성된 index directory를 다른 런타임으로 옮길 수 있도록 manifest를 만듭니다. manifest는 Qdrant, graph-lite, BM25, chunks 파일 목록과 sha256을 기록합니다.

이 계층은 검색 품질이나 답변 로직에 관여하지 않습니다. 역할은 산출물 묶음의 무결성과 재배치를 다루는 것입니다.

## Diagnostics

debug 응답에는 검색과 생성의 주요 결정을 추적할 수 있는 값이 포함됩니다.

- 사용된 검색 채널
- embedding provider와 model
- fusion channel weights
- weights source (`runtime_profile` 또는 `static`)
- query segment
- channel enablement
- evidence adjustment
- top document channel contribution
- provider routing

이 정보는 Colab 화면에서 “왜 이 근거가 올라왔는지”를 설명하기 위한 관찰 지점입니다.

## 테스트의 의미

`tests/`는 구현 계약을 고정합니다. 주요 테스트는 공개 API import, ingest 동작, notebook runtime contract, vector/graph/BM25 retrieval, provider routing, fusion profile fallback, artifact manifest를 확인합니다.

테스트는 샘플 데이터가 공개 저장소에 없다는 조건도 반영합니다. private dataset이 필요한 pipeline 테스트는 파일 존재 여부에 따라 skip되고, 공개 저장소에서 항상 검증 가능한 경로는 임시 데이터와 debug backend를 사용합니다.

## 패키지 역할

- `aerospace_rag.pipeline`: 공개 API와 build/query orchestration
- `aerospace_rag.models`: chunk, hit, response, build result 데이터 계약
- `aerospace_rag.ingestion`: 파일 발견, 파싱, chunk 생성
- `aerospace_rag.stores`: local index, Qdrant vector store, graph-lite store
- `aerospace_rag.retrieval`: BM25, embedding, extraction, fusion, weight/profile 결정
- `aerospace_rag.generation`: Ollama와 extractive provider
- `aerospace_rag.artifacts`: index artifact export/import manifest
- `aerospace_rag.cli`: 패키지 API를 감싼 얇은 CLI entry point
- `aerospace_rag.notebook_runtime`: Colab notebook이 호출하는 런타임 보조 함수
