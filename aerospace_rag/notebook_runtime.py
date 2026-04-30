from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from .ingestion import SUPPORTED_SUFFIXES, iter_supported_files


REQUIRED_NOTEBOOK_PACKAGES = {
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
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
