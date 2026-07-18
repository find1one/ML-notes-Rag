import importlib.util
from pathlib import Path


def test_config_loads_repository_env_file(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    code_dir = project_root / "code"
    code_dir.mkdir(parents=True)
    (project_root / ".env").write_text(
        "DATABASE_URL=mysql+pymysql://env_user:env_password@db:3306/env_db\n"
        "RAG_CHUNK_SIZE=1000\n"
        "RAG_CHUNK_OVERLAP=125\n",
        encoding="utf-8",
    )

    source = Path(__file__).resolve().parents[1] / "code" / "config.py"
    copied_config = code_dir / "config.py"
    copied_config.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    spec = importlib.util.spec_from_file_location("isolated_config", copied_config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_CONFIG.database_url == "mysql+pymysql://env_user:env_password@db:3306/env_db"
    assert module.DEFAULT_CONFIG.chunk_size == 1000
    assert module.DEFAULT_CONFIG.chunk_overlap == 125
