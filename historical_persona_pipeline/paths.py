"""Percorsi canonici del package.

Tutti i percorsi sono derivati dalla posizione di questo file, MAI dalla
working directory: l'applicazione deve funzionare da qualunque cartella.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

CONFIG_DIR = PACKAGE_ROOT / "config"
DATA_DIR = PACKAGE_ROOT / "data"
VALUE_DICT_DIR = DATA_DIR / "value_dictionaries"
SEMANTIC_FIELD_DIR = DATA_DIR / "semantic_fields"
STOPWORD_DIR = DATA_DIR / "stopwords"
PROJECTS_DIR = PACKAGE_ROOT / "projects"
CACHE_DIR = PACKAGE_ROOT / "cache"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"
LANGUAGES_CONFIG_PATH = CONFIG_DIR / "languages.yaml"


def project_dir(project_id: str) -> Path:
    """Cartella di un progetto, creata se assente."""
    d = PROJECTS_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d
