"""Fonti testuali online senza chiave API.

Tre fonti, scelte perche' interrogabili in modo anonimo e con licenze aperte:

- **Wikipedia** — biografia, epoca, opere: il contesto, non la voce.
- **Wikisource** — i testi originali dell'autore, quando esistono: e' la
  fonte che conta davvero per lo stile.
- **Project Gutenberg** — opere complete in testo integrale, utile per gli
  autori fuori dal dominio di Wikisource.

Regole rispettate: `User-Agent` identificativo (Wikimedia lo richiede),
una richiesta per volta con pausa fra le chiamate, nessun aggiramento di
robots.txt. Le eccezioni di rete non interrompono mai la pipeline: una fonte
irraggiungibile produce un avviso, non un errore fatale.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from ..utils.tls import enable_system_certificates

logger = logging.getLogger(__name__)

# Le reti con proxy TLS rifirmano i certificati: senza questo, ogni richiesta
# fallisce con CERTIFICATE_VERIFY_FAILED anche se il browser funziona.
enable_system_certificates()

# Pausa fra richieste alla stessa fonte. Le API Wikimedia sono gratuite e
# senza autenticazione: un ritmo piu' fitto fa scattare il 429 e, soprattutto,
# scarica su un servizio pubblico un costo che non gli spetta.
REQUEST_DELAY = 1.0
TIMEOUT = 30

# Ritentativi su 429/503, che indicano "rallenta", non "errore".
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF = 5.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class ResearchDocument:
    """Un documento recuperato online."""
    title: str
    text: str
    url: str
    source: str
    language: str = ""
    is_primary_source: bool = False  # testo dell'autore, non su di lui
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class BaseSource:
    name = "base"

    def __init__(self, user_agent: str, max_chars: int = 400_000):
        self.max_chars = max_chars
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request = 0.0
        self._request_delay = REQUEST_DELAY

    def search(self, query: str, language: str, limit: int) -> List[ResearchDocument]:
        raise NotImplementedError

    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """GET con rispetto del rate limit.

        Un 429 significa "stai chiedendo troppo": insistere allo stesso ritmo
        — come faceva la versione precedente — peggiora le cose e porta al
        blocco. Qui si attende, preferibilmente per il tempo indicato dal
        server in `Retry-After`, e si riprova un numero limitato di volte.
        """
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=TIMEOUT, **kwargs)
            except requests.RequestException as exc:
                logger.warning(f"[{self.name}] richiesta fallita: {exc}")
                return None

            if response.status_code == 200:
                return response

            if response.status_code not in RETRYABLE_STATUS:
                logger.warning(
                    f"[{self.name}] risposta {response.status_code} per {url}"
                )
                return None

            if attempt == MAX_RATE_LIMIT_RETRIES:
                logger.warning(
                    f"[{self.name}] ancora {response.status_code} dopo "
                    f"{attempt} tentativi: rinuncio a questa richiesta"
                )
                return None

            wait = self._retry_after(response) or RATE_LIMIT_BACKOFF * attempt
            logger.info(
                f"[{self.name}] limite di frequenza raggiunto "
                f"({response.status_code}): attendo {wait:.0f}s"
            )
            time.sleep(wait)
            # Da qui in poi si procede piu' lentamente per tutta la sessione.
            self._request_delay = min(self._request_delay * 1.5, 10.0)

        return None

    def _throttle(self) -> None:
        """Garantisce la pausa minima fra due richieste consecutive."""
        elapsed = time.monotonic() - self._last_request
        remaining = self._request_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    @staticmethod
    def _retry_after(response: requests.Response) -> Optional[float]:
        """Valore di `Retry-After`, quando il server lo indica."""
        header = response.headers.get("Retry-After")
        if not header:
            return None
        try:
            return min(float(header), 60.0)
        except ValueError:
            return None


class WikipediaSource(BaseSource):
    """Contesto biografico e storico."""

    name = "wikipedia"

    def search(self, query: str, language: str = "it", limit: int = 5) -> List[ResearchDocument]:
        api = f"https://{language}.wikipedia.org/w/api.php"

        response = self._get(api, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json", "srnamespace": 0,
        })
        if response is None:
            return []

        try:
            hits = response.json().get("query", {}).get("search", [])
        except ValueError:
            logger.warning(f"[{self.name}] risposta non JSON per '{query}'")
            return []

        documents: List[ResearchDocument] = []
        for hit in hits:
            title = hit.get("title", "")
            extract = self._fetch_extract(api, title)
            if not extract:
                continue
            documents.append(ResearchDocument(
                title=title,
                text=extract[:self.max_chars],
                url=f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                source=self.name,
                language=language,
                is_primary_source=False,
                metadata={"wordcount": hit.get("wordcount", 0)},
            ))
        return documents

    def _fetch_extract(self, api: str, title: str) -> str:
        response = self._get(api, params={
            "action": "query", "prop": "extracts", "titles": title,
            "explaintext": 1, "format": "json", "redirects": 1,
        })
        if response is None:
            return ""
        try:
            pages = response.json().get("query", {}).get("pages", {})
        except ValueError:
            return ""
        for page in pages.values():
            if "extract" in page:
                return page["extract"]
        return ""


class WikisourceSource(BaseSource):
    """Testi originali: la fonte primaria per lo stile."""

    name = "wikisource"

    # Note editoriali e apparati che non sono voce dell'autore.
    _EDITORIAL = re.compile(
        r"^\s*(nota|note|notes|editor|indice|sommario|contents|bibliograf)",
        re.IGNORECASE,
    )

    def search(self, query: str, language: str = "it", limit: int = 5) -> List[ResearchDocument]:
        api = f"https://{language}.wikisource.org/w/api.php"

        response = self._get(api, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json", "srnamespace": 0,
        })
        if response is None:
            return []

        try:
            hits = response.json().get("query", {}).get("search", [])
        except ValueError:
            return []

        documents: List[ResearchDocument] = []
        for hit in hits:
            title = hit.get("title", "")
            if self._EDITORIAL.match(title):
                continue
            text = self._fetch_text(api, title)
            if len(text.split()) < 100:
                continue
            documents.append(ResearchDocument(
                title=title,
                text=text[:self.max_chars],
                url=f"https://{language}.wikisource.org/wiki/{quote(title.replace(' ', '_'))}",
                source=self.name,
                language=language,
                is_primary_source=True,
            ))
        return documents

    def _fetch_text(self, api: str, title: str) -> str:
        response = self._get(api, params={
            "action": "query", "prop": "extracts", "titles": title,
            "explaintext": 1, "format": "json", "redirects": 1,
        })
        if response is None:
            return ""
        try:
            pages = response.json().get("query", {}).get("pages", {})
        except ValueError:
            return ""
        for page in pages.values():
            if "extract" in page:
                return page["extract"]
        return ""


class GutenbergSource(BaseSource):
    """Opere integrali via l'API pubblica Gutendex."""

    name = "gutenberg"

    API = "https://gutendex.com/books"

    # Codici lingua Gutenberg per le lingue supportate dalla pipeline.
    LANGUAGE_MAP = {
        "it": "it", "en": "en", "fr": "fr", "de": "de", "es": "es",
        "pt": "pt", "ru": "ru", "la": "la", "grc": "el", "el": "el", "hu": "hu",
    }

    def search(self, query: str, language: str = "it", limit: int = 3) -> List[ResearchDocument]:
        params = {"search": query}
        gutenberg_lang = self.LANGUAGE_MAP.get(language)
        if gutenberg_lang:
            params["languages"] = gutenberg_lang

        response = self._get(self.API, params=params)
        if response is None:
            return []

        try:
            results = response.json().get("results", [])
        except ValueError:
            return []

        documents: List[ResearchDocument] = []
        for book in results[:limit]:
            text_url = self._plain_text_url(book.get("formats", {}))
            if not text_url:
                continue

            text_response = self._get(text_url)
            if text_response is None:
                continue

            text = self._strip_boilerplate(text_response.text)
            if len(text.split()) < 500:
                continue

            authors = [a.get("name", "") for a in book.get("authors", [])]
            documents.append(ResearchDocument(
                title=book.get("title", ""),
                text=text[:self.max_chars],
                url=text_url,
                source=self.name,
                language=language,
                is_primary_source=True,
                metadata={"authors": authors, "gutenberg_id": book.get("id")},
            ))
        return documents

    @staticmethod
    def _plain_text_url(formats: Dict[str, str]) -> str:
        # Preferenza al testo semplice UTF-8; si scartano gli archivi .zip.
        for mime in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
            url = formats.get(mime)
            if url and not url.endswith(".zip"):
                return url
        return ""

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Rimuove licenza e intestazioni Gutenberg, che non sono dell'autore."""
        start_marker = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
        end_marker = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)

        start = start_marker.end() if start_marker else 0
        end = end_marker.start() if end_marker else len(text)
        return text[start:end].strip()


SOURCE_REGISTRY = {
    "wikipedia": WikipediaSource,
    "wikisource": WikisourceSource,
    "gutenberg": GutenbergSource,
}
