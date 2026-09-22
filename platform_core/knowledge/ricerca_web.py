"""Ricerca online: portare nel contesto pagine che il corpus non contiene.

Un corpus è chiuso per costruzione, ed è la sua virtù: si sa cosa c'è dentro,
e una risposta si può ancorare a un passaggio che qualcuno ha scelto di
metterci. Il mondo però cambia, e per certe personalità — una che commenta
l'attualità, una che risponde su un prodotto che si aggiorna — quella chiusura
è il difetto.

Da qui la ricerca online, con tre scelte che la rendono accettabile.

**È per personalità, non di piattaforma.** Sta in `rag_config` insieme al
resto del recupero, perché è una proprietà della voce: Seneca non ha bisogno
di sapere cosa è successo stamattina.

**La lista dei siti la applichiamo noi.** Si può chiedere al fornitore di
ricerca di limitarsi a certi domini, e lo fa *di solito*. «Di solito» non è
una garanzia: un risultato fuori lista che entra nel contesto di una voce
professionale è esattamente il caso che la lista doveva impedire. Quindi si
chiede, e poi si filtra: `_ammesso()` è l'unica cosa fra un risultato e il
prompt.

**Ciò che arriva dal web non entra mai nello strato stabile.** Cambia a ogni
richiesta, e lo strato stabile deve restare identico byte per byte o lo sconto
del prompt caching sparisce in silenzio. I passaggi web entrano in coda, con i
loro `[Kn]`, e il groundcheck li tratta come gli altri.

**Due modi.** `solo`: nient'altro che quei siti — è il caso della voce
aziendale, che deve citare la propria documentazione e nient'altro. `anche`:
ricerca aperta, con quei siti privilegiati — è il caso della voce che commenta
il mondo e ha delle fonti che preferisce.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: Lo spazio di nomi con cui si ricavano identificativi stabili dagli URL.
#: Stabili e non casuali: la stessa pagina recuperata due volte porta lo
#: stesso id, e una traccia di ieri si confronta con una di oggi.
SPAZIO_WEB = uuid.UUID("6f9b1e2c-77c4-4a1e-9a2f-2c0b8a4d5e31")

#: Quanti passaggi tenere da una singola pagina. Una pagina lunga non deve
#: occupare il contesto da sola: due passaggi pertinenti valgono più di
#: quindicimila caratteri di cui il modello leggerà l'inizio.
PASSAGGI_PER_PAGINA = 2

#: Il tetto di testo che si accetta da una pagina prima di dividerla. Oltre,
#: si taglia: certe pagine sono interi libri, e l'intero libro non serve.
CARATTERI_MASSIMI = 60_000


@dataclass(frozen=True)
class RisultatoWeb:
    """Una pagina trovata, già ripulita."""

    url: str
    titolo: str
    #: Il testo della pagina in markdown o testo semplice. Vuoto è possibile:
    #: certe pagine non si lasciano estrarre, e una pagina senza testo è da
    #: scartare, non da citare per il titolo.
    testo: str = ""
    #: La data di pubblicazione, quando il fornitore la dà. Conta più che sul
    #: corpus: su una notizia, «quando» è metà dell'informazione.
    pubblicato: str = ""

    @property
    def dominio(self) -> str:
        return dominio_di(self.url)


class RicercaNonDisponibile(RuntimeError):
    """La ricerca online è richiesta ma non c'è un fornitore che la faccia."""


class RicercaWeb(Protocol):
    """Ciò che un fornitore di ricerca deve saper fare."""

    nome: str

    def disponibile(self) -> bool:
        """Se è configurato abbastanza da poter essere interrogato."""
        ...

    async def cerca(
        self,
        domanda: str,
        *,
        siti: Sequence[str] = (),
        solo: bool = False,
        limite: int = 3,
    ) -> List[RisultatoWeb]:
        ...


# --- domini ----------------------------------------------------------------


def dominio_di(url: str) -> str:
    """L'host di un URL, minuscolo e senza `www.`."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def normalizza_sito(voce: str) -> str:
    """Accetta `example.com`, `www.example.com`, `https://example.com/x`.

    Chi compila la lista scrive quello che ha sotto mano — spesso un URL
    completo copiato dalla barra del browser. Rifiutarlo sarebbe pedanteria
    che si paga in liste sbagliate.
    """
    voce = (voce or "").strip()
    if not voce:
        return ""
    if "//" in voce:
        return dominio_di(voce)
    voce = voce.split("/", 1)[0].lower().rstrip(".")
    return voce[4:] if voce.startswith("www.") else voce


def _ammesso(url: str, siti: Sequence[str]) -> bool:
    """Se questo URL sta nella lista. Sottodomini compresi.

    `seneca.it` ammette `testi.seneca.it` ma non `falso-seneca.it`: il
    confronto è sul punto, non sul prefisso — ed è proprio l'errore che
    trasformerebbe una lista in un colabrodo.
    """
    host = dominio_di(url)
    if not host:
        return False
    return any(
        host == sito or host.endswith("." + sito)
        for sito in (normalizza_sito(s) for s in siti)
        if sito
    )


# --- configurazione per personalità ----------------------------------------


@dataclass(frozen=True)
class ConfigurazioneRicerca:
    """Cosa la versione di una personalità chiede alla ricerca online."""

    attiva: bool = False
    siti: List[str] = field(default_factory=list)
    #: `solo` — nient'altro che quei siti; `anche` — ricerca aperta, con
    #: quei siti privilegiati.
    modo: str = "anche"
    max_risultati: int = 3

    @property
    def solo_lista(self) -> bool:
        return self.modo == "solo"

    @staticmethod
    def da_rag(rag_config: Optional[Dict[str, Any]]) -> "ConfigurazioneRicerca":
        grezza = (rag_config or {}).get("ricerca_online") or {}
        if not isinstance(grezza, dict):
            return ConfigurazioneRicerca()

        siti = [
            s for s in (normalizza_sito(v) for v in grezza.get("siti") or [])
            if s
        ]
        modo = grezza.get("modo") if grezza.get("modo") in ("solo", "anche") else "anche"

        return ConfigurazioneRicerca(
            attiva=bool(grezza.get("attiva")),
            siti=siti,
            modo=modo,
            max_risultati=max(1, min(int(grezza.get("max_risultati") or 3), 10)),
        )

    def utilizzabile(self) -> tuple[bool, str]:
        """Se ha senso interrogare, e altrimenti perché no."""
        if not self.attiva:
            return False, ""
        if self.solo_lista and not self.siti:
            # Il caso che sarebbe un difetto silenzioso: «solo questi siti»
            # con la lista vuota significa *nessun* sito, non *tutti*.
            return False, (
                "ricerca online limitata a una lista di siti che è vuota: "
                "nessuna fonte da interrogare"
            )
        return True, ""


# --- fornitori -------------------------------------------------------------


class RicercaDisattivata:
    """Nessun fornitore configurato.

    Non solleva: una personalità con la ricerca accesa su un impianto che non
    ce l'ha deve rispondere lo stesso, col corpus che ha. Lo dice nella
    traccia, che è dove si guarda quando una risposta manca di qualcosa.
    """

    nome = "disattivata"

    def disponibile(self) -> bool:
        return False

    async def cerca(
        self, domanda: str, *, siti: Sequence[str] = (), solo: bool = False,
        limite: int = 3,
    ) -> List[RisultatoWeb]:
        return []


class RicercaFirecrawl:
    """Firecrawl: trova le pagine e le restituisce già ripulite.

    È la ragione per cui è il primo fornitore vero invece di un motore di
    ricerca puro. Un'API di ricerca restituisce *snippet* di due righe: su due
    righe non si ancora niente, e il groundcheck le boccerebbe quasi sempre.
    Qui la pagina arriva in markdown, cioè nella forma in cui i passaggi
    entrano nel contesto.

    Funziona sia con il servizio ospitato sia con un'istanza propria: cambia
    l'indirizzo, e con un'istanza propria le domande degli utenti non escono.
    """

    nome = "firecrawl"

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        *,
        timeout: float = 30.0,
    ) -> None:
        from ..settings import get_settings

        s = get_settings()
        self._base_url = (base_url or s.ricerca_base_url).rstrip("/")
        self._api_key = api_key or s.ricerca_api_key
        self._timeout = timeout

    def disponibile(self) -> bool:
        return bool(self._base_url)

    async def cerca(
        self, domanda: str, *, siti: Sequence[str] = (), solo: bool = False,
        limite: int = 3,
    ) -> List[RisultatoWeb]:
        import httpx

        if not self.disponibile():
            raise RicercaNonDisponibile(
                "nessun indirizzo per il servizio di ricerca "
                "(`PERSONA_RICERCA_BASE_URL`)"
            )

        intestazioni = {"Content-Type": "application/json"}
        if self._api_key:
            intestazioni["Authorization"] = f"Bearer {self._api_key}"

        corpo = {
            "query": _interrogazione(domanda, siti, solo),
            # Più di quanti ne servano: il filtro sui domini ne toglie, e
            # chiederne esattamente tre significherebbe restarne con uno.
            "limit": limite * 3 if solo else limite * 2,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            risposta = await client.post(
                f"{self._base_url}/v1/search", json=corpo, headers=intestazioni,
            )
            risposta.raise_for_status()
            dati = risposta.json()

        trovati: List[RisultatoWeb] = []
        for voce in dati.get("data") or []:
            url = (voce.get("url") or "").strip()
            if not url:
                continue
            trovati.append(RisultatoWeb(
                url=url,
                titolo=(voce.get("title") or url).strip(),
                testo=(voce.get("markdown") or voce.get("description") or "").strip(),
                pubblicato=_data_di(voce),
            ))
        return trovati


def _interrogazione(domanda: str, siti: Sequence[str], solo: bool) -> str:
    """La domanda con gli operatori di sito, quando servono.

    Si chiede al motore di limitarsi, ma non ci si fida: il filtro vero è
    `_ammesso()`, applicato ai risultati. Questo serve a non sprecare le
    risposte, non a garantirle.
    """
    puliti = [s for s in (normalizza_sito(v) for v in siti) if s]
    if not puliti:
        return domanda
    if solo:
        return f"{domanda} ({' OR '.join(f'site:{s}' for s in puliti[:8])})"
    return domanda


def _data_di(voce: Dict[str, Any]) -> str:
    """La data di pubblicazione, dai campi in cui Firecrawl la mette."""
    metadati = voce.get("metadata") or {}
    for chiave in ("publishedTime", "published_time", "article:published_time", "date"):
        valore = metadati.get(chiave) or voce.get(chiave)
        if valore:
            return str(valore)[:32]
    return ""


def fornitore_ricerca() -> RicercaWeb:
    """Il fornitore configurato per questo impianto."""
    from ..settings import get_settings

    scelto = (get_settings().ricerca_provider or "disattivata").strip().lower()
    if scelto in ("", "disattivata", "nessuno"):
        return RicercaDisattivata()
    if scelto == "firecrawl":
        return RicercaFirecrawl()

    logger.warning(
        "Fornitore di ricerca sconosciuto (%r): la ricerca online resta "
        "spenta. Valori ammessi: disattivata, firecrawl", scelto,
    )
    return RicercaDisattivata()


# --- dai risultati ai passaggi ---------------------------------------------


def passaggi_da_risultati(
    risultati: Sequence[RisultatoWeb],
    *,
    domanda: str,
    siti: Sequence[str] = (),
    solo: bool = False,
    limite: int = 3,
) -> List["PassaggioRecuperato"]:
    """Trasforma le pagine trovate in passaggi citabili.

    Due cose accadono qui, e sono quelle che decidono se la ricerca online
    aiuti o rovini una risposta.

    **Il filtro sui domini.** Applicato adesso, sui risultati, qualunque cosa
    il motore abbia capito della richiesta.

    **La scelta del pezzo.** Una pagina è lunga e la domanda riguarda un
    punto: prendere i primi quattromila caratteri significa quasi sempre
    prendere il menu di navigazione e l'introduzione. Si divide la pagina con
    lo stesso chunker del corpus e si tengono i passaggi che condividono più
    parole con la domanda — grossolano, ma è la differenza fra citare il
    paragrafo giusto e citare l'intestazione del sito.
    """
    from .chunker import dividi
    from .retriever import PassaggioRecuperato
    from .vector_store import Corrispondenza

    parole = _parole(domanda)
    scelti: List[PassaggioRecuperato] = []

    for risultato in risultati:
        if len(scelti) >= limite:
            break
        if solo and siti and not _ammesso(risultato.url, siti):
            logger.info(
                "Risultato fuori dalla lista dei siti, scartato: %s",
                risultato.url,
            )
            continue
        if not risultato.testo.strip():
            continue

        documento_id = uuid.uuid5(SPAZIO_WEB, risultato.url)
        pezzi = dividi(risultato.testo[:CARATTERI_MASSIMI]) or []
        migliori = sorted(
            pezzi, key=lambda p: _affinita(p.testo, parole), reverse=True,
        )[:PASSAGGI_PER_PAGINA]

        for pezzo in migliori:
            if len(scelti) >= limite:
                break
            scelti.append(PassaggioRecuperato(
                corrispondenza=Corrispondenza(
                    chunk_id=uuid.uuid5(SPAZIO_WEB, f"{risultato.url}#{pezzo.ordinale}"),
                    testo=pezzo.testo,
                    ordinale=pezzo.ordinale,
                    sezione=pezzo.sezione,
                    documento_id=documento_id,
                    documento_titolo=_titolo(risultato),
                    documento_uri=risultato.url,
                    # Il web non appartiene a nessuna knowledge base: l'id del
                    # documento fa anche da id di raccolta, così la traccia
                    # resta leggibile senza inventare una base fantasma.
                    kb_id=documento_id,
                    punteggio=0.0,
                    # La provenienza è ciò che distingue una fonte che qualcuno
                    # ha scelto di mettere nel corpus da una che ha trovato un
                    # motore di ricerca: vale diversamente, e chi legge la
                    # risposta ha diritto di saperlo.
                    provenienza="web",
                ),
                punteggio_rrf=0.0,
            ))

    return scelti


def _titolo(risultato: RisultatoWeb) -> str:
    """Titolo e dominio insieme, e la data quando c'è.

    Il dominio nel titolo non è ridondanza: nel prompt il modello vede questa
    riga e non l'URL, e «secondo un sito» è diverso da «secondo la
    documentazione ufficiale».
    """
    pezzi = [risultato.titolo or risultato.url]
    if risultato.dominio:
        pezzi.append(f"({risultato.dominio})")
    if risultato.pubblicato:
        pezzi.append(f"— {risultato.pubblicato[:10]}")
    return " ".join(pezzi)


def _parole(testo: str) -> set:
    return {p for p in re.findall(r"\w{4,}", testo.lower())}


def _affinita(testo: str, parole: set) -> int:
    if not parole:
        return 0
    return len(parole & _parole(testo))
