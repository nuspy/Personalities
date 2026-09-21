"""Verifica che una risposta sia ancorata a ciò che le è stato dato.

Due livelli, e la differenza fra loro non è di precisione ma di natura.

**`citations`** confronta i riferimenti citati con quelli forniti. È un
confronto fra due insiemi: costa zero, non interroga nessun modello, e **non
può sbagliare**. Se la risposta cita `[K9]` e i passaggi arrivavano a `[K5]`,
quel riferimento è inventato — non c'è interpretazione possibile.

**`nli`** interroga un giudice: estrae le affermazioni, le classifica, e per
quelle di fatto verifica se i passaggi le sostengano. Costa una chiamata e
può sbagliare, perché è a sua volta un modello.

**Il giudice non vede mai il prompt della personalità.** Non è una precauzione
di privacy: è che un giudice a cui si dice «stai leggendo Seneca» comincia a
valutare se la risposta *suoni* come Seneca, e finisce per accettare
un'invenzione ben scritta e respingere un fatto detto male. Vede il testo e i
passaggi, e nient'altro.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..knowledge.retriever import PassaggioRecuperato
from ..llm.base import GenerationError, GenerationRequest, Message
from ..observability.tracing import traccia
from .claims import Affermazione, EsitoGroundcheck, Tipo, tipo_da
from .policy import Guardrail

logger = logging.getLogger(__name__)

ISTRUZIONI_GIUDICE = """Sei un verificatore. Ricevi una risposta e i passaggi \
documentari che erano a disposizione di chi l'ha scritta. Il tuo compito è \
stabilire quali affermazioni della risposta siano **di fatto**, e se i \
passaggi le sostengano.

Non sai chi ha scritto la risposta né con quali istruzioni: non è affar tuo, e \
non devi valutare se sia ben scritta, convincente o appropriata.

Classifica ogni affermazione con uno di questi tipi:

- `fatto` — asserisce qualcosa di verificabile consultando una fonte: una \
data, un nome, un evento, ciò che qualcuno ha detto o scritto;
- `esortazione` — consiglio o imperativo rivolto a chi legge;
- `massima` — principio o giudizio di valore;
- `stile` — tono, immagini, metafore, modo di rivolgersi;
- `esperienza` — chi parla racconta qualcosa di sé.

**Solo i `fatto` vanno verificati.** Per gli altri tipi non stabilire se siano \
sostenuti: non sono pretese sul mondo, e una risposta che non contiene alcun \
fatto è corretta così com'è.

Un'affermazione di fatto è sostenuta se i passaggi la contengono o la \
implicano direttamente. Non lo è se va oltre ciò che dicono, per quanto \
plausibile suoni.

Rispondi in JSON con questa forma:

{"affermazioni": [{"testo": "...", "tipo": "fatto", "sostenuta": true, \
"riferimenti": ["K1"], "nota": "..."}]}

`riferimenti` elenca le etichette dei passaggi che la sostengono; vuoto se \
nessuno. `nota` è una riga sul perché, solo quando un fatto non è sostenuto."""


def riferimenti_citati(testo: str) -> set:
    """Le etichette `[Kn]` che compaiono in un testo."""
    import re

    return set(re.findall(r"\[(K\d+)\]", testo))


def controlla_citazioni(
    risposta: str, riferimenti_forniti: Sequence[str]
) -> EsitoGroundcheck:
    """Livello `citations`. Deterministico, e per questo il primo."""
    inventati = sorted(riferimenti_citati(risposta) - set(riferimenti_forniti))
    return EsitoGroundcheck(livello="citations", riferimenti_inventati=inventati)


class Giudice:
    """Il livello `nli`. Una chiamata, un verdetto strutturato."""

    def __init__(self, provider, *, max_tokens: int = 2048) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    async def valuta(
        self,
        risposta: str,
        passaggi: Sequence[PassaggioRecuperato],
        *,
        rubriche: Sequence[Guardrail] = (),
    ) -> EsitoGroundcheck:
        esito = controlla_citazioni(
            risposta, [p.etichetta for p in passaggi],
        )
        esito.livello = "nli"

        if not risposta.strip():
            return esito

        istruzioni = ISTRUZIONI_GIUDICE
        if rubriche:
            # Le rubriche dei guardrail si aggiungono in coda e in ordine di
            # slug: sono parte del prompt del giudice, non di quello del
            # personaggio, e non devono mai attraversare il confine.
            righe = [
                f"\n\n## {g.nome}\n\n{g.verifica.strip()}"
                for g in sorted(rubriche, key=lambda g: g.slug)
            ]
            istruzioni = istruzioni + "\n\n---\n\nCriteri aggiuntivi:" + "".join(righe)

        messaggi = [
            Message(role="system", content=istruzioni),
            Message(role="user", content=_dossier(risposta, passaggi)),
        ]

        try:
            with traccia("groundcheck.nli", passaggi=len(passaggi)):
                dati = await self._provider.complete_json(GenerationRequest(
                    messages=messaggi,
                    # Bassa ma non nulla: a zero alcuni modelli entrano in
                    # cicli ripetitivi su input strutturati.
                    temperature=0.1,
                    max_tokens=self._max_tokens,
                ))
        except (GenerationError, ValueError) as exc:
            # Un giudice che non risponde non rende la risposta fondata: rende
            # il controllo non eseguito, e la differenza va detta. Trattarlo
            # come un via libera significherebbe che ogni guasto del
            # verificatore approva silenziosamente tutto.
            logger.warning("Giudice non disponibile: %s", exc)
            esito.non_eseguito = f"il verificatore non ha risposto: {exc}"
            return esito

        esito.affermazioni = _affermazioni_da(dati)
        return esito


def _dossier(risposta: str, passaggi: Sequence[PassaggioRecuperato]) -> str:
    """Ciò che il giudice vede: i passaggi e la risposta. Nient'altro."""
    righe: List[str] = ["## Passaggi disponibili", ""]

    if passaggi:
        for p in passaggi:
            fonte = p.corrispondenza.documento_titolo
            if p.corrispondenza.sezione:
                fonte = f"{fonte} — {p.corrispondenza.sezione}"
            righe.append(f"[{p.etichetta}] ({fonte})")
            righe.append(p.corrispondenza.testo.strip())
            righe.append("")
    else:
        righe.append("(nessun passaggio fornito)")
        righe.append("")

    righe.extend(["## Risposta da verificare", "", risposta.strip()])
    return "\n".join(righe)


def _affermazioni_da(dati: Any) -> List[Affermazione]:
    """Interpreta il verdetto, tollerando le forme che i modelli producono."""
    if isinstance(dati, list):
        grezze = dati
    elif isinstance(dati, dict):
        grezze = (
            dati.get("affermazioni")
            or dati.get("claims")
            or dati.get("risultati")
            or []
        )
    else:
        return []

    affermazioni: List[Affermazione] = []
    for voce in grezze:
        if not isinstance(voce, dict):
            continue
        testo = str(voce.get("testo") or voce.get("text") or "").strip()
        if not testo:
            continue

        riferimenti = voce.get("riferimenti") or voce.get("references") or []
        if isinstance(riferimenti, str):
            riferimenti = [riferimenti]

        affermazioni.append(Affermazione(
            testo=testo,
            tipo=tipo_da(voce.get("tipo") or voce.get("type")),
            sostenuta=bool(voce.get("sostenuta", voce.get("supported", False))),
            riferimenti=[str(r) for r in riferimenti],
            nota=str(voce.get("nota") or voce.get("note") or ""),
        ))

    return affermazioni
