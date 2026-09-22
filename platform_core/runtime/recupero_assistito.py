"""Il modello che aiuta il recupero, prima e dopo.

Due punti in cui una chiamata in più cambia davvero la qualità di una
risposta, e sono diversi fra loro.

**Prima: la domanda con cui si interroga il corpus.** In una conversazione le
domande si appoggiano a quelle di prima — «e lui cosa ne pensava?», «e dopo?»
— e cercare quelle parole nel corpus non trova niente, perché il soggetto sta
tre turni indietro. La riscrittura lo rimette dentro. È il caso in cui il
recupero fallisce in modo silenzioso: nessun errore, semplicemente passaggi
che non c'entrano, e una risposta vaga che sembra colpa del modello.

**Dopo: quali passaggi tenere.** Il recupero ne porta sei perché sei è un
numero ragionevole, non perché sei siano pertinenti. Quelli che non c'entrano
non sono neutri: occupano contesto, e danno al modello materiale per rispondere
di fianco alla domanda.

**Si sceglie, non si riscrive.** La tentazione è far condensare i passaggi al
modello. Non si fa, e la ragione è precisa: il testo dei passaggi è ciò che
l'utente vede nelle citazioni ed è ciò contro cui il giudice verifica le
affermazioni. Un passaggio riscritto da un modello è una fonte che nessuno ha
scritto — e una citazione che non corrisponde al documento è peggio di un
passaggio inutile. Qui si tengono i testi parola per parola e si decide solo
quali entrano.

**Costa una chiamata per volta, e per questo è spento di default.** Si accende
per personalità: una voce che risponde di fatti su un corpus grande ne
guadagna molto, una che conversa e basta pagherebbe latenza per nulla.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..llm.base import GenerationRequest, Message

logger = logging.getLogger(__name__)

#: Il tetto di token per la riscrittura e per la selezione.
#:
#: Largo, e non perché servano risposte lunghe — una interrogazione sta in
#: quindici parole e una selezione in cinque etichette. È che **un modello che
#: ragiona spende il budget prima di cominciare a scrivere**: con un tetto da
#: centoventi token esaurisce il ragionamento e restituisce testo vuoto, e la
#: funzione ricade sull'originale a ogni chiamata senza che nulla sembri
#: rotto. Misurato su Bonsai 2 27B, che ragiona: falliva sempre.
#:
#: Su un modello che non ragiona non costa niente: si ferma da sé quando ha
#: finito, e il tetto resta un tetto. Le riscritture che divagano le scarta
#: comunque il controllo sulla lunghezza, più sotto.
TOKEN_DOMANDA = 900
TOKEN_SELEZIONE = 900

#: Quanto di ciascun passaggio si mostra al selezionatore. Non serve tutto:
#: per decidere se un passaggio c'entra bastano le prime righe, e mandarne
#: quattromila caratteri per sei passaggi costerebbe più della risposta.
ANTEPRIMA = 600

ISTRUZIONI_DOMANDA = """Riscrivi la domanda dell'utente in una forma adatta a \
cercare in un archivio di documenti.

Regole:
- rimetti dentro ciò che la domanda dà per sottinteso perché già detto nei \
turni precedenti: nomi propri, argomento, periodo;
- conserva le parole dell'utente dove sono già buone per cercare: non è una \
parafrasi elegante, è una interrogazione;
- niente domande multiple, niente spiegazioni, niente virgolette;
- se la domanda è già autosufficiente, restituiscila identica.

Rispondi con la sola interrogazione, su una riga."""

ISTRUZIONI_SELEZIONE = """Ricevi una domanda e alcuni passaggi numerati.

Indica quali passaggi servono davvero a rispondere, dal più utile al meno \
utile. Scarta quelli che parlano d'altro, per quanto interessanti.

Non riassumere e non riscrivere nulla: rispondi con le sole etichette, \
separate da virgole, per esempio: K3, K1, K5

Se nessun passaggio c'entra, rispondi: nessuno"""


@dataclass(frozen=True)
class ConfigurazioneRecupero:
    """Cosa la versione di una personalità chiede al recupero assistito."""

    #: Riscrivere la domanda prima di cercare.
    riscrivi_domanda: bool = False
    #: Tenere solo i passaggi pertinenti fra quelli recuperati.
    seleziona_passaggi: bool = False

    @property
    def attivo(self) -> bool:
        return self.riscrivi_domanda or self.seleziona_passaggi

    @staticmethod
    def da_rag(rag_config: Optional[Dict[str, Any]]) -> "ConfigurazioneRecupero":
        grezza = (rag_config or {}).get("recupero_assistito") or {}
        if not isinstance(grezza, dict):
            return ConfigurazioneRecupero()
        return ConfigurazioneRecupero(
            riscrivi_domanda=bool(grezza.get("riscrivi_domanda")),
            seleziona_passaggi=bool(grezza.get("seleziona_passaggi")),
        )


class RecuperoAssistito:
    """Chiede al modello del compito «recupero» di aiutare le due fasi.

    **Nessuno dei due metodi solleva.** Un aiuto che fa fallire la richiesta
    che doveva aiutare è peggio di nessun aiuto: si torna alla domanda
    originale, ai passaggi originali, e si scrive perché nella traccia.
    """

    def __init__(self, provider) -> None:
        self._provider = provider

    async def domanda_per_recupero(
        self, domanda: str, *, storico: Sequence[Message] = (),
    ) -> tuple[str, str]:
        """La domanda da cercare, e il motivo se è rimasta quella originale."""
        if not domanda.strip():
            return domanda, "domanda vuota"

        contesto = "\n".join(
            f"{m.role}: {m.content[:400]}" for m in list(storico)[-6:]
        )
        richiesta = GenerationRequest(
            messages=[
                Message(role="system", content=ISTRUZIONI_DOMANDA),
                Message(
                    role="user",
                    content=(
                        (f"Turni precedenti:\n{contesto}\n\n" if contesto else "")
                        + f"Domanda: {domanda}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=TOKEN_DOMANDA,
        )

        try:
            riscritta = (await self._provider.complete(richiesta)).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Riscrittura della domanda fallita: %s", exc)
            return domanda, f"riscrittura fallita: {exc}"

        # Prima la riga, poi le virgolette: al contrario, una risposta
        # virgolettata seguita da una spiegazione perderebbe solo la virgoletta
        # d'apertura e si porterebbe dietro quella di chiusura.
        prima_riga = riscritta.splitlines()[0] if riscritta else ""
        riscritta = prima_riga.strip().strip('"«»“”').strip()
        if not riscritta:
            return domanda, "il modello non ha restituito una domanda"
        # Una riscrittura lunga il triplo dell'originale non è una
        # interrogazione: è il modello che ha risposto invece di riscrivere.
        if len(riscritta) > max(200, len(domanda) * 3):
            logger.info("Riscrittura scartata perché troppo lunga")
            return domanda, "riscrittura troppo lunga: scartata"
        return riscritta, ""

    async def scegli_passaggi(
        self, domanda: str, passaggi: Sequence[Any],
    ) -> tuple[List[Any], List[Any], str]:
        """Divide i passaggi in tenuti e scartati, nell'ordine deciso.

        Restituisce `(tenuti, scartati, motivo)`. Con un motivo valorizzato i
        tenuti sono tutti quelli di partenza: si è preferito non decidere
        piuttosto che decidere a caso.
        """
        if len(passaggi) < 2:
            # Con zero o un passaggio non c'è niente da scegliere, e una
            # chiamata al modello per confermarlo è una chiamata sprecata.
            return list(passaggi), [], ""

        elenco = "\n\n".join(
            f"[{p.etichetta}] {p.corrispondenza.testo[:ANTEPRIMA]}"
            for p in passaggi
        )
        richiesta = GenerationRequest(
            messages=[
                Message(role="system", content=ISTRUZIONI_SELEZIONE),
                Message(role="user", content=f"Domanda: {domanda}\n\n{elenco}"),
            ],
            temperature=0.0,
            max_tokens=TOKEN_SELEZIONE,
        )

        try:
            risposta = (await self._provider.complete(richiesta)).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Selezione dei passaggi fallita: %s", exc)
            return list(passaggi), [], f"selezione fallita: {exc}"

        etichette = _etichette(risposta)
        per_etichetta = {p.etichetta: p for p in passaggi}
        tenuti = [
            per_etichetta[e] for e in etichette if e in per_etichetta
        ]

        if not tenuti:
            # «Nessuno c'entra» è una risposta legittima del selezionatore, ma
            # togliere ogni passaggio lascerebbe il modello senza niente a cui
            # ancorarsi — e una risposta senza fonti è esattamente ciò che il
            # recupero doveva impedire. Si tiene tutto e lo si dichiara.
            return list(passaggi), [], (
                "il selezionatore non ha indicato passaggi utilizzabili: "
                "tenuti tutti"
            )

        scelti = {p.etichetta for p in tenuti}
        scartati = [p for p in passaggi if p.etichetta not in scelti]
        return tenuti, scartati, ""


def _etichette(risposta: str) -> List[str]:
    """Le etichette `Kn` citate, nell'ordine in cui compaiono, senza ripetizioni."""
    import re

    viste: List[str] = []
    for trovata in re.findall(r"\bK\d+\b", risposta.upper()):
        if trovata not in viste:
            viste.append(trovata)
    return viste
