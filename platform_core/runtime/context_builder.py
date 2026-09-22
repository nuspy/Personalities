"""Costruzione del prompt, a strati.

L'ordine non è estetico: è ciò che rende possibile lo sconto sul contesto.

    STRATO 0 — STABILE         prompt della personalità, regole, CAG
    ───────── punto di cache ─────────
    STRATO 1 — memorie e riassunto della sessione
    STRATO 2 — passaggi recuperati, numerati [K1]…[Kn]
    STRATO 3 — la domanda

**Perché il prefisso deve essere byte-identico.** Il prompt caching dei
fornitori — `cache_control` di Anthropic, quello automatico di OpenAI, il
prefix caching di vLLM — riconosce un prefisso già visto solo se coincide
carattere per carattere. Un timestamp nello strato 0, un insieme iterato senza
ordine, un elenco che cambia ordine fra due richieste: ciascuna di queste cose
annulla lo sconto, e **lo annulla in silenzio**. Non c'è errore, non c'è
avviso: solo una bolletta più alta e nessuno che sappia perché.

Per questo lo strato 0 si costruisce da dati ordinati in modo deterministico, e
un test verifica che due richieste alla stessa personalità producano lo stesso
identico testo. Quel test è l'unica difesa contro una regressione che
altrimenti nessuno noterebbe.

**Perché i passaggi stanno sotto il punto di cache** anche se cambiano poco:
cambiano a ogni domanda, e basta che cambi un byte dopo il punto di cache
perché tutto ciò che segue non sia riutilizzabile. Ciò che è stabile in cima,
ciò che varia in fondo — è l'unica regola che conta.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..knowledge.retriever import EsitoRecupero, PassaggioRecuperato
from ..llm.base import Message

logger = logging.getLogger(__name__)

#: Intestazione dei passaggi recuperati. Fissa, perché anche questa fa parte
#: del testo che il modello impara a riconoscere.
INTESTAZIONE_PASSAGGI = "Passaggi dai documenti, con il loro riferimento:"

#: Istruzione su come citare. Sta nello strato 0 — è stabile — e senza di essa
#: il modello cita in modo diverso ogni volta, rendendo impossibile la verifica
#: deterministica delle citazioni.
ISTRUZIONE_CITAZIONI = (
    "Quando un'affermazione viene dai passaggi forniti, indica il riferimento "
    "fra parentesi quadre — per esempio [K1] — subito dopo l'affermazione. "
    "Non inventare riferimenti: usa soltanto quelli presenti nei passaggi. "
    "Se i passaggi non contengono la risposta, dillo apertamente invece di "
    "colmare il vuoto."
)


@dataclass
class StratoStabile:
    """Ciò che non cambia fra una domanda e l'altra.

    Se qualcosa qui dentro cambia da una richiesta all'altra senza che la
    personalità sia cambiata, il difetto è qui.
    """

    prompt_personalita: str
    regole: List[str] = field(default_factory=list)

    #: Le istruzioni dei guardrail: paragrafi, non righe di elenco, e per
    #: questo separate dalle regole. Sono la metà preventiva di una politica —
    #: l'altra metà, la rubrica del giudice, non arriva mai qui: se il
    #: personaggio leggesse con quali criteri sarà verificato, imparerebbe a
    #: soddisfarli invece di comportarsi bene.
    politiche: List[str] = field(default_factory=list)

    #: Documenti piccoli e stabili inclusi per intero (CAG): entrano nel
    #: prefisso e quindi nello sconto, a differenza dei passaggi recuperati.
    documenti_integrali: List[str] = field(default_factory=list)
    istruzione_citazioni: str = ISTRUZIONE_CITAZIONI

    def rendi(self) -> str:
        parti = [self.prompt_personalita.strip()]

        if self.regole:
            # Ordinate: un insieme iterato in ordine diverso fra due richieste
            # romperebbe il prefisso senza cambiare una virgola di contenuto.
            parti.append(
                "Regole di comportamento:\n"
                + "\n".join(f"- {r}" for r in sorted(self.regole))
            )

        if self.politiche:
            # Anche queste ordinate, e per la stessa ragione: l'ordine in cui
            # i file dei guardrail vengono letti dal disco non è garantito.
            parti.append("\n\n".join(sorted(p.strip() for p in self.politiche)))

        if self.documenti_integrali:
            parti.append(
                "Materiale di riferimento:\n\n"
                + "\n\n".join(self.documenti_integrali)
            )

        parti.append(self.istruzione_citazioni)
        return "\n\n".join(p for p in parti if p.strip())


@dataclass
class StratoVolatile:
    """Ciò che cambia a ogni turno."""

    memorie: List[str] = field(default_factory=list)
    riassunto_sessione: str = ""
    passaggi: List[PassaggioRecuperato] = field(default_factory=list)
    #: Le altre voci chiamate in causa con `@Nome`. Qui e non nello strato 0:
    #: cambiano a ogni domanda, e sopra il punto di cache annullerebbero lo
    #: sconto sul prefisso di tutte le altre.
    menzioni: List[str] = field(default_factory=list)

    def rendi(self) -> str:
        parti: List[str] = []

        if self.riassunto_sessione.strip():
            parti.append(f"Fin qui, in questa conversazione:\n{self.riassunto_sessione.strip()}")

        if self.memorie:
            parti.append(
                "Cosa ricordi di chi ti parla:\n"
                + "\n".join(f"- {m}" for m in self.memorie)
            )

        if self.menzioni:
            parti.append(
                "Chi ti scrive chiama in causa "
                + ", ".join(self.menzioni)
                + ". I passaggi segnati «di …» vengono dai suoi scritti, non dai "
                "tuoi: puoi citarli e discuterli come parole sue, mai "
                "attribuirteli."
            )

        if self.passaggi:
            righe = [INTESTAZIONE_PASSAGGI, ""]
            for passaggio in self.passaggi:
                c = passaggio.corrispondenza
                fonte = c.documento_titolo
                if c.sezione:
                    fonte = f"{fonte} — {c.sezione}"
                if passaggio.voce:
                    fonte = f"di {passaggio.voce}: {fonte}"
                righe.append(f"[{passaggio.etichetta}] ({fonte})")
                righe.append(c.testo.strip())
                righe.append("")
            parti.append("\n".join(righe).strip())

        return "\n\n".join(parti)


@dataclass
class ContestoCostruito:
    """Il prompt pronto, e ciò che serve a spiegarlo dopo."""

    messaggi: List[Message]
    #: Indice del messaggio dopo il quale finisce la parte stabile. È ciò che
    #: un fornitore con prompt caching traduce nel proprio meccanismo.
    punto_di_cache: Optional[int]
    testo_stabile: str
    caratteri_stabili: int
    caratteri_volatili: int
    passaggi: List[PassaggioRecuperato] = field(default_factory=list)

    def riferimenti_validi(self) -> set:
        """Le etichette che il modello può legittimamente citare.

        Il groundcheck deterministico della fase 2 confronta con questo
        insieme: una citazione che non vi compare è inventata, e lo si stabilisce
        senza interrogare nessun modello.
        """
        return {p.etichetta for p in self.passaggi}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "punto_di_cache": self.punto_di_cache,
            "caratteri_stabili": self.caratteri_stabili,
            "caratteri_volatili": self.caratteri_volatili,
            "riferimenti": sorted(self.riferimenti_validi()),
        }


class ContextBuilder:
    """Mette insieme gli strati in una conversazione."""

    def costruisci(
        self,
        *,
        stabile: StratoStabile,
        volatile: Optional[StratoVolatile] = None,
        storico: Sequence[Message] = (),
        domanda: str = "",
    ) -> ContestoCostruito:
        volatile = volatile or StratoVolatile()

        testo_stabile = stabile.rendi()
        messaggi: List[Message] = [Message(role="system", content=testo_stabile)]

        # Il punto di cache cade subito dopo il messaggio di sistema: è
        # l'ultimo contenuto identico fra due richieste alla stessa
        # personalità. Lo storico che segue è già diverso al secondo turno.
        punto_di_cache = 0

        messaggi.extend(storico)

        testo_volatile = volatile.rendi()
        if testo_volatile:
            # In un messaggio di sistema a sé e non fuso con il primo: fonderli
            # metterebbe materiale che cambia a ogni turno dentro il prefisso
            # da riusare, annullando lo sconto per intero.
            messaggi.append(Message(role="system", content=testo_volatile))

        if domanda:
            messaggi.append(Message(role="user", content=domanda))

        return ContestoCostruito(
            messaggi=messaggi,
            punto_di_cache=punto_di_cache,
            testo_stabile=testo_stabile,
            caratteri_stabili=len(testo_stabile),
            caratteri_volatili=len(testo_volatile),
            passaggi=list(volatile.passaggi),
        )


def strato_volatile_da_recupero(
    esito: EsitoRecupero,
    *,
    memorie: Optional[Sequence[str]] = None,
    riassunto: str = "",
    menzioni: Optional[Sequence[str]] = None,
) -> StratoVolatile:
    return StratoVolatile(
        memorie=list(memorie or []),
        riassunto_sessione=riassunto,
        passaggi=esito.scelti,
        menzioni=list(menzioni or []),
    )
