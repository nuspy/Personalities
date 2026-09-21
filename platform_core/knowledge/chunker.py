"""Divisione dei testi in passaggi recuperabili.

**Perché non si riusa il chunker della pipeline.** Quello taglia a 4000
caratteri netti, senza sovrapposizione, e per l'analisi stilistica va benissimo:
conta la distribuzione del lessico, non dove cade il confine. Per il recupero
no. Un taglio netto spezza il ragionamento a metà, e il passaggio che contiene
la premessa non contiene la conclusione: la domanda recupera l'uno o l'altro, e
in entrambi i casi il modello legge mezzo pensiero.

Tre scelte, e la ragione di ciascuna:

**Si taglia fra frasi, mai dentro.** Una frase troncata è rumore per
l'embedding — che rappresenta il significato di ciò che legge — e imbarazzante
in una citazione, dove l'utente vede il testo.

**Le frasi si sovrappongono.** L'ultima frase di un passaggio è anche la prima
del successivo, così il legame fra i due non va perduto da nessuna delle due
parti. Il costo è una ripetizione modesta; il beneficio è che una domanda sul
punto di giuntura trova un passaggio che lo contiene per intero.

**Il budget è in token, non in caratteri.** È il modello di embedding a imporre
il limite, e conta i token: misurare in caratteri significa sbagliare di un
fattore che cambia con la lingua — il latino e l'italiano non hanno lo stesso
rapporto fra caratteri e token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

#: Quanti token puntare per passaggio. Sotto il tetto di qualunque modello di
#: embedding ragionevole, e abbastanza per contenere un ragionamento compiuto:
#: passaggi molto brevi recuperano bene ma danno al modello frammenti senza
#: contesto, che è il modo tipico di ottenere risposte ancorate e inutili.
TOKEN_OBIETTIVO = 480

#: Il minimo perché un passaggio valga da solo. Sotto questa soglia si fonde
#: con il precedente: una coda di due righe non risponde a nessuna domanda, ma
#: occupa un posto fra i risultati.
TOKEN_MINIMI = 80

#: Quante frasi ripetere fra un passaggio e il successivo.
FRASI_DI_SOVRAPPOSIZIONE = 1

#: Rapporto fra caratteri e token, misurato su prosa europea. È una stima, e
#: basta: serve a decidere dove tagliare, non a riempire un contesto al byte.
#: Chi vuole il conteggio esatto passa un contatore vero.
CARATTERI_PER_TOKEN = 3.6

#: Fine frase: punto, punto interrogativo o esclamativo, eventuali virgolette
#: o parentesi di chiusura, poi spazio e una maiuscola o una cifra.
#:
#: Le abbreviazioni sono escluse a parte, perché un punto dopo «Dott» non
#: chiude una frase — e un chunker che ci cade produce passaggi di tre parole.
_FINE_FRASE = re.compile(
    r"""(?<=[.!?])["'»”’\)\]]*\s+(?=["'«“\(\[]*[A-ZÀ-ÞÆØ0-9])""",
    re.VERBOSE,
)

_ABBREVIAZIONI = {
    "sig", "dott", "prof", "avv", "ing", "rev", "on", "sen", "mons",
    "ecc", "cfr", "vs", "pag", "pagg", "art", "artt", "cap", "capp",
    "vol", "voll", "fig", "n", "nn", "num", "sec", "secc", "ca",
    "a.c", "d.c", "p.es", "op.cit", "ibid",
}


@dataclass
class Passaggio:
    """Un pezzo di testo pronto per essere indicizzato."""

    testo: str
    ordinale: int
    sezione: Optional[str] = None
    token_stimati: int = 0
    #: Le frasi ripetute dal passaggio precedente. Conoscerle permette di
    #: scontarle quando si misura la copertura di un corpus, altrimenti la
    #: sovrapposizione la gonfierebbe.
    frasi_sovrapposte: int = 0


@dataclass
class ConfigurazioneChunking:
    token_obiettivo: int = TOKEN_OBIETTIVO
    token_minimi: int = TOKEN_MINIMI
    frasi_di_sovrapposizione: int = FRASI_DI_SOVRAPPOSIZIONE
    caratteri_per_token: float = CARATTERI_PER_TOKEN

    def stima_token(self, testo: str) -> int:
        return max(1, round(len(testo) / self.caratteri_per_token))


def dividi_in_frasi(testo: str) -> List[str]:
    """Spezza in frasi, senza cadere sulle abbreviazioni.

    Non è un analizzatore linguistico: è una regola che funziona sulla prosa e
    che, quando sbaglia, sbaglia unendo due frasi invece di spezzarne una a
    metà. È il verso giusto in cui sbagliare, perché un passaggio leggermente
    più lungo è innocuo mentre uno troncato è rumore.
    """
    testo = testo.strip()
    if not testo:
        return []

    grezze = _FINE_FRASE.split(testo)

    # Ricongiunge i tagli caduti dopo un'abbreviazione.
    frasi: List[str] = []
    for pezzo in grezze:
        pezzo = pezzo.strip()
        if not pezzo:
            continue
        if frasi and _finisce_con_abbreviazione(frasi[-1]):
            frasi[-1] = f"{frasi[-1]} {pezzo}"
        else:
            frasi.append(pezzo)
    return frasi


def _finisce_con_abbreviazione(frase: str) -> bool:
    ultima = frase.rstrip().rsplit(" ", 1)[-1].lower().rstrip(".")
    return ultima in _ABBREVIAZIONI


def dividi(
    testo: str,
    *,
    sezione: Optional[str] = None,
    config: Optional[ConfigurazioneChunking] = None,
    ordinale_iniziale: int = 0,
) -> List[Passaggio]:
    """Divide un testo in passaggi sovrapposti."""
    config = config or ConfigurazioneChunking()
    frasi = dividi_in_frasi(testo)
    if not frasi:
        return []

    passaggi: List[Passaggio] = []
    corrente: List[str] = []
    token_correnti = 0
    sovrapposte = 0
    ordinale = ordinale_iniziale

    for frase in frasi:
        token_frase = config.stima_token(frase)

        # Una frase che da sola supera il budget non si spezza: tagliarla
        # produrrebbe proprio il frammento privo di senso che si vuole
        # evitare. Sta in un passaggio suo, più lungo del previsto.
        if token_frase >= config.token_obiettivo and corrente:
            passaggi.append(_chiudi(corrente, ordinale, sezione, config, sovrapposte))
            ordinale += 1
            corrente, token_correnti, sovrapposte = [], 0, 0

        if token_correnti + token_frase > config.token_obiettivo and corrente:
            passaggi.append(_chiudi(corrente, ordinale, sezione, config, sovrapposte))
            ordinale += 1
            coda = corrente[-config.frasi_di_sovrapposizione:] if config.frasi_di_sovrapposizione else []
            corrente = list(coda)
            token_correnti = sum(config.stima_token(f) for f in corrente)
            sovrapposte = len(coda)

        corrente.append(frase)
        token_correnti += token_frase

    if corrente:
        ultimo = _chiudi(corrente, ordinale, sezione, config, sovrapposte)
        # Una coda troppo corta torna nel passaggio precedente invece di
        # restare da sola: occuperebbe un posto fra i risultati senza poter
        # rispondere a nulla.
        if (
            passaggi
            and ultimo.token_stimati < config.token_minimi
            and ultimo.frasi_sovrapposte < len(corrente)
        ):
            nuove = corrente[ultimo.frasi_sovrapposte:]
            fuso = f"{passaggi[-1].testo} {' '.join(nuove)}".strip()
            passaggi[-1].testo = fuso
            passaggi[-1].token_stimati = config.stima_token(fuso)
        else:
            passaggi.append(ultimo)

    return passaggi


def _chiudi(
    frasi: Sequence[str],
    ordinale: int,
    sezione: Optional[str],
    config: ConfigurazioneChunking,
    sovrapposte: int,
) -> Passaggio:
    testo = " ".join(frasi).strip()
    return Passaggio(
        testo=testo,
        ordinale=ordinale,
        sezione=sezione,
        token_stimati=config.stima_token(testo),
        frasi_sovrapposte=sovrapposte,
    )


@dataclass
class Sezione:
    """Una porzione di documento con un titolo proprio."""

    titolo: Optional[str]
    testo: str


_TITOLO_MARKDOWN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def dividi_per_sezioni(testo: str) -> List[Sezione]:
    """Separa un documento nelle sue sezioni, se ne dichiara.

    I titoli non sono decorazione: dicono di cosa parla ciò che segue, e
    portarli nel passaggio aiuta sia il recupero sia chi legge la citazione.
    Un testo senza titoli resta una sezione sola, che è il caso normale.
    """
    titoli = list(_TITOLO_MARKDOWN.finditer(testo))
    if not titoli:
        return [Sezione(titolo=None, testo=testo)]

    sezioni: List[Sezione] = []

    preambolo = testo[: titoli[0].start()].strip()
    if preambolo:
        sezioni.append(Sezione(titolo=None, testo=preambolo))

    for i, titolo in enumerate(titoli):
        inizio = titolo.end()
        fine = titoli[i + 1].start() if i + 1 < len(titoli) else len(testo)
        corpo = testo[inizio:fine].strip()
        if corpo:
            sezioni.append(Sezione(titolo=titolo.group(2), testo=corpo))

    return sezioni


def dividi_documento(
    testo: str, *, config: Optional[ConfigurazioneChunking] = None
) -> List[Passaggio]:
    """Divide un documento intero, sezione per sezione.

    La numerazione è continua su tutto il documento, perché `ordinal` serve a
    ritrovare i vicini di un passaggio: ricominciare da zero a ogni sezione
    renderebbe due passaggi lontani apparentemente adiacenti.
    """
    config = config or ConfigurazioneChunking()
    passaggi: List[Passaggio] = []

    for sezione in dividi_per_sezioni(testo):
        passaggi.extend(
            dividi(
                sezione.testo,
                sezione=sezione.titolo,
                config=config,
                ordinale_iniziale=len(passaggi),
            )
        )
    return passaggi
