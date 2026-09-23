"""Ottenere JSON da un modello, nonostante i modelli.

Migrato da `historical_persona_pipeline/pipeline/utils/llm_provider.py`, dove
queste tre difese sono nate da guasti veri. Non esistono né in `llmswitch` né
in `bookwriter`.

**Negoziazione del dialetto.** Il parametro che impone il formato JSON non è
lo stesso ovunque: OpenAI accetta `json_object`, LM Studio recente accetta
solo `json_schema` o `text`, altri non accettano nulla. Si prova in ordine e
si ricorda ciò che ha funzionato. Senza, un server risponde `400` e l'errore
parla di `response_format` — cioè del parametro, non della ragione.

**Estrazione tollerante.** Un modello che ha appena promesso JSON lo incornicia
comunque in ```json … ```, o ci premette una frase di cortesia. Rifiutare quelle
risposte significa buttare via output validi per una questione di contorno.

**Budget adattivo.** I modelli che ragionano spendono la maggior parte dei
token a pensare, e il pensiero non finisce nella risposta: con un tetto basso
il ragionamento consuma tutto e `content` resta vuoto. Non è un errore del
prompt: è un budget insufficiente, e la cura è darne di più. Misurato sul
campo: 4094 token su 4096 spesi a ragionare, contenuto vuoto, diciotto
conversazioni perse in silenzio.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Quante volte alzare il budget prima di arrendersi.
TENTATIVI_DI_BUDGET = 3

#: Il tetto oltre il quale non si sale. Serve a distinguere «serve più spazio»
#: da «questo modello non produrrà mai una risposta»: senza un limite, un
#: modello che ragiona all'infinito farebbe crescere il costo a ogni tentativo.
TETTO_TOKEN = 32_768

#: Da dove parte la scala dei budget per un modello che ragiona.
#:
#: La scala triplica a ogni tentativo, e con un modello che ragiona ogni
#: tentativo è una generazione intera: partire da duemila token significa
#: pagare due giri prima di arrivare dove si sarebbe potuti partire. Misurato
#: sul giudice con Bonsai 2 27B: circa due minuti per tentativo. Si applica
#: solo ai modelli che lo dichiarano (`ragiona` nella loro configurazione),
#: perché su un modello che non ragiona sarebbe solo un tetto più alto e
#: inutile.
BUDGET_RAGIONAMENTO = 6144

_RECINTO = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

#: I dialetti, nell'ordine in cui si provano. `None` significa nessun vincolo
#: formale: il prompt chiede comunque JSON e l'estrazione tollerante fa il
#: resto — è un ripiego, non un fallimento.
DIALETTI = ("json_object", "json_schema", None)


class JsonNonInterpretabile(ValueError):
    """La risposta non conteneva JSON, in nessuna forma riconoscibile."""


def formato_risposta(dialetto: Optional[str]) -> Optional[Dict[str, Any]]:
    """Il payload `response_format` per il dialetto indicato."""
    if dialetto == "json_object":
        return {"type": "json_object"}
    if dialetto == "json_schema":
        # Schema deliberatamente permissivo: serve a ottenere JSON valido, non
        # a imporre una forma. Le risposte attese hanno chiavi diverse a
        # seconda di chi le chiede, e uno schema stretto costringerebbe a
        # duplicarlo qui per ogni chiamante.
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "risposta",
                "strict": False,
                "schema": {"type": "object", "additionalProperties": True},
            },
        }
    return None


def e_rifiuto_del_formato(errore: Exception) -> bool:
    """Vero se il server ha rifiutato proprio il vincolo di formato.

    Distinguerlo conta: un rifiuto del formato si risolve cambiando dialetto,
    qualunque altro errore no — e riprovare a oltranza su un guasto di rete
    moltiplicherebbe soltanto l'attesa.
    """
    testo = str(errore).lower()
    return "response_format" in testo or "json_schema" in testo


def estrai_json(testo: str) -> Any:
    """Interpreta un JSON eventualmente incorniciato o circondato da prosa.

    Origine: `bookwriter/llm/client.py:141-163`.
    """
    testo = testo.strip()

    try:
        return json.loads(testo)
    except json.JSONDecodeError:
        pass

    if (recinto := _RECINTO.search(testo)) is not None:
        try:
            return json.loads(recinto.group(1))
        except json.JSONDecodeError:
            pass

    # Ultimo tentativo: dalla prima parentesi graffa o quadra all'ultima.
    inizio = min(
        (i for i in (testo.find("{"), testo.find("[")) if i != -1), default=-1,
    )
    fine = max(testo.rfind("}"), testo.rfind("]"))
    if inizio != -1 and fine > inizio:
        try:
            return json.loads(testo[inizio : fine + 1])
        except json.JSONDecodeError:
            pass

    raise JsonNonInterpretabile(
        f"nessun JSON riconoscibile nella risposta: {testo[:200]!r}"
    )


#: Origine: `bookwriter/llm/client.py:72-92`. Euristica tarata su casi reali.
_FINESTRA = 400
_SOGLIA_FUGA = 8_000


def sembra_degenerata(coda: str, generati: int) -> bool:
    """Vero se il modello sta ripetendo senza fine invece di procedere.

    Due segnali, tarati per non toccare mai una generazione legittima:

    - alfabeto poverissimo nella finestra osservata (« .  .  .  . »): non è
      prosa in nessuna lingua, e vale subito, perché è il caso che blocca
      davvero — tipicamente gli indici estratti dai PDF;
    - la stessa sequenza ripetuta otto volte, ma **solo** quando il testo
      prodotto ha già passato ogni misura ragionevole. Sotto quella soglia la
      ripetizione è il modo normale in cui questi modelli affinano una frase,
      e interromperli lì rovinerebbe il risultato.
    """
    if len(coda) < _FINESTRA:
        return False
    finestra = coda[-_FINESTRA:]
    if len(set(finestra)) <= 4:
        return True
    if generati < _SOGLIA_FUGA:
        return False
    return coda.count(finestra[-200:]) >= 8


def prossimo_budget(corrente: Optional[int], predefinito: int = 4096) -> int:
    """Il budget del tentativo successivo.

    Triplo e non doppio: raddoppiare costa un tentativo in più prima di
    arrivare dove serve, e ogni tentativo è una generazione intera pagata.
    """
    return min((corrente or predefinito) * 3, TETTO_TOKEN)
