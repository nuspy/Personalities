"""Il motore: da una domanda a una risposta in carattere e ancorata.

Orchestra recupero, costruzione del prompt e generazione, e registra come la
risposta è nata.

**Sulla scelta del modo.** Una personalità può essere servita in tre modi —
`rag` (prompt e recupero), `lora` (pesi addestrati) o `finetune` — e la
differenza non riguarda solo dove si prende il carattere: riguarda quanto
prompt serve. Reiniettare il profilo stilistico completo sopra pesi già
addestrati raddoppia le istruzioni e produce una caricatura, ed è il rischio 4
del piano. Per questo `strato_persona` si riduce quando i pesi portano già la
voce.

**Sulla degradazione.** Se una personalità punta a una build che nessun worker
può servire — GPU spenta, adapter rimosso — si ricade su `rag` invece di
fallire, **dichiarandolo nella traccia**. Una risposta un po' meno in carattere
è meglio di nessuna risposta; una degradazione silenziosa, invece, è peggio di
entrambe.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from ..domain.knowledge_models import PersonalityVersion
from ..knowledge.retriever import EsitoRecupero, Retriever
from ..knowledge.ricerca_web import ConfigurazioneRicerca, RicercaWeb
from .recupero_assistito import ConfigurazioneRecupero, RecuperoAssistito
from ..llm.base import CacheHint, GenerationRequest, LLMProvider, Message, StreamChunk, Usage
from ..settings import get_settings
from ..observability.tracing import traccia
from .context_strategy import ContextStrategy, scegli_strategia
from .context_builder import (
    ContestoCostruito, ContextBuilder, StratoStabile, StratoVolatile,
    strato_volatile_da_recupero,
)

logger = logging.getLogger(__name__)

#: I modi in cui una personalità può essere servita.
MODI = ("rag", "lora", "finetune")


#: Quanti passaggi porta ciascuna voce chiamata in causa. Pochi: è un
#: interlocutore, non la fonte della risposta, e il contesto è di chi risponde.
PASSAGGI_PER_MENZIONE = 3


@dataclass(frozen=True)
class Menzione:
    """Un'altra voce chiamata in causa con `@Nome`, già risolta e permessa."""

    slug: str
    nome: str
    kb_ids: tuple


@dataclass
class EsitoTurno:
    """Cosa è successo in un turno, oltre al testo."""

    testo: str = ""
    usage: Optional[Usage] = None
    cache: Optional[CacheHint] = None
    recupero: Optional[EsitoRecupero] = None
    contesto: Optional[ContestoCostruito] = None
    modo: str = "rag"
    modo_richiesto: str = "rag"
    motivo_degrado: str = ""
    tempi_ms: Dict[str, int] = field(default_factory=dict)
    #: I nomi delle voci chiamate in causa che hanno portato passaggi.
    menzioni: List[str] = field(default_factory=list)
    #: Cosa ha fatto il recupero assistito: la domanda con cui si è davvero
    #: cercato, quanti passaggi sono stati tenuti, o perché non si è fatto
    #: nulla. Senza, una ricerca che non trova niente sembra colpa del corpus
    #: mentre è la domanda a essere stata riscritta male.
    recupero_assistito: Dict[str, Any] = field(default_factory=dict)
    #: Cosa ha fatto la ricerca online: quanti passaggi ha portato, da quali
    #: domini, o perché non ne ha portati. Vuoto quando non era accesa.
    #: Sta nella traccia perché quando una risposta manca di un fatto recente
    #: la prima domanda è se la ricerca sia avvenuta, e senza questo si
    #: risponde per tentativi.
    ricerca: Dict[str, Any] = field(default_factory=dict)

    @property
    def degradato(self) -> bool:
        return self.modo != self.modo_richiesto

    def traccia_risposta(self) -> Dict[str, Any]:
        """Il contenuto di `answer_traces` per questo turno.

        **Le chiavi sono esattamente i parametri di `TraceRepository`**, e
        tutto ciò che non è una colonna sta dentro `usage`. Non è pedanteria:
        prima il router filtrava tre chiavi e scartava le altre in silenzio,
        e due cose aggiunte qui — cosa aveva fatto la ricerca online, con
        quale domanda si era cercato — non arrivavano mai al database. Una
        traccia che perde pezzi senza dirlo è peggio di una traccia assente,
        perché la si legge credendo che quei pezzi non ci fossero.
        """
        return {
            "retrieved": self.recupero.to_dict() if self.recupero else None,
            "usage": {
                **(self.usage.to_dict() if self.usage else {}),
                "contesto": self.contesto.to_dict() if self.contesto else None,
                # Quale strategia ha servito il prefisso, e con che chiave:
                # insieme ai token letti da cache è ciò che dice se il
                # risparmio è avvenuto e perché no quando non avviene.
                "strategia": (
                    {"nome": self.cache.strategia, "modo": self.cache.modo,
                     "chiave": self.cache.chiave, "slot": self.cache.slot}
                    if self.cache else None
                ),
                "modo": {
                    "usato": self.modo,
                    "richiesto": self.modo_richiesto,
                    "motivo": self.motivo_degrado,
                },
                **({"ricerca": self.ricerca} if self.ricerca else {}),
                **(
                    {"recupero_assistito": self.recupero_assistito}
                    if self.recupero_assistito else {}
                ),
            },
            "latency_ms": self.tempi_ms,
        }


def strato_stabile_da_versione(
    versione: PersonalityVersion,
    *,
    modo: str = "rag",
    politiche: Sequence[str] = (),
) -> StratoStabile:
    """Costruisce lo strato 0 da una versione pubblicata.

    Con `lora` o `finetune` il prompt si accorcia: i pesi portano già lessico e
    ritmo, e ripeterli a parole li spinge oltre il segno — il modello smette di
    scrivere *come* quella persona e comincia a imitarla.
    """
    regole = list((versione.behavior_rules or {}).get("regole", []))

    if modo in ("lora", "finetune"):
        prompt = (versione.behavior_rules or {}).get("prompt_ridotto") or ""
        if not prompt:
            # Nessuno strato ridotto preparato: si usa la prima frase del
            # prompt completo, che di norma è l'identità, senza le istruzioni
            # di stile che i pesi già contengono.
            prompt = versione.system_prompt.split("\n\n", 1)[0]
    else:
        prompt = versione.system_prompt

    return StratoStabile(
        prompt_personalita=prompt,
        regole=regole,
        politiche=list(politiche),
        documenti_integrali=list((versione.rag_config or {}).get("documenti_integrali", [])),
    )


class PersonaEngine:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        retriever: Optional[Retriever] = None,
        builder: Optional[ContextBuilder] = None,
        strategia: Optional[ContextStrategy] = None,
        ricerca: Optional["RicercaWeb"] = None,
        assistente: Optional[RecuperoAssistito] = None,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        #: Chi cerca sul web quando una personalità lo chiede. Iniettabile
        #: perché altrimenti il ramo sarebbe verificabile solo con una rete e
        #: un servizio veri.
        self._ricerca = ricerca
        #: Chi riscrive la domanda e sceglie i passaggi, quando la voce lo
        #: chiede. Usa il modello del compito «recupero», che di norma non è
        #: quello che risponde.
        self._assistente = assistente
        self._builder = builder or ContextBuilder()
        #: Come il prefisso stabile viene fatto riconoscere al motore. Scelta
        #: dal fornitore se non indicata: è lui a sapere se dietro c'è una
        #: KV-cache locale, un prompt caching cloud o niente.
        self.strategia = strategia or scegli_strategia(
            provider, slot=get_settings().llm_kv_slots,
        )

    async def prepara(
        self,
        *,
        versione: PersonalityVersion,
        domanda: str,
        kb_ids: Sequence[uuid.UUID] = (),
        storico: Sequence[Message] = (),
        memorie: Sequence[str] = (),
        riassunto: str = "",
        modo: str = "rag",
        modi_disponibili: Sequence[str] = ("rag",),
        politiche: Sequence[str] = (),
        menzioni: Sequence["Menzione"] = (),
    ) -> EsitoTurno:
        """Recupera e costruisce il prompt, senza ancora generare.

        Separato dalla generazione perché è la parte ispezionabile: un test —
        e la console di amministrazione — possono guardare il prompt esatto che
        sarebbe stato mandato, senza spendere una chiamata al modello.
        """
        esito = EsitoTurno(modo_richiesto=modo, modo=modo)

        if modo not in modi_disponibili:
            esito.modo = "rag"
            esito.motivo_degrado = (
                f"modo '{modo}' non servibile ora (disponibili: "
                f"{', '.join(modi_disponibili)}): risposta in RAG"
            )
            logger.info("Degradazione del modo: %s", esito.motivo_degrado)

        rag_config = versione.rag_config or {}
        limite = int(rag_config.get("max_chunks", 6))
        assistenza = ConfigurazioneRecupero.da_rag(rag_config)

        # La domanda con cui si cerca può non essere quella scritta: in una
        # conversazione «e lui cosa ne pensava?» non contiene il soggetto, e
        # cercare quelle parole nel corpus non trova niente.
        da_cercare = domanda
        if assistenza.riscrivi_domanda and self._assistente is not None:
            inizio_riscrittura = time.perf_counter()
            da_cercare, motivo = await self._assistente.domanda_per_recupero(
                domanda, storico=storico,
            )
            esito.tempi_ms["riscrittura"] = int(
                (time.perf_counter() - inizio_riscrittura) * 1000
            )
            esito.recupero_assistito["domanda_cercata"] = da_cercare
            if motivo:
                esito.recupero_assistito["riscrittura"] = motivo

        inizio = time.perf_counter()
        if self._retriever is not None and kb_ids and esito.modo == "rag":
            esito.recupero = await self._retriever.cerca(
                da_cercare, kb_ids, limite=limite,
            )

        # Le voci chiamate in causa: pochi passaggi ciascuna, numerati dopo
        # quelli propri. Dopo e non mescolati: la voce che risponde resta la
        # fonte principale, e l'altra entra solo dove la si è chiamata.
        chiamate = []
        if self._retriever is not None:
            for menzione in menzioni:
                if not menzione.kb_ids:
                    continue
                trovati = await self._retriever.cerca(
                    da_cercare, menzione.kb_ids, limite=PASSAGGI_PER_MENZIONE,
                )
                if not trovati.scelti:
                    continue
                if esito.recupero is None:
                    esito.recupero = EsitoRecupero(domanda=domanda)
                for passaggio in trovati.scelti:
                    passaggio.voce = menzione.nome
                    passaggio.etichetta = f"K{len(esito.recupero.scelti) + 1}"
                    esito.recupero.scelti.append(passaggio)
                esito.recupero.kb_interrogate.extend(str(k) for k in menzione.kb_ids)
                chiamate.append(menzione.nome)
        esito.menzioni = chiamate
        esito.tempi_ms["recupero"] = int((time.perf_counter() - inizio) * 1000)

        # Il web per ultimo, e in coda ai passaggi del corpus: è materiale che
        # nessuno ha scelto di mettere nella voce, e l'ordine nel prompt è già
        # un giudizio su quanto pesi.
        if esito.modo == "rag":
            await self._cerca_online(esito, versione, da_cercare)

        # La scelta viene per ultima, sul mucchio intero — corpus, voci
        # chiamate in causa e web insieme: scegliere prima del web
        # significherebbe giudicare metà del materiale.
        if (
            assistenza.seleziona_passaggi
            and self._assistente is not None
            and esito.recupero is not None
        ):
            inizio_scelta = time.perf_counter()
            tenuti, scartati, motivo = await self._assistente.scegli_passaggi(
                da_cercare, esito.recupero.scelti,
            )
            esito.tempi_ms["selezione"] = int(
                (time.perf_counter() - inizio_scelta) * 1000
            )
            esito.recupero.scelti = tenuti
            # Gli scartati restano leggibili: quando una risposta manca di un
            # fatto che il corpus contiene, «trovato e poi scartato» è una
            # risposta diversa da «non trovato», e senza questo non si
            # distinguono.
            esito.recupero.scartati.extend(scartati)
            esito.recupero_assistito["tenuti"] = [p.etichetta for p in tenuti]
            esito.recupero_assistito["scartati_dalla_scelta"] = [
                p.etichetta for p in scartati
            ]
            if motivo:
                esito.recupero_assistito["selezione"] = motivo

        volatile = (
            strato_volatile_da_recupero(
                esito.recupero, memorie=memorie, riassunto=riassunto,
                menzioni=chiamate,
            )
            if esito.recupero
            else StratoVolatile(memorie=list(memorie), riassunto_sessione=riassunto)
        )

        esito.contesto = self._builder.costruisci(
            stabile=strato_stabile_da_versione(
                versione, modo=esito.modo, politiche=politiche,
            ),
            volatile=volatile,
            storico=storico,
            domanda=domanda,
        )
        return esito

    async def _cerca_online(
        self, esito: EsitoTurno, versione: PersonalityVersion, domanda: str,
    ) -> None:
        """Aggiunge al recupero i passaggi trovati sul web, se la voce lo vuole.

        **Non fa mai fallire il turno.** Un servizio di ricerca lento o caduto
        è un contrattempo esterno: la risposta esce con il corpus che c'è, e
        la traccia dice cosa è mancato. L'alternativa — un errore in faccia a
        chi ha fatto una domanda — trasformerebbe una funzione accessoria nel
        punto più fragile della piattaforma.
        """
        configurazione = ConfigurazioneRicerca.da_rag(versione.rag_config)
        utilizzabile, motivo = configurazione.utilizzabile()

        if not configurazione.attiva:
            return
        if not utilizzabile:
            esito.ricerca = {"eseguita": False, "motivo": motivo}
            logger.info("Ricerca online non eseguita: %s", motivo)
            return

        fornitore = self._ricerca
        if fornitore is None:
            from ..knowledge.ricerca_web import fornitore_ricerca

            fornitore = fornitore_ricerca()

        if not fornitore.disponibile():
            esito.ricerca = {
                "eseguita": False,
                "motivo": (
                    "la personalità chiede la ricerca online ma questo "
                    "impianto non ha un fornitore configurato "
                    "(`PERSONA_RICERCA_PROVIDER`)"
                ),
            }
            logger.warning("%s", esito.ricerca["motivo"])
            return

        from ..knowledge.ricerca_web import passaggi_da_risultati

        inizio = time.perf_counter()
        try:
            risultati = await fornitore.cerca(
                domanda,
                siti=configurazione.siti,
                solo=configurazione.solo_lista,
                limite=configurazione.max_risultati,
            )
        except Exception as exc:  # noqa: BLE001
            esito.ricerca = {"eseguita": False, "motivo": f"ricerca fallita: {exc}"}
            logger.warning("Ricerca online fallita", exc_info=True)
            return

        passaggi = passaggi_da_risultati(
            risultati,
            domanda=domanda,
            siti=configurazione.siti,
            solo=configurazione.solo_lista,
            limite=configurazione.max_risultati,
        )

        if esito.recupero is None:
            esito.recupero = EsitoRecupero(domanda=domanda)
        for passaggio in passaggi:
            passaggio.etichetta = f"K{len(esito.recupero.scelti) + 1}"
            esito.recupero.scelti.append(passaggio)

        esito.tempi_ms["ricerca_online"] = int((time.perf_counter() - inizio) * 1000)
        esito.ricerca = {
            "eseguita": True,
            "fornitore": fornitore.nome,
            "modo": configurazione.modo,
            "siti": list(configurazione.siti),
            "trovati": len(risultati),
            "usati": len(passaggi),
            # I domini che sono davvero entrati nella risposta: è la
            # verifica che la lista abbia funzionato, e si legge senza
            # aprire i passaggi uno per uno.
            "domini": sorted({
                p.corrispondenza.documento_uri.split("/")[2]
                for p in passaggi
                if p.corrispondenza.documento_uri
                and len(p.corrispondenza.documento_uri.split("/")) > 2
            }),
        }

    async def rispondi_in_streaming(
        self, turno: EsitoTurno, *, versione: PersonalityVersion,
    ) -> AsyncIterator[StreamChunk]:
        """Genera la risposta, aggiornando `turno` mentre procede."""
        if turno.contesto is None:
            raise ValueError("chiamare prima `prepara()`")

        llm_config = versione.llm_config or {}
        richiesta = GenerationRequest(
            messages=turno.contesto.messaggi,
            model=llm_config.get("model", ""),
            temperature=float(llm_config.get("temperature", 0.7)),
            max_tokens=llm_config.get("max_tokens"),
            cache_breakpoint_after=turno.contesto.punto_di_cache,
        )
        richiesta = self.strategia.applica(
            richiesta, testo_stabile=turno.contesto.testo_stabile,
        )
        turno.cache = richiesta.cache

        pezzi: List[str] = []
        inizio = time.perf_counter()
        primo_token_a: Optional[float] = None

        with traccia(
            "generazione",
            modo=turno.modo,
            passaggi=len(turno.contesto.passaggi),
        ):
            async for chunk in self._provider.stream(richiesta):
                if chunk.text and primo_token_a is None:
                    primo_token_a = time.perf_counter()
                if chunk.text:
                    pezzi.append(chunk.text)
                if chunk.done:
                    turno.usage = chunk.usage
                yield chunk

        turno.testo = "".join(pezzi)
        turno.tempi_ms["generazione"] = int((time.perf_counter() - inizio) * 1000)
        if primo_token_a is not None:
            turno.tempi_ms["primo_token"] = int((primo_token_a - inizio) * 1000)


def riferimenti_citati(testo: str) -> set:
    """Le etichette `[Kn]` che compaiono in una risposta.

    Base del groundcheck deterministico: confrontate con i riferimenti
    effettivamente forniti, dicono se il modello ha citato qualcosa che non
    esisteva. Costa zero e non richiede un secondo modello — per questo è il
    primo dei due livelli della fase 2.
    """
    import re

    return set(re.findall(r"\[(K\d+)\]", testo))
