"""L'endpoint di conversazione, con un modello finto.

Il modello è sostituito da un generatore controllato: ciò che si prova qui non
è la qualità delle risposte — quella non è verificabile con un'asserzione — ma
che il percorso attorno regga. In particolare i tre casi in cui è facile
sbagliare e difficile accorgersene:

- il turno dell'utente dev'essere salvato **prima** della generazione, o una
  domanda va perduta ogni volta che il modello fallisce;
- un errore del modello dev'essere un evento nel flusso, non una connessione
  che si chiude a metà: il client non distinguerebbe un guasto da una risposta
  finita;
- la risposta si salva quando il flusso è concluso, e quella scrittura avviene
  dopo che la sessione della richiesta è già chiusa.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, List

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from platform_core.api.app import create_app
from platform_core.api.deps import get_llm_provider
from platform_core.auth.dependencies import current_principal
from platform_core.domain.repositories import ConversationRepository, UserRepository
from platform_core.domain.session import get_db_session, get_session_factory
from platform_core.llm.base import GenerationError, StreamChunk, Usage

from .conftest import richiede_database

pytestmark = richiede_database


class ModelloFinto:
    """Emette parole prestabilite, o fallisce a comando."""

    name = "finto"

    def __init__(self, parole: List[str] | None = None, errore: Exception | None = None):
        self.parole = parole or ["Ciao", ", ", "come ", "stai", "?"]
        self.errore = errore
        self.richieste: List = []

    async def stream(self, request) -> AsyncIterator[StreamChunk]:
        self.richieste.append(request)
        if self.errore:
            raise self.errore
        for parola in self.parole:
            yield StreamChunk(text=parola)
        yield StreamChunk(
            done=True,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


@pytest_asyncio.fixture
async def client(session, session_factory, principal_utente, monkeypatch):
    """Applicazione con identità finta e database del test."""
    app = create_app()
    app.dependency_overrides[current_principal] = lambda: principal_utente

    async def sessione_del_test():
        yield session

    app.dependency_overrides[get_db_session] = sessione_del_test
    # Anche la fabbrica: la scrittura finale dello streaming avviene dopo la
    # risposta, con una sessione propria, e deve restare dentro la transazione
    # del test.
    app.dependency_overrides[get_session_factory] = lambda: session_factory

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def eventi_di(testo: str) -> List[tuple]:
    """Scompone una risposta SSE in coppie (evento, dati)."""
    risultato = []
    tipo = None
    for riga in testo.splitlines():
        if riga.startswith("event:"):
            tipo = riga[6:].strip()
        elif riga.startswith("data:"):
            risultato.append((tipo, json.loads(riga[5:])))
    return risultato


def installa_modello(client, modello) -> None:
    """Sostituisce il fornitore sull'applicazione sotto prova.

    Per dipendenza e non per monkeypatch del modulo: e' il meccanismo che
    FastAPI offre apposta, e non lascia residui fra un test e l'altro.
    """
    client._transport.app.dependency_overrides[get_llm_provider] = lambda: modello


class TestAutenticazione:
    async def test_senza_token_non_si_conversa(self, session):
        """Senza sovrascrittura dell'identità: il vero controllo è attivo."""
        app = create_app()

        async def sessione_del_test():
            yield session

        app.dependency_overrides[get_db_session] = sessione_del_test

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            risposta = await c.post("/chat", json={"message": "ciao"})

        assert risposta.status_code == 401

    async def test_senza_token_non_si_elencano_conversazioni(self, session):
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            assert (await c.get("/conversations")).status_code == 401


class TestFlusso:
    async def test_la_risposta_arriva_a_pezzi(self, client, monkeypatch):
        installa_modello(client, ModelloFinto(["Uno", " due", " tre"]))

        risposta = await client.post("/chat", json={"message": "conta"})

        assert risposta.status_code == 200
        assert risposta.headers["content-type"].startswith("text/event-stream")

        eventi = eventi_di(risposta.text)
        tipi = [e[0] for e in eventi]
        assert tipi[0] == "start"
        assert tipi[-1] == "done"
        assert tipi.count("token") == 3

        testo = "".join(d["text"] for t, d in eventi if t == "token")
        assert testo == "Uno due tre"

    async def test_il_primo_evento_porta_l_identificativo(self, client, monkeypatch):
        """Il client deve poter proseguire la conversazione dal primo istante."""
        installa_modello(client, ModelloFinto())

        risposta = await client.post("/chat", json={"message": "ciao"})
        _, dati = eventi_di(risposta.text)[0]

        assert "conversation_id" in dati
        assert "correlation_id" in dati

    async def test_il_consumo_viene_riportato(self, client, monkeypatch):
        installa_modello(client, ModelloFinto())

        risposta = await client.post("/chat", json={"message": "ciao"})
        _, finale = eventi_di(risposta.text)[-1]

        assert finale["usage"]["total_tokens"] == 15

    async def test_il_ragionamento_non_esce(self, client, monkeypatch):
        """Il pensiero del modello si segnala, non si trasmette."""

        class ModelloPensante(ModelloFinto):
            async def stream(self, request):
                yield StreamChunk(reasoning="l'utente vuole sapere se...")
                yield StreamChunk(text="Risposta.")
                yield StreamChunk(done=True)

        installa_modello(client, ModelloPensante())
        risposta = await client.post("/chat", json={"message": "ciao"})

        assert "l'utente vuole sapere" not in risposta.text
        assert ("thinking", {}) in eventi_di(risposta.text)


class TestPersistenza:
    async def test_domanda_e_risposta_restano(
        self, client, session, principal_utente, monkeypatch
    ):
        installa_modello(client, ModelloFinto(["Va", " bene"]))

        risposta = await client.post("/chat", json={"message": "tutto ok?"})
        conversazione_id = eventi_di(risposta.text)[0][1]["conversation_id"]

        import uuid

        utente = await UserRepository(session).ensure(principal_utente)
        messaggi = await ConversationRepository(session).messages(
            utente, uuid.UUID(conversazione_id)
        )

        assert [m.role for m in messaggi] == ["user", "assistant"]
        assert messaggi[0].content == "tutto ok?"
        assert messaggi[1].content == "Va bene"

    async def test_la_domanda_sopravvive_al_guasto_del_modello(
        self, client, session, principal_utente, monkeypatch
    ):
        """Il caso che giustifica il commit prima della generazione.

        Se la domanda si salvasse solo a risposta completata, un modello
        irraggiungibile la farebbe sparire — e l'utente la riscriverebbe senza
        capire perché.
        """
        installa_modello(
            client, ModelloFinto(errore=GenerationError("motore spento"))
        )

        risposta = await client.post("/chat", json={"message": "una domanda"})
        conversazione_id = eventi_di(risposta.text)[0][1]["conversation_id"]

        import uuid

        utente = await UserRepository(session).ensure(principal_utente)
        messaggi = await ConversationRepository(session).messages(
            utente, uuid.UUID(conversazione_id)
        )

        assert len(messaggi) == 1
        assert messaggi[0].content == "una domanda"

    async def test_il_titolo_viene_dalla_prima_domanda(self, client, monkeypatch):
        installa_modello(client, ModelloFinto())
        await client.post("/chat", json={"message": "Parlami di Seneca"})

        elenco = (await client.get("/conversations")).json()
        assert elenco[0]["title"] == "Parlami di Seneca"

    async def test_il_secondo_turno_porta_lo_storico(self, client, monkeypatch):
        modello = ModelloFinto()
        installa_modello(client, modello)

        prima = await client.post("/chat", json={"message": "primo"})
        cid = eventi_di(prima.text)[0][1]["conversation_id"]
        await client.post("/chat", json={"message": "secondo", "conversation_id": cid})

        ultima_richiesta = modello.richieste[-1]
        contenuti = [m.content for m in ultima_richiesta.messages]
        assert "primo" in contenuti, "lo storico non è stato passato al modello"
        assert contenuti[-1] == "secondo"


class TestGuasti:
    async def test_un_errore_del_modello_diventa_un_evento(self, client, monkeypatch):
        """Non una connessione chiusa a metà: quella sembra una fine normale."""
        installa_modello(
            client, ModelloFinto(errore=GenerationError("irraggiungibile"))
        )

        risposta = await client.post("/chat", json={"message": "ciao"})

        assert risposta.status_code == 200
        eventi = dict((t, d) for t, d in eventi_di(risposta.text))
        assert "error" in eventi
        assert eventi["error"]["recoverable"] is True

    async def test_il_messaggio_di_errore_non_rivela_l_interno(
        self, client, monkeypatch
    ):
        installa_modello(
            client,
            ModelloFinto(errore=GenerationError(
                "500 da http://10.0.0.5:1234/v1: modello non caricato"
            )),
        )

        risposta = await client.post("/chat", json={"message": "ciao"})

        assert "10.0.0.5" not in risposta.text
        assert "1234" not in risposta.text


class TestIsolamentoSugliEndpoint:
    async def test_non_si_scrive_in_una_conversazione_altrui(
        self, client, session, altro_utente, monkeypatch
    ):
        installa_modello(client, ModelloFinto())
        sua = await ConversationRepository(session).create(altro_utente)
        await session.flush()

        risposta = await client.post(
            "/chat", json={"message": "intrusione", "conversation_id": str(sua.id)}
        )

        assert risposta.status_code == 404

    async def test_non_si_leggono_i_messaggi_altrui(
        self, client, session, altro_utente
    ):
        repo = ConversationRepository(session)
        sua = await repo.create(altro_utente)
        await repo.add_message(sua, role="user", content="riservato")
        await session.flush()

        risposta = await client.get(f"/conversations/{sua.id}/messages")

        assert risposta.status_code == 404
        assert "riservato" not in risposta.text


class TestValidazione:
    async def test_un_messaggio_vuoto_viene_respinto(self, client):
        assert (await client.post("/chat", json={"message": ""})).status_code == 422

    async def test_una_temperatura_assurda_viene_respinta(self, client):
        risposta = await client.post(
            "/chat", json={"message": "ciao", "temperature": 99}
        )
        assert risposta.status_code == 422

    async def test_un_identificativo_malformato_non_diventa_un_500(self, client):
        risposta = await client.post(
            "/chat", json={"message": "ciao", "conversation_id": "non-un-uuid"}
        )
        assert risposta.status_code == 422
