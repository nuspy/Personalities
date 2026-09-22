"""Le altre voci chiamate in causa con `@Nome`.

Le promesse:

1. **solo su menzione esplicita** — senza `@`, gli scritti dell'altra voce non
   entrano, anche se la domanda la riguarda;
2. **i passaggi dicono di chi sono** — nel prompt e nelle fonti, o il modello
   si attribuirebbe le parole di un altro;
3. **una menzione non aggira il piano** — categoria non compresa: rifiutata
   col motivo; privata di un altro: inesistente;
4. **lo strato stabile non cambia** — le menzioni stanno sotto il punto di
   cache, o ogni domanda con una chiocciola romperebbe lo sconto.
"""
from __future__ import annotations

import uuid

import pytest_asyncio

from platform_core.domain.base import utcnow
from platform_core.domain.knowledge_models import (
    CommercialCategory, KnowledgeBase, Personality, PersonalityKnowledgeBase,
    PersonalityVersion,
)
from platform_core.knowledge.chunker import ConfigurazioneChunking
from platform_core.knowledge.indexer import Indexer
from platform_core.runtime.menzioni import trova

from .conftest import richiede_database
from .test_chat_personalita import (  # noqa: F401
    ModelloCitante, client, embedder, eventi_di, installa, personalita,
)
from .test_retrieval import TESTO_TEMPO


class TestRiconoscimento:
    def test_nomi_semplici_e_composti(self):
        assert trova("Che ne direbbe @Seneca, e @Marco_Aurelio?") == ["seneca", "marco-aurelio"]

    def test_un_indirizzo_non_e_una_menzione(self):
        assert trova("scrivimi a lucilio@example.com") == []

    def test_la_stessa_voce_due_volte_conta_una(self):
        assert trova("@Seneca e ancora @seneca") == ["seneca"]


async def _altra_voce(session, utente, embedder, *, nome="Il Cronista", proprietario=None,
                      visibilita="pubblica", categoria=None) -> Personality:  # noqa: F811
    kb = KnowledgeBase(slug=f"cronache-{uuid.uuid4().hex[:6]}", name="Cronache",
                       embed_model=embedder.modello, owner_id=(proprietario or utente).id)
    session.add(kb)
    await session.flush()
    await Indexer(session, embedder, config=ConfigurazioneChunking(token_obiettivo=60)).indicizza(
        kb, titolo="Sul tempo", testo=TESTO_TEMPO, lingua="it",
    )
    p = Personality(slug=f"cronista-{uuid.uuid4().hex[:6]}", display_name=nome, status="published",
                    visibility=visibilita, owner_id=(proprietario or utente).id,
                    commercial_category_id=categoria.id if categoria else None)
    session.add(p)
    await session.flush()
    v = PersonalityVersion(personality_id=p.id, version=1, system_prompt="Sei un cronista.",
                           published_at=utcnow())
    session.add(v)
    await session.flush()
    p.current_version_id = v.id
    session.add(PersonalityKnowledgeBase(personality_id=p.id, kb_id=kb.id, role="knowledge"))
    await session.commit()
    return p


def _voci_nelle_fonti(risposta) -> set:
    fonti = next((d for t, d in eventi_di(risposta.text) if t == "sources"), {"passaggi": []})
    return {f.get("voce") for f in fonti["passaggi"]}


@richiede_database
class TestMenzioni:
    async def test_la_voce_chiamata_porta_i_suoi_passaggi(
        self, client, session, personalita, utente, embedder,  # noqa: F811
    ):
        p, _, _ = personalita
        await _altra_voce(session, utente, embedder)
        modello = ModelloCitante()
        installa(client, modello)

        r = await client.post("/chat", json={
            "message": "Che cosa direbbe @Il_Cronista del tempo e della virtù?", "personality": p.slug,
        })

        menzioni = next(d for t, d in eventi_di(r.text) if t == "menzioni")
        assert [m["nome"] for m in menzioni["incluse"]] == ["Il Cronista"]
        assert "Il Cronista" in _voci_nelle_fonti(r)
        prompt = "\n".join(m.content for m in modello.richieste[0].messages)
        assert "chiama in causa Il Cronista" in prompt
        assert "(di Il Cronista:" in prompt, "il passaggio dice di chi è"

    async def test_senza_chiocciola_non_entra(self, client, session, personalita, utente, embedder):  # noqa: F811
        p, _, _ = personalita
        await _altra_voce(session, utente, embedder)
        installa(client, ModelloCitante())

        r = await client.post("/chat", json={
            "message": "Che cosa direbbe il cronista del tempo?", "personality": p.slug,
        })

        assert not any(t == "menzioni" for t, _ in eventi_di(r.text))
        assert _voci_nelle_fonti(r) <= {None}

    async def test_la_privata_di_un_altro_non_esiste(
        self, client, session, personalita, utente, altro_utente, embedder,  # noqa: F811
    ):
        p, _, _ = personalita
        await _altra_voce(session, utente, embedder, proprietario=altro_utente, visibilita="privata")
        installa(client, ModelloCitante())

        r = await client.post("/chat", json={
            "message": "Che cosa direbbe @Il_Cronista del tempo?", "personality": p.slug,
        })

        assert not any(t == "menzioni" for t, _ in eventi_di(r.text)), "nemmeno un rifiuto: non esiste"
        assert "Il Cronista" not in _voci_nelle_fonti(r)

    async def test_una_voce_fuori_dal_piano_e_rifiutata_col_motivo(
        self, client, session, personalita, utente, embedder,  # noqa: F811
    ):
        p, _, _ = personalita
        oro = CommercialCategory(slug=f"oro-{uuid.uuid4().hex[:6]}", name="Oro", rank=2)
        session.add(oro)
        await session.flush()
        await _altra_voce(session, utente, embedder, categoria=oro)
        installa(client, ModelloCitante())

        r = await client.post("/chat", json={
            "message": "Che cosa direbbe @Il_Cronista del tempo?", "personality": p.slug,
        })

        menzioni = next(d for t, d in eventi_di(r.text) if t == "menzioni")
        assert menzioni["incluse"] == []
        assert menzioni["escluse"][0]["motivo"] == "il tuo piano non comprende questa voce"
        assert "Il Cronista" not in _voci_nelle_fonti(r)

    async def test_lo_strato_stabile_non_cambia(self, client, session, personalita, utente, embedder):  # noqa: F811
        p, _, _ = personalita
        await _altra_voce(session, utente, embedder)
        modello = ModelloCitante()
        installa(client, modello)

        await client.post("/chat", json={"message": "Che cos'è la virtù?", "personality": p.slug})
        await client.post("/chat", json={"message": "E per @Il_Cronista?", "personality": p.slug})

        primo, secondo = (r.messages[0].content for r in modello.richieste)
        assert primo == secondo
