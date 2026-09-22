"""Voce e avatar, dall'API.

Due promesse che vanno verificate insieme perché è insieme che si rompono:

- **il server non manda mai fotogrammi**: audio e tempi, e il client disegna;
- **i tempi o sono misurati o non ci sono**. Un labiale animato su una stima
  si vede fuori sincrono, ed è peggio di una bocca ferma perché sembra un
  difetto del modello 3D invece che una funzione mancante.
"""
from __future__ import annotations

import base64
import uuid
from typing import Any, Dict, List, Sequence

import pytest_asyncio

from platform_core.api.deps import get_allineatore, get_tts
from platform_core.domain.avatar_models import Avatar
from platform_core.voice.base import Sintesi, SintesiNonDisponibile
from platform_core.voice.visemi import VISEMI, Parola

from .conftest import richiede_database
from .test_billing_api import piani  # noqa: F401
from .test_chat_personalita import client, embedder, personalita  # noqa: F401

pytestmark = richiede_database


class VoceFinta:
    """Restituisce audio e basta, come quasi tutti i fornitori veri."""

    name = "finta"

    def __init__(self, *, esplode: str = "") -> None:
        self.chiamate: List[Dict[str, Any]] = []
        self._esplode = esplode

    async def voci(self) -> Sequence[Dict[str, Any]]:
        return [{"id": "grave", "lingua": "it"}]

    async def sintetizza(self, testo, *, voce="", lingua="it") -> Sintesi:
        if self._esplode:
            raise SintesiNonDisponibile(self._esplode)
        self.chiamate.append({"testo": testo, "voce": voce, "lingua": lingua})
        return Sintesi(audio=b"suono", media_type="audio/mpeg", voce=voce)


class AllineatoreFinto:
    """Misura tempi plausibili senza caricare un modello da mezzo giga."""

    def __init__(self, *, riesce: bool = True) -> None:
        self._riesce = riesce

    def allinea(self, audio, testo, *, lingua="it") -> List[Parola]:
        if not self._riesce:
            return []
        parole = testo.split()
        return [
            Parola(p, i * 0.4, i * 0.4 + 0.35) for i, p in enumerate(parole)
        ]


def installa_voce(client, tts=None, allineatore=None) -> None:
    app = client._transport.app
    if tts is not None:
        app.dependency_overrides[get_tts] = lambda: tts
    app.dependency_overrides[get_allineatore] = lambda: allineatore


@pytest_asyncio.fixture
async def con_voce(session, utente, piani):  # noqa: F811
    """Un abbonamento che comprende la voce."""
    from platform_core.billing.plans import GestoreAbbonamenti

    gestore = GestoreAbbonamenti(session)
    piano = await gestore.piano_per_slug("gold")
    await gestore.sottoscrivi(utente.id, piano)
    await session.flush()


class TestDirittoAllaVoce:
    async def test_senza_il_diritto_e_403_col_motivo(
        self, client, piani,  # noqa: F811
    ):
        """`Diritti.voce` esisteva da prima ed era l'unico a non essere mai
        consultato."""
        installa_voce(client, VoceFinta())

        risposta = await client.post("/voice/speak", json={"testo": "Ciao"})

        assert risposta.status_code == 403
        assert "voce" in risposta.json()["detail"].lower()

    async def test_col_piano_giusto_si_parla(
        self, client, con_voce,  # noqa: F811
    ):
        installa_voce(client, VoceFinta())

        risposta = await client.post("/voice/speak", json={"testo": "Ciao"})

        assert risposta.status_code == 200

    async def test_l_elenco_delle_voci_pretende_lo_stesso_diritto(
        self, client, piani,  # noqa: F811
    ):
        installa_voce(client, VoceFinta())

        assert (await client.get("/voice/voices")).status_code == 403


class TestSintesi:
    async def test_torna_audio_e_non_fotogrammi(
        self, client, con_voce,  # noqa: F811
    ):
        installa_voce(client, VoceFinta(), AllineatoreFinto())

        corpo = (await client.post(
            "/voice/speak", json={"testo": "La vita è lunga"},
        )).json()

        assert base64.b64decode(corpo["audio"]) == b"suono"
        assert corpo["media_type"].startswith("audio/")
        assert "video" not in corpo

    async def test_i_visemi_arrivano_insieme_all_audio(
        self, client, con_voce,  # noqa: F811
    ):
        """Due risposte separate significherebbero un client che anima prima
        di avere il suono o riproduce prima di avere le forme."""
        installa_voce(client, VoceFinta(), AllineatoreFinto())

        corpo = (await client.post(
            "/voice/speak", json={"testo": "La vita è lunga"},
        )).json()

        assert corpo["visemi"]
        assert corpo["parole"]
        assert {v["forma"] for v in corpo["visemi"]} <= set(VISEMI)

    async def test_i_tempi_dichiarano_da_dove_vengono(
        self, client, con_voce,  # noqa: F811
    ):
        installa_voce(client, VoceFinta(), AllineatoreFinto())

        corpo = (await client.post(
            "/voice/speak", json={"testo": "Due parole"},
        )).json()

        assert corpo["origine_tempi"] == "allineamento"

    async def test_senza_allineamento_l_audio_arriva_senza_labiale(
        self, client, con_voce,  # noqa: F811
    ):
        """Non un errore: inventare tempi darebbe una bocca che si muove sulle
        parole sbagliate, peggio di una ferma."""
        installa_voce(client, VoceFinta(), None)

        corpo = (await client.post(
            "/voice/speak", json={"testo": "Due parole"},
        )).json()

        assert corpo["audio"]
        assert corpo["visemi"] == []
        assert corpo["origine_tempi"] == ""

    async def test_un_allineamento_che_fallisce_non_fa_fallire_la_voce(
        self, client, con_voce,  # noqa: F811
    ):
        installa_voce(client, VoceFinta(), AllineatoreFinto(riesce=False))

        risposta = await client.post("/voice/speak", json={"testo": "Ciao"})

        assert risposta.status_code == 200
        assert risposta.json()["visemi"] == []

    async def test_chiedere_di_saltare_il_labiale_risparmia_l_allineamento(
        self, client, con_voce,  # noqa: F811
    ):
        """Per un ritratto fermo misurare i tempi è lavoro buttato, e il
        client lo sa dal descrittore dell'avatar."""
        allineatore = AllineatoreFinto()
        installa_voce(client, VoceFinta(), allineatore)

        corpo = (await client.post(
            "/voice/speak", json={"testo": "Ciao", "labiale": False},
        )).json()

        assert corpo["visemi"] == []
        assert corpo["origine_tempi"] == ""

    async def test_un_fornitore_irraggiungibile_e_un_503(
        self, client, con_voce,  # noqa: F811
    ):
        """Non dipende dall'utente, e ripetere fra un minuto può funzionare."""
        installa_voce(client, VoceFinta(esplode="nessuna risposta"))

        risposta = await client.post("/voice/speak", json={"testo": "Ciao"})

        assert risposta.status_code == 503

    async def test_un_testo_sterminato_viene_rifiutato(
        self, client, con_voce,  # noqa: F811
    ):
        """Occuperebbe il fornitore per minuti per un file che nessuno
        ascolterà intero."""
        installa_voce(client, VoceFinta())

        risposta = await client.post(
            "/voice/speak", json={"testo": "a" * 5000},
        )

        assert risposta.status_code == 422

    async def test_la_voce_della_personalita_si_usa_se_non_se_ne_chiede_altra(
        self, client, con_voce, personalita, session,  # noqa: F811
    ):
        p, versione, _ = personalita
        versione.voice_config = {"voce": "grave"}
        await session.flush()

        fornitore = VoceFinta()
        installa_voce(client, fornitore, None)

        await client.post(
            "/voice/speak", json={"testo": "Ciao", "personality": p.slug},
        )

        assert fornitore.chiamate[-1]["voce"] == "grave"

    async def test_la_voce_chiesta_vince_su_quella_della_personalita(
        self, client, con_voce, personalita, session,  # noqa: F811
    ):
        p, versione, _ = personalita
        versione.voice_config = {"voce": "grave"}
        await session.flush()

        fornitore = VoceFinta()
        installa_voce(client, fornitore, None)

        await client.post("/voice/speak", json={
            "testo": "Ciao", "personality": p.slug, "voce": "acuta",
        })

        assert fornitore.chiamate[-1]["voce"] == "acuta"


class TestAvatarDiUnaPersonalita:
    async def test_senza_volto_non_e_un_errore(
        self, client, personalita,  # noqa: F811
    ):
        """La personalità esiste e non ha ritratto: è lo stato normale, e
        distinguerlo da «non esiste» è ciò che permette al client di mostrare
        il nome senza sembrare rotto."""
        p, _, _ = personalita

        risposta = await client.get(f"/personalities/{p.slug}/avatar")

        assert risposta.status_code == 200
        assert risposta.json()["avatar"] is None

    async def test_il_volto_arriva_come_descrittore(
        self, client, personalita, session,  # noqa: F811
    ):
        p, _, _ = personalita
        avatar = Avatar(
            slug=f"ritratto-{uuid.uuid4().hex[:6]}", name="Ritratto",
            kind="immagine", config={"uri": "https://esempio/x.jpg"},
        )
        session.add(avatar)
        await session.flush()
        p.avatar_id = avatar.id
        await session.flush()

        corpo = (await client.get(f"/personalities/{p.slug}/avatar")).json()

        assert corpo["avatar"]["tipo"] == "immagine"
        assert corpo["avatar"]["labiale"] is False
        assert corpo["avatar"]["uri"].endswith("x.jpg")

    async def test_un_avatar_configurato_male_lo_dice(
        self, client, personalita, session,  # noqa: F811
    ):
        """L'alternativa è un riquadro vuoto e un errore nella console del
        browser: nessuna informazione per chi l'ha configurato."""
        p, _, _ = personalita
        avatar = Avatar(
            slug=f"rotto-{uuid.uuid4().hex[:6]}", name="Rotto",
            kind="modello", config={"uri": "t.glb"},   # senza pose
        )
        session.add(avatar)
        await session.flush()
        p.avatar_id = avatar.id
        await session.flush()

        risposta = await client.get(f"/personalities/{p.slug}/avatar")

        assert risposta.status_code == 409
        assert "pose" in risposta.json()["detail"]

    async def test_il_ritratto_si_vede_anche_senza_il_diritto_alla_voce(
        self, client, personalita, session,  # noqa: F811
    ):
        """Nasconderlo a chi non ha il piano giusto renderebbe il catalogo
        anonimo proprio a chi sta decidendo se abbonarsi."""
        p, _, _ = personalita
        avatar = Avatar(
            slug=f"r-{uuid.uuid4().hex[:6]}", name="R",
            kind="immagine", config={"uri": "x.jpg"},
        )
        session.add(avatar)
        await session.flush()
        p.avatar_id = avatar.id
        await session.flush()

        # Nessun abbonamento: `con_voce` non è fra le fixture.
        risposta = await client.get(f"/personalities/{p.slug}/avatar")

        assert risposta.status_code == 200
        assert risposta.json()["avatar"] is not None


class AscoltoFinto:
    """Trascrive senza caricare Whisper."""

    name = "finto"

    def __init__(self, testo: str = "Ciao", fiducia: float = 0.9) -> None:
        self._testo = testo
        self._fiducia = fiducia
        self.ricevuti: List[bytes] = []

    async def ascolta(self, audio, *, media_type="", lingua="it"):
        from platform_core.voice.base import Ascolto

        self.ricevuti.append(audio)
        return Ascolto(
            testo=self._testo, lingua=lingua, fiducia=self._fiducia, durata=1.2,
        )


class TestDettatura:
    """Il ripiego del riconoscimento del browser.

    Vale la pena provarlo perché è la strada di chi non ha l'altra: un difetto
    qui non lo vede chi sviluppa — il suo browser il riconoscimento ce l'ha —
    e lo vede solo l'utente che ne è sprovvisto.
    """

    def _installa(self, client, stt):
        from platform_core.api.deps import get_stt

        client._transport.app.dependency_overrides[get_stt] = lambda: stt

    async def test_trascrive_e_restituisce_il_testo(
        self, client, con_voce,  # noqa: F811
    ):
        self._installa(client, AscoltoFinto("Il tempo è nostro"))

        risposta = await client.post(
            "/voice/listen",
            files={"file": ("d.webm", b"finto-audio", "audio/webm")},
        )

        assert risposta.status_code == 200
        assert risposta.json()["testo"] == "Il tempo è nostro"
        assert risposta.json()["da_confermare"] is False

    async def test_una_fiducia_bassa_chiede_conferma(
        self, client, con_voce,  # noqa: F811
    ):
        """Whisper produce testo verosimile anche su registrazioni pessime, e
        una frase inventata con sicurezza fa rispondere a una domanda che
        nessuno ha fatto."""
        self._installa(client, AscoltoFinto("boh", fiducia=0.3))

        corpo = (await client.post(
            "/voice/listen",
            files={"file": ("d.webm", b"rumore", "audio/webm")},
        )).json()

        assert corpo["da_confermare"] is True

    async def test_una_trascrizione_vuota_chiede_conferma(
        self, client, con_voce,  # noqa: F811
    ):
        """Mandare una stringa vuota come domanda farebbe rispondere al
        nulla."""
        self._installa(client, AscoltoFinto("", fiducia=0.99))

        corpo = (await client.post(
            "/voice/listen",
            files={"file": ("d.webm", b"silenzio", "audio/webm")},
        )).json()

        assert corpo["da_confermare"] is True

    async def test_senza_whisper_e_un_503_che_indica_l_altra_strada(
        self, client, con_voce,  # noqa: F811
    ):
        self._installa(client, None)

        risposta = await client.post(
            "/voice/listen",
            files={"file": ("d.webm", b"x", "audio/webm")},
        )

        assert risposta.status_code == 503
        assert "browser" in risposta.json()["detail"]

    async def test_una_registrazione_sterminata_viene_rifiutata(
        self, client, con_voce,  # noqa: F811
    ):
        """Oltre un paio di minuti è un caricamento travestito da dettatura, e
        va per l'ingestione — dove c'è una coda e nessuno che aspetti."""
        from platform_core.api.routers.voice import BYTE_MASSIMI

        self._installa(client, AscoltoFinto())

        risposta = await client.post(
            "/voice/listen",
            files={"file": ("d.webm", b"x" * (BYTE_MASSIMI + 10), "audio/webm")},
        )

        assert risposta.status_code == 413

    async def test_la_dettatura_pretende_il_diritto_alla_voce(
        self, client, piani,  # noqa: F811
    ):
        self._installa(client, AscoltoFinto())

        risposta = await client.post(
            "/voice/listen",
            files={"file": ("d.webm", b"x", "audio/webm")},
        )

        assert risposta.status_code == 403
