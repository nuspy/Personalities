"""Guardrail e verifica di fondatezza.

Il test che conta è `TestRischioDelloStile`: protegge la proprietà da cui
dipende l'utilizzabilità del controllo. Se il groundcheck segnala come «non
ancorata» ogni risposta scritta in carattere, qualcuno lo spegnerà — e con
esso sparirà anche la rilevazione delle invenzioni vere. Il rimedio non è una
soglia indulgente: è che solo ciò che asserisce un fatto possa essere
infondato.
"""
from __future__ import annotations

import textwrap
import uuid

import pytest

from platform_core.guards.claims import (
    Affermazione, EsitoGroundcheck, Tipo, tipo_da,
)
from platform_core.guards.groundcheck import (
    Giudice, controlla_citazioni, riferimenti_citati,
)
from platform_core.guards.policy import (
    Guardrail, GuardrailMalformato, RegistroGuardrail, carica_da, interpreta,
)
from platform_core.knowledge.retriever import PassaggioRecuperato
from platform_core.knowledge.vector_store import Corrispondenza
from platform_core.llm.base import GenerationError
from platform_core.paths import DATA_DIR


def passaggio(etichetta: str, testo: str = "un passaggio") -> PassaggioRecuperato:
    p = PassaggioRecuperato(
        corrispondenza=Corrispondenza(
            chunk_id=uuid.uuid4(), testo=testo, ordinale=0, sezione=None,
            documento_id=uuid.uuid4(), documento_titolo="Fonte",
            documento_uri=None, kb_id=uuid.uuid4(), punteggio=0.8,
        ),
        punteggio_rrf=0.03,
    )
    p.etichetta = etichetta
    return p


GUARDRAIL_MINIMO = textwrap.dedent("""\
    ---
    slug: prova
    nome: Guardrail di prova
    severita: segnala
    ---

    ## Istruzioni

    Non inventare nulla.

    ## Verifica

    Segnala le affermazioni non sostenute.
    """)


class TestLetturaDeiGuardrail:
    def test_le_due_sezioni_hanno_destini_diversi(self):
        g = interpreta(GUARDRAIL_MINIMO)

        assert g.previene and g.rileva
        assert "Non inventare" in g.istruzioni
        assert "Segnala le affermazioni" in g.verifica

    def test_solo_prevenzione(self):
        testo = "---\nslug: p\n---\n\n## Istruzioni\n\nQualcosa.\n"
        g = interpreta(testo)

        assert g.previene and not g.rileva

    def test_solo_rilevazione(self):
        """Un guardrail può misurare senza influenzare: è il modo di scoprire
        quanto spesso un problema si presenti prima di decidere come
        affrontarlo."""
        testo = "---\nslug: p\n---\n\n## Verifica\n\nQualcosa.\n"
        g = interpreta(testo)

        assert g.rileva and not g.previene

    def test_senza_sezioni_viene_rifiutato(self):
        testo = "---\nslug: p\n---\n\nSolo prosa senza sezioni.\n"
        with pytest.raises(GuardrailMalformato, match="non previene e non rileva"):
            interpreta(testo)

    def test_senza_frontmatter_viene_rifiutato(self):
        with pytest.raises(GuardrailMalformato, match="frontmatter"):
            interpreta("## Istruzioni\n\nQualcosa.")

    def test_senza_slug_viene_rifiutato(self):
        with pytest.raises(GuardrailMalformato, match="slug"):
            interpreta("---\nnome: Senza slug\n---\n\n## Istruzioni\n\nX.\n")

    def test_severita_sconosciuta_viene_rifiutata(self):
        testo = "---\nslug: p\nseverita: distruggi\n---\n\n## Istruzioni\n\nX.\n"
        with pytest.raises(GuardrailMalformato, match="severità"):
            interpreta(testo)

    def test_un_file_rotto_non_ferma_gli_altri(self, tmp_path):
        """Un refuso in un guardrail accessorio non deve disattivare quelli che
        contano."""
        (tmp_path / "buono.md").write_text(GUARDRAIL_MINIMO, encoding="utf-8")
        (tmp_path / "rotto.md").write_text("niente frontmatter", encoding="utf-8")

        caricati = carica_da(tmp_path)

        assert [g.slug for g in caricati] == ["prova"]

    def test_una_cartella_inesistente_non_esplode(self, tmp_path):
        assert carica_da(tmp_path / "non-esiste") == []


class TestGuardrailDelProgetto:
    """I guardrail veri, quelli in `data/guards/`."""

    def test_si_caricano(self):
        registro = RegistroGuardrail()
        assert len(registro) >= 3

    def test_la_fondatezza_esiste_ed_e_completa(self):
        registro = RegistroGuardrail()
        fondatezza = next(g for g in registro.per() if g.slug == "fondatezza")

        assert fondatezza.previene and fondatezza.rileva

    def test_la_rubrica_esclude_lo_stile_esplicitamente(self):
        """Il testo della rubrica è ciò che protegge dal rischio: se qualcuno
        lo riscrive senza quell'elenco, il giudice comincia a misurare la
        voce."""
        registro = RegistroGuardrail()
        fondatezza = next(g for g in registro.per() if g.slug == "fondatezza")

        for parola in ("esortazioni", "massime", "carattere"):
            assert parola in fondatezza.verifica.lower(), (
                f"la rubrica non esclude più «{parola}»"
            )


class TestFiltroPerPersonalita:
    def test_senza_vincolo_vale_per_tutte(self):
        g = Guardrail(slug="x", nome="X", istruzioni="Y")
        assert g.vale_per("seneca") and g.vale_per(None)

    def test_con_vincolo_vale_solo_per_quelle(self):
        g = Guardrail(slug="x", nome="X", istruzioni="Y", applica_a=("seneca",))
        assert g.vale_per("seneca")
        assert not g.vale_per("rockefeller")

    def test_il_registro_filtra(self):
        registro = RegistroGuardrail([
            Guardrail(slug="a", nome="A", istruzioni="uno"),
            Guardrail(slug="b", nome="B", istruzioni="due", applica_a=("altra",)),
        ])

        assert [g.slug for g in registro.per("seneca")] == ["a"]


class TestOrdineStabile:
    def test_le_istruzioni_escono_ordinate(self):
        """Lo strato 0 dev'essere byte-identico fra richieste: un ordine che
        dipende da come i file sono stati letti dal disco non lo è."""
        registro = RegistroGuardrail([
            Guardrail(slug="zeta", nome="Z", istruzioni="ultima"),
            Guardrail(slug="alfa", nome="A", istruzioni="prima"),
        ])

        assert registro.istruzioni_per() == ["prima", "ultima"]


class TestCitazioni:
    def test_riconosce_i_riferimenti(self):
        assert riferimenti_citati("Così [K1], e anche [K12].") == {"K1", "K12"}

    def test_una_citazione_inesistente_viene_rilevata(self):
        esito = controlla_citazioni("Come noto [K9].", ["K1", "K2"])

        assert esito.riferimenti_inventati == ["K9"]
        assert not esito.fondata

    def test_le_citazioni_valide_passano(self):
        esito = controlla_citazioni("Come noto [K1].", ["K1", "K2"])

        assert esito.riferimenti_inventati == []
        assert esito.fondata

    def test_una_risposta_senza_citazioni_e_fondata(self):
        """Non citare non è un difetto: dipende da cosa si sta dicendo."""
        assert controlla_citazioni("Non temere.", ["K1"]).fondata

    def test_il_livello_e_dichiarato(self):
        """«Verificato con il confronto» e «verificato da un giudice» sono
        garanzie diverse, e chi legge la traccia deve poterle distinguere."""
        assert controlla_citazioni("x", []).livello == "citations"


class TestRischioDelloStile:
    """La proprietà da cui dipende l'utilizzabilità del controllo."""

    def test_solo_i_fatti_possono_essere_infondati(self):
        for tipo in Tipo:
            a = Affermazione(testo="x", tipo=tipo, sostenuta=False)
            assert a.infondata == (tipo is Tipo.FATTO)

    def test_una_risposta_tutta_in_carattere_e_fondata(self):
        """Il caso peggiore: esortazioni e massime, nessun fatto.

        È anche la risposta meglio riuscita che il sistema possa produrre.
        Segnalarla sarebbe l'errore che porta a spegnere la verifica.
        """
        esito = EsitoGroundcheck(livello="nli", affermazioni=[
            Affermazione("Non temere la morte.", Tipo.ESORTAZIONE),
            Affermazione("La virtù è l'unico bene.", Tipo.MASSIMA),
            Affermazione("Ti scrivo come a un amico.", Tipo.STILE),
            Affermazione("Ho conosciuto la malattia.", Tipo.ESPERIENZA),
        ])

        assert esito.fondata
        assert esito.fatti == []

    def test_un_solo_fatto_infondato_basta(self):
        esito = EsitoGroundcheck(livello="nli", affermazioni=[
            Affermazione("Non temere.", Tipo.ESORTAZIONE),
            Affermazione("Nerone morì nel 68.", Tipo.FATTO, sostenuta=False),
        ])

        assert not esito.fondata
        assert len(esito.infondate) == 1

    def test_un_etichetta_ignota_non_diventa_un_accusa(self):
        """Il costo di sbagliare verso `stile` è un'invenzione non rilevata;
        verso `fatto` è un sistema che qualcuno spegne."""
        assert tipo_da("qualcosa-di-mai-visto") is Tipo.STILE
        assert tipo_da(None) is Tipo.STILE

    def test_i_sinonimi_inglesi_si_riconoscono(self):
        """I modelli li producono spontaneamente, qualunque lingua usi il prompt."""
        assert tipo_da("factual") is Tipo.FATTO
        assert tipo_da("stylistic") is Tipo.STILE


class TestGiudice:
    """Il livello `nli`, con un fornitore controllato."""

    class ProviderFinto:
        def __init__(self, risposta=None, errore=None):
            self.risposta = risposta
            self.errore = errore
            self.richieste = []

        async def complete_json(self, request):
            self.richieste.append(request)
            if self.errore:
                raise self.errore
            return self.risposta

    async def test_il_giudice_non_vede_il_prompt_della_personalita(self):
        """Un giudice a cui si dice «stai leggendo Seneca» valuta se la
        risposta *suoni* come Seneca, e accetta un'invenzione ben scritta."""
        provider = self.ProviderFinto({"affermazioni": []})

        await Giudice(provider).valuta("una risposta", [passaggio("K1")])

        inviato = "\n".join(m.content for m in provider.richieste[0].messages)
        assert "Sei Seneca" not in inviato
        assert "verificatore" in inviato.lower()

    async def test_vede_i_passaggi_e_la_risposta(self):
        provider = self.ProviderFinto({"affermazioni": []})

        await Giudice(provider).valuta(
            "la mia risposta", [passaggio("K1", "il testo del passaggio")],
        )

        inviato = provider.richieste[0].messages[-1].content
        assert "il testo del passaggio" in inviato
        assert "la mia risposta" in inviato

    async def test_interpreta_il_verdetto(self):
        provider = self.ProviderFinto({"affermazioni": [
            {"testo": "Un fatto", "tipo": "fatto", "sostenuta": True,
             "riferimenti": ["K1"]},
            {"testo": "Non temere", "tipo": "esortazione"},
        ]})

        esito = await Giudice(provider).valuta("x [K1]", [passaggio("K1")])

        assert esito.livello == "nli"
        assert esito.fondata
        assert len(esito.fatti) == 1

    async def test_un_giudice_muto_non_approva(self):
        """Trattare un guasto del verificatore come un via libera
        significherebbe che ogni suo malfunzionamento approva tutto."""
        provider = self.ProviderFinto(errore=GenerationError("irraggiungibile"))

        esito = await Giudice(provider).valuta("x", [passaggio("K1")])

        assert esito.non_eseguito
        assert "irraggiungibile" in esito.non_eseguito

    async def test_le_citazioni_si_controllano_comunque(self):
        """Il livello deterministico non dipende dal giudice: vale anche
        quando quello tace."""
        provider = self.ProviderFinto(errore=GenerationError("giù"))

        esito = await Giudice(provider).valuta("come noto [K9]", [passaggio("K1")])

        assert esito.riferimenti_inventati == ["K9"]

    async def test_le_rubriche_entrano_nel_prompt_del_giudice(self):
        provider = self.ProviderFinto({"affermazioni": []})
        rubrica = Guardrail(
            slug="x", nome="Regola X", verifica="Cerca le cose strane.",
        )

        await Giudice(provider).valuta(
            "x", [passaggio("K1")], rubriche=[rubrica],
        )

        inviato = provider.richieste[0].messages[0].content
        assert "Cerca le cose strane" in inviato

    async def test_una_risposta_vuota_non_costa_una_chiamata(self):
        provider = self.ProviderFinto({"affermazioni": []})

        await Giudice(provider).valuta("   ", [passaggio("K1")])

        assert provider.richieste == []

    async def test_tollera_una_forma_diversa_del_verdetto(self):
        """I modelli rinominano le chiavi: `claims` invece di `affermazioni`."""
        provider = self.ProviderFinto({"claims": [
            {"text": "A fact", "type": "factual", "supported": True},
        ]})

        esito = await Giudice(provider).valuta("x", [passaggio("K1")])

        assert len(esito.fatti) == 1
        assert esito.fondata
