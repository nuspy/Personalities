"""Il capolettera spezzato.

Nelle edizioni a stampa la prima lettera di un paragrafo è grande e decorata;
chi digitalizza la marca come elemento a sé, e l'estrazione del testo la lascia
staccata: «D opo tanto tempo ho riveduto i tuoi luoghi Pompei».

Il danno non è tipografico. Quel passaggio non si trova più cercando «dopo» —
nell'indice compare la radice `opo`, che non è una parola di nessuna lingua —
e un adapter addestrato su quel testo impara a scrivere «D opo».

**La tentazione è una regola sulla forma**, e non funziona: «O tre volte
beati» comincia con una maiuscola isolata e unirla darebbe «Otre». Il
discriminante che funziona guarda la frequenza, non la forma.
"""
from __future__ import annotations

import pytest

from platform_core.knowledge.normalization import (
    RAPPORTO_MINIMO, ricongiungi_capolettera, vocabolario,
)


class TestVocabolario:
    def test_conta_le_parole(self):
        v = vocabolario(["la virtù e la vita", "la vita è breve"])
        assert v["la"] == 3
        assert v["vita"] == 2

    def test_ignora_maiuscole_e_punteggiatura(self):
        v = vocabolario(["Dopo tanto tempo.", "dopo poco"])
        assert v["dopo"] == 2

    def test_le_lettere_singole_non_contano(self):
        """Servono le parole, e una lettera sola non lo è."""
        v = vocabolario(["D opo tanto tempo"])
        assert "d" not in v
        assert v["opo"] == 1


class TestCapolettera:
    #: Un corpus in cui «dopo» è comune e «opo» compare solo nel testo rotto.
    CORPUS = [
        "D opo tanto tempo ho riveduto i tuoi luoghi Pompei.",
        "Dopo quella lettera non scrissi più.",
        "Molto dopo, quando tutto era finito, capii.",
        "E dopo averlo detto, tacque a lungo.",
    ]

    def test_ricompone_la_parola_spezzata(self):
        esito = ricongiungi_capolettera(self.CORPUS)

        assert esito.ricongiunti == 1
        assert esito.testi[0].startswith("Dopo tanto tempo")

    def test_non_tocca_il_resto_del_testo(self):
        esito = ricongiungi_capolettera(self.CORPUS)

        assert esito.testi[0] == "Dopo tanto tempo ho riveduto i tuoi luoghi Pompei."
        assert esito.testi[1:] == self.CORPUS[1:]

    def test_una_congiunzione_resta_una_congiunzione(self):
        """«E che» non diventa «Eche»: `che` è una parola frequentissima, e la
        lettera isolata è una parola a sé."""
        corpus = [
            "E che dirai di questo?",
            "Non so che fare.",
            "Che tu sia felice.",
            "Dimmi che cosa pensi.",
        ]
        esito = ricongiungi_capolettera(corpus)

        assert esito.ricongiunti == 0
        assert esito.testi[0] == corpus[0]

    def test_un_interiezione_resta_separata(self):
        """Il caso che smonta la regola sulla forma: «O tre volte beati»
        comincia con una maiuscola isolata ma `tre` è una parola vera."""
        corpus = [
            "O tre volte beati quelli che morirono in patria.",
            "Ne restano tre soltanto.",
            "Tre giorni dopo tornò.",
        ]
        esito = ricongiungi_capolettera(corpus)

        assert esito.ricongiunti == 0
        assert esito.testi[0].startswith("O tre")

    def test_la_decisione_e_ispezionabile(self):
        """Chi dubita di una fusione deve poterla controllare senza rileggere
        l'intero corpus."""
        esito = ricongiungi_capolettera(self.CORPUS)
        prima, dopo, unito = esito.decisioni[0]

        assert prima == "D opo"
        assert dopo == "Dopo"
        assert unito is True


class TestPosizione:
    def test_anche_dopo_l_apparato_del_curatore(self):
        """Quando il passaggio comincia con un richiamo bibliografico, il
        capolettera non è il primo carattere: cercarlo solo in testa non ne
        troverebbe nemmeno uno."""
        corpus = [
            "Post longum intervallum etc. Ep. lxx . D opo tanto tempo ho riveduto.",
            "Dopo la partenza non tornò.",
            "Molto dopo capii il motivo.",
        ]
        esito = ricongiungi_capolettera(corpus)

        assert esito.ricongiunti == 1
        assert "Dopo tanto tempo" in esito.testi[0]

    def test_in_mezzo_a_una_frase_non_si_tocca(self):
        """Lì una maiuscola isolata non è un capolettera, e il vocabolario lo
        conferma comunque."""
        corpus = ["Scrisse a D e poi partì.", "La lettera a D arrivò tardi."]
        esito = ricongiungi_capolettera(corpus)

        assert esito.testi == corpus


class TestCorpusComeDizionario:
    def test_senza_confronti_non_si_unisce(self):
        """Un corpus di un testo solo non ha dati per decidere: nel dubbio si
        lascia com'è, perché una fusione sbagliata perde una parola e non si
        recupera."""
        esito = ricongiungi_capolettera(["D opo tanto tempo."])
        assert esito.ricongiunti == 0

    def test_basta_superare_il_frammento(self):
        """La soglia è deliberatamente bassa: i due casi sono così distanti —
        diciannove contro uno, zero contro novecento — che alzarla
        escluderebbe solo i capolettera rari, cioè quelli che nessun altro
        meccanismo recupererebbe."""
        assert RAPPORTO_MINIMO == 1.0

    def test_la_lingua_non_serve(self):
        """Il criterio è statistico: funziona su un corpus latino come su uno
        italiano, senza sapere quale sia."""
        corpus = [
            "Q uisque suam fortunam facit hoc modo.",
            "Quisque enim suam habet.",
            "Nam quisque pro se laborat.",
        ]
        esito = ricongiungi_capolettera(corpus)

        assert esito.ricongiunti == 1
        assert esito.testi[0].startswith("Quisque")
