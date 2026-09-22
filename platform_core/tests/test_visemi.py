"""Le forme della bocca.

Codice puro, senza database: si prova tutto, e conviene, perché è la parte
dove un errore non produce nessun messaggio — solo un volto che si muove in
modo vagamente sbagliato, che nessuno sa da dove venga.

Tre proprietà da difendere:

1. l'ortografia italiana si legge come si pronuncia (`gli`, `gn`, `sc`, `ch`,
   le doppie), o il labiale scandisce lettere invece di suoni;
2. i tempi coprono l'audio senza buchi né sovrapposizioni, o la bocca salta;
3. nessuna forma dura meno di quanto una bocca impieghi a raggiungerla, o
   l'animazione trema invece di parlare.
"""
from __future__ import annotations

import pytest

from platform_core.voice.visemi import (
    DURATA_MINIMA, VISEMI, Parola, forme_di, visemi_da,
)


class TestOrtografia:
    @pytest.mark.parametrize(
        "parola, atteso",
        [
            ("ciao", ["CH", "A", "O"]),
            ("quando", ["KG", "A", "NN", "TH", "O"]),
            ("perché", ["MBP", "E", "RR", "KG", "E"]),
            ("bisogno", ["MBP", "I", "SS", "O", "NN", "O"]),
            ("famiglia", ["FV", "A", "MBP", "I", "L", "A"]),
            ("scienza", ["CH", "E", "NN", "SS", "A"]),
        ],
    )
    def test_i_digrammi_italiani_danno_una_forma_sola(self, parola, atteso):
        """`gn` non è «g» più «n», `gli` non è «g-l-i»: pronunciarli lettera
        per lettera è esattamente ciò che fa sembrare un labiale una macchina
        che compita."""
        assert forme_di(parola) == atteso

    def test_le_doppie_non_raddoppiano_la_forma(self):
        """Le labbra si chiudono una volta per «babbo», non due, e
        raddoppiarla produce uno scatto."""
        assert forme_di("babbo") == ["MBP", "A", "MBP", "O"]
        assert forme_di("ferro") == ["FV", "E", "RR", "O"]

    def test_l_acca_non_ha_forma(self):
        """Muta in italiano: darle una forma fermerebbe la bocca su un suono
        che non esiste."""
        assert forme_di("ho") == ["O"]
        assert forme_di("hanno") == ["A", "NN", "O"]

    def test_gli_accenti_non_cambiano_la_bocca(self):
        assert forme_di("perché") == forme_di("perche")
        assert forme_di("città") == forme_di("citta")

    def test_la_punteggiatura_si_ignora(self):
        assert forme_di("ciao!") == forme_di("ciao")
        assert forme_di("«ciao»") == forme_di("ciao")

    def test_ogni_forma_prodotta_e_una_di_quelle_dichiarate(self):
        """Il client disegna solo le forme che conosce: una fuori elenco
        sarebbe una bocca che non si muove, senza errori da nessuna parte."""
        testo = (
            "Non è che abbiamo poco tempo, ma ne perdiamo molto: "
            "la vita che riceviamo è abbastanza lunga."
        )
        for parola in testo.split():
            assert set(forme_di(parola)) <= set(VISEMI)

    def test_una_parola_senza_lettere_non_produce_forme(self):
        assert forme_di("—") == []
        assert forme_di("...") == []


class TestTempi:
    def test_le_forme_coprono_la_parola_senza_buchi(self):
        visemi = visemi_da([Parola("mondo", 0.0, 1.0)])

        assert visemi[0].inizio == pytest.approx(0.0)
        assert visemi[-1].fine == pytest.approx(1.0)
        for prima, dopo in zip(visemi, visemi[1:]):
            assert dopo.inizio == pytest.approx(prima.fine)

    def test_fra_le_parole_la_bocca_torna_a_riposo(self):
        """Senza, una pausa lascerebbe l'ultima forma congelata sul volto —
        l'effetto per cui un'animazione sembra bloccata invece che silenziosa."""
        visemi = visemi_da([
            Parola("ciao", 0.0, 0.4), Parola("mondo", 1.2, 1.8),
        ])

        riposo = [v for v in visemi if v.forma == "X"]
        assert riposo, "nessun ritorno a riposo nella pausa"
        assert any(
            v.inizio == pytest.approx(0.4) and v.fine == pytest.approx(1.2)
            for v in riposo
        )

    def test_la_coda_dopo_l_ultima_parola_e_silenzio(self):
        visemi = visemi_da([Parola("ciao", 0.0, 0.4)], durata_totale=2.0)

        assert visemi[-1].forma == "X"
        assert visemi[-1].fine == pytest.approx(2.0)

    def test_una_pausa_impercettibile_non_produce_un_riposo(self):
        """Dieci millisecondi fra due parole non sono una pausa: infilarci una
        forma a riposo farebbe chiudere e riaprire la bocca dentro una frase
        continua."""
        visemi = visemi_da([
            Parola("la", 0.0, 0.2), Parola("vita", 0.21, 0.6),
        ])

        assert not any(v.forma == "X" for v in visemi)

    def test_nessuna_forma_dura_meno_del_minimo(self):
        """Sotto i sessanta millisecondi la bocca non raggiunge la posizione:
        il risultato è un tremolio, non un labiale."""
        visemi = visemi_da([Parola("straordinariamente", 0.0, 0.5)])

        assert visemi
        for v in visemi:
            assert v.durata >= DURATA_MINIMA - 1e-9, f"{v.forma} dura {v.durata}"

    def test_fondere_le_brevi_non_cambia_la_durata_totale(self):
        """La forma precedente si allunga: il tempo complessivo resta quello
        dell'audio, o il labiale finirebbe prima o dopo il suono."""
        parole = [Parola("costituzionalmente", 0.0, 0.4), Parola("no", 0.4, 0.9)]

        visemi = visemi_da(parole)

        assert visemi[0].inizio == pytest.approx(0.0)
        assert visemi[-1].fine == pytest.approx(0.9)

    def test_due_forme_uguali_di_seguito_si_fondono(self):
        """«la ampia»: la `a` finale e quella iniziale sono la stessa forma, e
        riaprirla a metà fa scattare la mascella."""
        visemi = visemi_da([Parola("la", 0.0, 0.3), Parola("ampia", 0.3, 0.9)])

        for prima, dopo in zip(visemi, visemi[1:]):
            assert prima.forma != dopo.forma

    def test_una_parola_senza_forme_non_rompe_i_tempi(self):
        """Un trattino fra due parole ha tempi ma nessun suono da disegnare."""
        visemi = visemi_da([
            Parola("ciao", 0.0, 0.4),
            Parola("—", 0.4, 0.5),
            Parola("mondo", 0.5, 1.0),
        ])

        assert visemi
        assert visemi[-1].fine == pytest.approx(1.0)

    def test_senza_parole_non_ci_sono_forme(self):
        assert visemi_da([]) == []

    def test_un_audio_muto_non_produce_una_bocca_immobile_e_basta(self):
        """Una sola forma sotto soglia si allunga invece di sparire: una bocca
        senza nessuna forma non si muove affatto, e sembra un guasto."""
        visemi = visemi_da([Parola("a", 0.0, 0.01)])

        assert len(visemi) == 1
        assert visemi[0].durata >= DURATA_MINIMA - 1e-9
