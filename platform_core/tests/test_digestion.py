"""La digestione dei corpora.

Il test che conta di più è `TestDifesaSullaPulizia`. La pulizia è l'unica
operazione della digestione che **modifica il testo**, e su un corpus reale il
modello ha proposto di togliere la prosa dell'autore lasciando l'incipit
latino del curatore — l'esatto contrario di ciò che serviva. La stringa
c'era davvero nel passaggio, quindi una verifica sulla sola presenza l'avrebbe
accettata.

La difesa è che l'apparato **sta sempre in testa**: pretendere che la
rimozione sia un prefisso rende quell'errore impossibile invece che
improbabile.
"""
from __future__ import annotations

import pytest

from platform_core.knowledge.digest_runner import (
    SOGLIA_DUPLICATO, _perche_non_rimuovere,
)
from platform_core.knowledge.digestion import (
    CARATTERI_MINIMI, SOGLIA_QUALITA, _interpreta, scarto_a_vista,
)
from platform_core.knowledge.taxonomy import (
    Categoria, Etichettatura, Provenienza, categoria_da, provenienza_da,
)
from platform_core.llm.openai_compatible import _sembra_troncato


class TestTassonomia:
    def test_le_categorie_della_voce(self):
        """Si impara come parla da chi parla, non da chi lo descrive."""
        assert Categoria.LESSICO.insegna_la_voce
        assert Categoria.STILE.insegna_la_voce
        assert Categoria.PENSIERO.insegna_la_voce
        assert not Categoria.CARATTERE.insegna_la_voce
        assert not Categoria.AVVENIMENTO.insegna_la_voce

    def test_le_categorie_che_portano_fatti(self):
        """Su un passaggio di stile non c'è nulla da fondare."""
        assert Categoria.AVVENIMENTO.porta_fatti
        assert not Categoria.STILE.porta_fatti
        assert not Categoria.LESSICO.porta_fatti

    def test_l_apparato_si_scarta(self):
        assert Categoria.APPARATO.da_scartare
        assert not Categoria.VALORI.da_scartare

    def test_l_autore_pesa_piu_di_tutti(self):
        """Non è attendibilità storica: è quanto avvicina a come parlava."""
        assert Provenienza.AUTORE.peso > Provenienza.CONTEMPORANEO.peso
        assert Provenienza.CONTEMPORANEO.peso > Provenienza.POSTERIORE.peso
        assert Provenienza.EDITORIALE.peso < Provenienza.IGNOTA.peso

    def test_un_etichetta_ignota_non_si_forza(self):
        """Tradurla a forza attribuirebbe al passaggio una natura che nessuno
        gli ha riconosciuto."""
        assert categoria_da("qualcosa-di-mai-visto") is None
        assert categoria_da("") is None

    def test_i_sinonimi_si_riconoscono(self):
        assert categoria_da("vocabolario") is Categoria.LESSICO
        assert categoria_da("ragionamento") is Categoria.PENSIERO
        assert provenienza_da("curatore") is Provenienza.EDITORIALE

    def test_la_provenienza_ignota_e_uno_stato_legittimo(self):
        assert provenienza_da(None) is Provenienza.IGNOTA
        assert provenienza_da("boh") is Provenienza.IGNOTA


class TestEtichettatura:
    def test_la_principale_e_la_piu_alta(self):
        e = Etichettatura(categorie={
            Categoria.LESSICO: 0.4, Categoria.VALORI: 0.9,
        })
        assert e.principale is Categoria.VALORI

    def test_senza_categorie_non_c_e_principale(self):
        assert Etichettatura().principale is None

    def test_insegna_la_voce_anche_come_seconda(self):
        """Un passaggio per metà racconto e per metà lessico insegna comunque
        come parla."""
        e = Etichettatura(categorie={
            Categoria.AVVENIMENTO: 0.9, Categoria.LESSICO: 0.5,
        })
        assert e.insegna_la_voce

    def test_una_traccia_debole_non_basta(self):
        e = Etichettatura(categorie={
            Categoria.AVVENIMENTO: 0.9, Categoria.LESSICO: 0.1,
        })
        assert not e.insegna_la_voce


class TestScartoAVista:
    """Il livello che non costa nulla, e va prima di tutto."""

    def test_un_frammento_breve_non_insegna(self):
        assert scarto_a_vista("Ok.")

    def test_i_numeri_di_pagina(self):
        assert scarto_a_vista("  142  \n\n  143  \n\n 144 " * 4)

    def test_una_riga_di_indice(self):
        assert scarto_a_vista(
            "pag. 45 — Della brevità della vita, e di come si sprechi "
            "il tempo che pure ci appartiene per intero, secondo il libro terzo"
        )

    def test_una_tavola_di_numeri(self):
        assert scarto_a_vista(
            "1789 12 1790 15 1791 18 1792 22 1793 27 1794 31 1795 40 "
            "1796 44 1797 51 1798 59 1799 63 1800 70 1801 77 1802 81"
        )

    def test_la_prosa_passa(self):
        testo = (
            "La virtù non si impara a parole ma con l'esercizio quotidiano, e "
            "chi rimanda l'esercizio rimanda la vita stessa. Nessuno diventa "
            "saggio per caso, né la saggezza arriva col tempo soltanto."
        )
        assert scarto_a_vista(testo) == ""

    def test_la_soglia_e_dichiarata(self):
        assert CARATTERI_MINIMI >= 100


class TestDifesaSullaPulizia:
    """La difesa nata da un errore reale."""

    TESTO = (
        "Magnam ex epistola tua percepi voluptatem etc. Ep. lix . "
        "G ran piacere ho preso dalla tua epistola: concedimi ch'io mi serva "
        "di questo modo di parlar comune, nè lo tirare a quel senso che lo "
        "tirano gli Stoici. Perchè con tutto che noi crediamo."
    )

    def test_un_prefisso_vero_si_toglie(self):
        assert _perche_non_rimuovere(
            self.TESTO, "Magnam ex epistola tua percepi voluptatem etc. Ep. lix .",
        ) == ""

    def test_la_prosa_dell_autore_non_si_toglie(self):
        """Il caso accaduto: la stringa c'era davvero nel passaggio, e una
        verifica sulla sola presenza l'avrebbe accettata."""
        motivo = _perche_non_rimuovere(
            self.TESTO,
            "G ran piacere ho preso dalla tua epistola: concedimi ch'io mi "
            "serva di questo modo di parlar comune, nè lo tirare a quel senso "
            "che lo tirano gli Stoici.",
        )
        assert motivo == "non è in testa al passaggio"

    def test_una_rimozione_enorme_viene_fermata(self):
        """Sopra un terzo non è un richiamo bibliografico: è un riassunto."""
        motivo = _perche_non_rimuovere(self.TESTO, self.TESTO[: int(len(self.TESTO) * 0.5)])
        assert "troppo lunga" in motivo

    def test_una_stringa_assente_non_si_applica(self):
        assert _perche_non_rimuovere(self.TESTO, "qualcosa che non c'è")

    def test_un_passaggio_troppo_corto_e_gia_stato_scartato(self):
        """Il motivo per cui basta una difesa sulla proporzione.

        Perché dopo una rimozione sotto un terzo resti un frammento inutile,
        il passaggio dovrebbe essere più corto della soglia con cui
        `scarto_a_vista` lo ha già tolto di mezzo. Le due regole si coprono a
        vicenda, e una terza non avrebbe mai occasione di scattare.
        """
        corto = "Ep. lxvii . O tre volte beati."
        assert scarto_a_vista(corto), "sarebbe già fuori prima della pulizia"


class TestInterpretazione:
    def test_la_forma_attesa(self):
        etichette = _interpreta({"passaggi": [
            {"n": 1, "categorie": {"lessico": 0.9, "valori": 0.6},
             "provenienza": "autore", "qualita": 0.9,
             "da_togliere": "Ep. x ."},
        ]}, 1)

        assert len(etichette) == 1
        assert etichette[0].principale is Categoria.LESSICO
        assert etichette[0].provenienza is Provenienza.AUTORE
        assert etichette[0].da_togliere == "Ep. x ."

    def test_l_apparato_dominante_si_scarta(self):
        etichette = _interpreta({"passaggi": [
            {"n": 1, "categorie": {"apparato": 0.9}, "qualita": 0.8},
        ]}, 1)

        assert etichette[0].scartato
        assert "apparato" in etichette[0].motivo_scarto

    def test_la_qualita_bassa_si_scarta(self):
        etichette = _interpreta({"passaggi": [
            {"n": 1, "categorie": {"contesto": 0.5},
             "qualita": SOGLIA_QUALITA - 0.1},
        ]}, 1)

        assert etichette[0].scartato

    def test_i_passaggi_saltati_restano_senza_etichetta(self):
        """Si riprendono al giro successivo invece di essere scartati per
        omissione: una classificazione mancata non è un giudizio."""
        etichette = _interpreta({"passaggi": [{"n": 2, "categorie": {"stile": 0.8}}]}, 3)

        assert len(etichette) == 3
        assert etichette[0].categorie == {}
        assert etichette[1].principale is Categoria.STILE
        assert etichette[2].categorie == {}

    def test_senza_n_si_usa_la_posizione(self):
        etichette = _interpreta({"passaggi": [
            {"categorie": {"stile": 0.8}}, {"categorie": {"valori": 0.9}},
        ]}, 2)

        assert etichette[0].principale is Categoria.STILE
        assert etichette[1].principale is Categoria.VALORI

    def test_una_risposta_illeggibile_non_esplode(self):
        assert len(_interpreta("non è un oggetto", 3)) == 3
        assert len(_interpreta(None, 2)) == 2


class TestTroncamento:
    """Il difetto che ha fatto fallire la prima digestione."""

    def test_un_json_completo_passa(self):
        assert not _sembra_troncato('{"a": 1}')
        assert not _sembra_troncato('```json\n{"a": [1, 2]}\n```')

    def test_un_json_interrotto_si_riconosce(self):
        """Quattro lotti su dodici persi, con in log un JSON che cominciava
        bene e si fermava a metà. Il budget adattivo c'era e non scattava mai,
        perché si attiva solo su risposta vuota."""
        assert _sembra_troncato('{"passaggi": [{"n": 1, "categorie": {"lessico": 0.9')

    def test_una_stringa_non_chiusa(self):
        assert _sembra_troncato('{"a": "testo che non finisce')

    def test_la_prosa_non_e_un_json_troncato(self):
        """Riprovare con più spazio non cambierebbe nulla."""
        assert not _sembra_troncato("Mi dispiace, non posso aiutarti.")

    def test_le_graffe_dentro_le_stringhe_non_contano(self):
        assert not _sembra_troncato('{"a": "una { graffa nel testo"}')


class TestSoglie:
    def test_i_duplicati_dei_passaggi_hanno_soglia_piu_alta_delle_memorie(self):
        """Su testi lunghi la somiglianza coseno è naturalmente più alta: due
        paragrafi diversi dello stesso autore sullo stesso tema arrivano a
        0,90 senza essere duplicati."""
        from platform_core.memory.consolidation import (
            SOGLIA_DUPLICATO as SOGLIA_MEMORIE,
        )

        assert SOGLIA_DUPLICATO > SOGLIA_MEMORIE
