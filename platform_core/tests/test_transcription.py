"""Trascrizione di audio e video.

Il gruppo che conta è `TestLimiteDelleDifese`, e protegge un'ammissione
invece di una garanzia: **nessuna difesa automatica riconosce una
trascrizione sbagliata ma plausibile**. Misurato su una registrazione
difficile, Whisper ha prodotto «La vita non è brieva, ma non è la rendima
tale» con fiducia 0,60 — nella norma — e il classificatore della digestione
le ha dato qualità 0,95 inventando un significato per quelle parole.

I test qui sotto verificano che il codice dica la verità su cosa copre: il
silenzio trascritto come parole, sì; un errore verosimile, no. Se un giorno
qualcuno trasformasse `fiducia_media` in un via libera automatico, cadrebbero.
"""
from __future__ import annotations

import pathlib

import pytest

from platform_core.knowledge.transcription import (
    ESTENSIONI_AUDIO, ESTENSIONI_VIDEO, MODELLO_PREDEFINITO, SOGLIA_ATTENZIONE,
    SOGLIA_FIDUCIA, Segmento, Trascrizione, _fiducia, e_multimediale,
)


class Finto:
    """Un segmento come lo restituisce faster-whisper."""

    def __init__(self, avg_logprob=None, no_speech_prob=0.0):
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class TestRiconoscimento:
    def test_audio_e_video_passano_di_qui(self):
        assert e_multimediale(pathlib.Path("intervista.mp3"))
        assert e_multimediale(pathlib.Path("conferenza.mp4"))
        assert e_multimediale(pathlib.Path("REGISTRAZIONE.WAV"))

    def test_i_documenti_no(self):
        assert not e_multimediale(pathlib.Path("lettera.pdf"))
        assert not e_multimediale(pathlib.Path("appunti.txt"))

    def test_le_due_famiglie_non_si_sovrappongono(self):
        assert not (ESTENSIONI_AUDIO & ESTENSIONI_VIDEO)


class TestTimestamp:
    def test_la_marca_sotto_l_ora(self):
        assert Segmento(0, 4, "x").marca == "[0:00]"
        assert Segmento(125, 130, "x").marca == "[2:05]"

    def test_la_marca_sopra_l_ora(self):
        """Una citazione da una registrazione lunga deve poter dire dove."""
        assert Segmento(3725, 3730, "x").marca == "[1:02:05]"

    def test_la_marca_entra_nel_testo(self):
        """Nel testo e non solo nei metadati: è l'unico modo perché
        sopravviva al chunking, che dei metadati non sa nulla."""
        s = Segmento(65, 70, "una frase")
        assert s.marca == "[1:05]"


class TestFiducia:
    def test_senza_logprob_si_assume_buona(self):
        assert _fiducia(Finto()) == 1.0

    def test_il_logprob_diventa_probabilita(self):
        import math

        assert _fiducia(Finto(avg_logprob=math.log(0.8))) == pytest.approx(0.8)

    def test_il_silenzio_azzera_la_fiducia(self):
        """`no_speech_prob` alta significa che lì non parlava nessuno: il
        testo prodotto è rumore interpretato come parole. È il caso che le
        difese coprono davvero."""
        assert _fiducia(Finto(avg_logprob=-0.1, no_speech_prob=0.9)) == 0.0

    def test_un_segmento_debole_non_e_affidabile(self):
        assert not Segmento(0, 1, "x", fiducia=0.2).affidabile
        assert Segmento(0, 1, "x", fiducia=0.9).affidabile


class TestLimiteDelleDifese:
    """Ciò che il codice ammette di non saper fare."""

    def test_la_fiducia_media_di_una_trascrizione_sbagliata_e_normale(self):
        """Il caso misurato: «La vita non è brieva, ma non è la rendima tale».

        Ogni segmento aveva 0,60 — sopra la soglia di scarto. Se qualcuno
        alzasse `SOGLIA_FIDUCIA` fino a prendere questo caso, scarterebbe
        anche le trascrizioni buone: il valore non separa le due cose.
        """
        sbagliata = Trascrizione(segmenti=[
            Segmento(0, 4, "La vita non è brieva, ma non è la rendima tale.", 0.602),
            Segmento(4, 9, "Non è biomo poco tempo, è ne per demomolto.", 0.602),
        ])

        assert sbagliata.fiducia_media > SOGLIA_FIDUCIA, (
            "queste parole non esistono, eppure la fiducia le promuove"
        )

    def test_ma_la_segnala_come_da_riascoltare(self):
        """L'unica cosa utile che la fiducia sa fare: mettere in coda i file
        da ascoltare per primi."""
        sbagliata = Trascrizione(segmenti=[
            Segmento(0, 4, "La vita non è brieva", 0.602),
        ])
        assert sbagliata.da_riascoltare

    def test_una_trascrizione_sicura_non_si_segnala(self):
        buona = Trascrizione(segmenti=[
            Segmento(0, 4, "La vita non è breve, ma noi la rendiamo tale.", 0.95),
        ])
        assert not buona.da_riascoltare

    def test_la_soglia_di_attenzione_non_scarta(self):
        """Segnala e basta: buttare via un'ora di parlato per un valore medio
        significherebbe perdere il buono insieme al disturbato."""
        assert SOGLIA_ATTENZIONE > SOGLIA_FIDUCIA

    def test_una_trascrizione_vuota_non_ha_fiducia(self):
        assert Trascrizione().fiducia_media == 0.0


class TestMetadati:
    def test_il_registro_parlato_e_sempre_presente(self):
        """È il campo che impedisce di addestrare un modello a scrivere come
        si parla: senza, le trascrizioni entrano nel dataset indistinguibili
        dai testi scritti."""
        assert Trascrizione().meta()["registro"] == "parlato"

    def test_i_metadati_portano_il_modello_usato(self):
        """`small` e `large-v3` sbagliano in modo molto diverso: sapere quale
        ha prodotto un testo è la prima cosa da guardare quando quel testo
        sembra strano."""
        t = Trascrizione(modello="large-v3")
        assert t.meta()["trascritto_con"] == "large-v3"

    def test_i_segmenti_scartati_si_contano(self):
        t = Trascrizione(scartati=3)
        assert t.meta()["segmenti_scartati"] == 3

    def test_l_avviso_arriva_nei_metadati(self):
        t = Trascrizione(segmenti=[Segmento(0, 1, "x", 0.4)])
        assert t.meta()["da_riascoltare"] is True


class TestConfigurazione:
    def test_il_modello_predefinito_e_il_piu_grande(self):
        """La trascrizione si fa una volta sola: risparmiare qui significa
        portarsi gli errori dentro il corpus per sempre."""
        assert MODELLO_PREDEFINITO == "large-v3"
