"""Gli avatar: validazione e descrittori.

La validazione è il punto. Un avatar configurato male e accettato produce, a
distanza di giorni, un riquadro vuoto in una pagina e un errore nella console
del browser di un utente — cioè nessuna informazione per chi l'ha configurato,
e nessun modo di collegarla alla modifica che l'ha causata. Qui si verifica
che il rifiuto arrivi al salvataggio e dica cosa manca.
"""
from __future__ import annotations

import pytest

from platform_core.avatar.engine import (
    ConfigurazioneAvatarNonValida, ImmagineFerma, Modello3D, VideoInCiclo,
    descrivi, motore_per, valida,
)
from platform_core.voice.visemi import VISEMI


class TestImmagine:
    def test_serve_un_indirizzo(self):
        with pytest.raises(ConfigurazioneAvatarNonValida, match="uri"):
            valida("immagine", {})

    def test_un_ritratto_non_ha_labiale_e_va_bene(self):
        """Molte voci storiche hanno un solo ritratto: pretendere un modello
        3D per poter parlare significherebbe che la voce funziona solo per chi
        ha commissionato una testa."""
        d = descrivi("immagine", {"uri": "https://esempio/seneca.jpg"})

        assert d.labiale is False
        assert d.uri.endswith("seneca.jpg")

    def test_l_onda_sonora_e_accesa_di_suo(self):
        """Dice «sta parlando» senza fingere un labiale che un'immagine ferma
        non può avere."""
        d = descrivi("immagine", {"uri": "x.jpg"})

        assert d.extra["onda"] is True


class TestVideo:
    def test_serve_almeno_la_clip_di_riposo(self):
        """È quella che si vede quando non succede nulla, cioè quasi sempre."""
        with pytest.raises(ConfigurazioneAvatarNonValida, match="fermo"):
            valida("video", {"clip": {"parlante": "p.mp4"}})

    def test_uno_stato_sconosciuto_viene_rifiutato_con_l_elenco(self):
        """Un refuso in uno stato darebbe una clip che non viene mai
        riprodotta, e nessun errore da nessuna parte."""
        with pytest.raises(ConfigurazioneAvatarNonValida) as exc:
            valida("video", {"clip": {"fermo": "f.mp4", "parlnte": "p.mp4"}})

        assert "parlnte" in str(exc.value)
        assert "parlante" in str(exc.value)

    def test_il_descrittore_porta_tutte_le_clip(self):
        d = descrivi("video", {
            "clip": {"fermo": "f.mp4", "parlante": "p.mp4"},
        })

        assert d.uri == "f.mp4"
        assert d.extra["clip"]["parlante"] == "p.mp4"
        assert d.labiale is False

    def test_la_dissolvenza_ha_un_valore_di_suo(self):
        """Un taglio netto fra le clip si vede come uno scatto proprio nel
        momento in cui l'attenzione è sul volto."""
        d = descrivi("video", {"clip": {"fermo": "f.mp4"}})

        assert d.extra["dissolvenza"] > 0


class TestModello:
    def _pose_complete(self):
        return {forma: f"viseme_{forma}" for forma in VISEMI}

    def test_senza_pose_non_si_puo_animare(self):
        """Il client avrebbe i tempi e non saprebbe quale forma muovere."""
        with pytest.raises(ConfigurazioneAvatarNonValida, match="pose"):
            valida("modello", {"uri": "testa.glb"})

    def test_la_posa_a_riposo_e_obbligatoria(self):
        """Senza, una pausa lascerebbe il volto congelato nell'ultima forma
        pronunciata."""
        pose = self._pose_complete()
        del pose["X"]

        with pytest.raises(ConfigurazioneAvatarNonValida, match="riposo"):
            valida("modello", {"uri": "testa.glb", "pose": pose})

    def test_un_visema_inventato_viene_rifiutato(self):
        """I nomi variano fra strumenti: indovinarli produce un volto immobile
        senza alcun errore da nessuna parte."""
        with pytest.raises(ConfigurazioneAvatarNonValida) as exc:
            valida("modello", {
                "uri": "t.glb", "pose": {"X": "rest", "ZZ": "boh"},
            })

        assert "ZZ" in str(exc.value)

    def test_un_modello_completo_ha_il_labiale(self):
        d = descrivi("modello", {
            "uri": "testa.glb", "pose": self._pose_complete(),
        })

        assert d.labiale is True
        assert d.pose["MBP"] == "viseme_MBP"
        assert d.extra["pose_mancanti"] == []

    def test_le_pose_mancanti_si_dichiarano_invece_di_rifiutare(self):
        """Dieci pose su quindici animano peggio, non male: chi ha configurato
        l'avatar deve sapere quali mancano invece di scoprirlo guardando."""
        d = descrivi("modello", {
            "uri": "t.glb", "pose": {"X": "rest", "A": "aa", "MBP": "pp"},
        })

        assert d.labiale is True
        assert "FV" in d.extra["pose_mancanti"]


class TestTipi:
    def test_un_tipo_sconosciuto_elenca_quelli_previsti(self):
        with pytest.raises(ConfigurazioneAvatarNonValida) as exc:
            motore_per("ologramma")

        assert "immagine" in str(exc.value)

    @pytest.mark.parametrize(
        "motore", [ImmagineFerma(), VideoInCiclo(), Modello3D()],
    )
    def test_ogni_motore_dichiara_il_proprio_tipo(self, motore):
        assert motore_per(motore.tipo) is not None

    def test_solo_il_modello_dichiara_il_labiale(self):
        """Il client lo guarda prima di chiedere la sintesi con i tempi:
        misurarli costa una trascrizione dell'audio, e per un ritratto fermo è
        lavoro buttato."""
        assert descrivi("immagine", {"uri": "x.jpg"}).labiale is False
        assert descrivi("video", {"clip": {"fermo": "f.mp4"}}).labiale is False
        assert descrivi(
            "modello", {"uri": "t.glb", "pose": {"X": "rest"}},
        ).labiale is True
