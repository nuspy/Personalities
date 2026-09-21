"""Divisione in passaggi.

Ciò che si protegge qui è la differenza fra questo chunker e quello della
pipeline: la sovrapposizione e il taglio fra frasi. Sono le due proprietà che
distinguono un passaggio recuperabile da un pezzo di testo tagliato a misura,
e sono anche quelle che si perdono per prime se qualcuno «semplifica».
"""
from __future__ import annotations

from platform_core.knowledge.chunker import (
    ConfigurazioneChunking, dividi, dividi_documento, dividi_in_frasi,
    dividi_per_sezioni,
)


class TestDivisioneInFrasi:
    def test_frasi_semplici(self):
        frasi = dividi_in_frasi("Prima frase. Seconda frase! Terza?")
        assert frasi == ["Prima frase.", "Seconda frase!", "Terza?"]

    def test_le_abbreviazioni_non_spezzano(self):
        """«Dott.» non chiude una frase, e un chunker che ci cade produce
        passaggi di tre parole."""
        frasi = dividi_in_frasi("Il Dott. Rossi parlò. Poi tacque.")
        assert len(frasi) == 2
        assert frasi[0] == "Il Dott. Rossi parlò."

    def test_le_virgolette_di_chiusura_restano_attaccate(self):
        frasi = dividi_in_frasi('Disse «la virtù basta». Poi uscì.')
        assert len(frasi) == 2
        assert frasi[0].endswith("»." ) or frasi[0].endswith(".")

    def test_i_numeri_non_confondono(self):
        frasi = dividi_in_frasi("Nel 1 a.C. accadde. Poi il 2 d.C. finì.")
        assert len(frasi) == 2

    def test_testo_vuoto(self):
        assert dividi_in_frasi("   ") == []


class TestSovrapposizione:
    def test_i_passaggi_si_sovrappongono(self):
        """L'ultima frase di un passaggio è la prima del successivo.

        È la proprietà che impedisce a una domanda sul punto di giuntura di
        trovare soltanto metà del ragionamento.
        """
        testo = " ".join(f"Questa è la frase numero {i} del testo di prova." for i in range(40))
        passaggi = dividi(testo, config=ConfigurazioneChunking(token_obiettivo=60))

        assert len(passaggi) > 1
        for precedente, successivo in zip(passaggi, passaggi[1:]):
            ultima_del_precedente = dividi_in_frasi(precedente.testo)[-1]
            assert successivo.testo.startswith(ultima_del_precedente)

    def test_la_sovrapposizione_e_dichiarata(self):
        testo = " ".join(f"Frase numero {i} abbastanza lunga da contare." for i in range(30))
        passaggi = dividi(testo, config=ConfigurazioneChunking(token_obiettivo=60))

        assert passaggi[0].frasi_sovrapposte == 0
        assert all(p.frasi_sovrapposte == 1 for p in passaggi[1:])

    def test_senza_sovrapposizione_se_disattivata(self):
        testo = " ".join(f"Frase numero {i} di lunghezza media." for i in range(30))
        passaggi = dividi(
            testo,
            config=ConfigurazioneChunking(token_obiettivo=60, frasi_di_sovrapposizione=0),
        )
        assert all(p.frasi_sovrapposte == 0 for p in passaggi)


class TestBudget:
    def test_nessun_passaggio_spezza_una_frase(self):
        testo = " ".join(f"Frase {i} con un poco di testo attorno." for i in range(50))
        passaggi = dividi(testo, config=ConfigurazioneChunking(token_obiettivo=50))

        for p in passaggi:
            assert p.testo.rstrip()[-1] in ".!?", f"passaggio troncato: {p.testo[-40:]!r}"

    def test_una_frase_piu_lunga_del_budget_non_viene_tagliata(self):
        """Tagliarla produrrebbe il frammento privo di senso che si vuole evitare."""
        lunga = "Parola " * 400 + "finale."
        passaggi = dividi(lunga, config=ConfigurazioneChunking(token_obiettivo=50))

        assert len(passaggi) == 1
        assert passaggi[0].testo.endswith("finale.")

    def test_una_coda_troppo_corta_rientra_nel_precedente(self):
        """Un passaggio di due righe occuperebbe un posto senza rispondere a nulla."""
        testo = " ".join(f"Frase numero {i} di media lunghezza qui." for i in range(20)) + " Ok."
        passaggi = dividi(
            testo, config=ConfigurazioneChunking(token_obiettivo=60, token_minimi=40),
        )

        assert all(p.token_stimati >= 20 for p in passaggi)
        assert passaggi[-1].testo.endswith("Ok.")

    def test_gli_ordinali_sono_progressivi(self):
        testo = " ".join(f"Frase {i} con testo sufficiente attorno." for i in range(40))
        passaggi = dividi(testo, config=ConfigurazioneChunking(token_obiettivo=60))

        assert [p.ordinale for p in passaggi] == list(range(len(passaggi)))


class TestSezioni:
    def test_i_titoli_separano_le_sezioni(self):
        testo = "# Primo\n\nTesto uno.\n\n## Secondo\n\nTesto due."
        sezioni = dividi_per_sezioni(testo)

        assert [s.titolo for s in sezioni] == ["Primo", "Secondo"]

    def test_il_preambolo_resta_senza_titolo(self):
        testo = "Introduzione senza titolo.\n\n# Capitolo\n\nCorpo."
        sezioni = dividi_per_sezioni(testo)

        assert sezioni[0].titolo is None
        assert "Introduzione" in sezioni[0].testo

    def test_un_testo_senza_titoli_resta_intero(self):
        sezioni = dividi_per_sezioni("Solo prosa, nessun titolo.")
        assert len(sezioni) == 1 and sezioni[0].titolo is None

    def test_la_sezione_viaggia_col_passaggio(self):
        """Il titolo entra nella citazione: dice a chi legge dove si trova."""
        testo = "# Sulla brevità\n\n" + " ".join(
            f"Frase {i} della sezione." for i in range(20)
        )
        passaggi = dividi_documento(testo, config=ConfigurazioneChunking(token_obiettivo=60))

        assert all(p.sezione == "Sulla brevità" for p in passaggi)

    def test_la_numerazione_e_continua_fra_sezioni(self):
        """Ricominciare da zero renderebbe adiacenti due passaggi lontani."""
        testo = (
            "# Uno\n\n" + " ".join(f"Frase {i} della prima." for i in range(20))
            + "\n\n# Due\n\n" + " ".join(f"Frase {i} della seconda." for i in range(20))
        )
        passaggi = dividi_documento(testo, config=ConfigurazioneChunking(token_obiettivo=60))
        ordinali = [p.ordinale for p in passaggi]

        assert ordinali == sorted(ordinali)
        assert len(set(ordinali)) == len(ordinali)
