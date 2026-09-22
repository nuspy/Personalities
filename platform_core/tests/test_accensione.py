"""Accensione a richiesta del motore locale.

Tre proprietà valgono più delle altre, e i test che le proteggono sono
scritti per fallire se qualcuno le indebolisce:

- **non si spegne ciò che si sta usando.** Una digestione di tre ore uccisa
  da un timer di inattività perderebbe tre ore di analisi, e il difetto
  comparirebbe solo sui corpora grandi — cioè in produzione;
- **non si spegne ciò che non si è acceso.** Il modello può essere acceso da
  qualcun altro su quella macchina: spegnerlo perché *noi* abbiamo finito
  significa interrompere il lavoro di un estraneo;
- **il comando non passa mai da una shell.** Viene dalla configurazione, non
  dai dati, ma un giorno qualcuno vorrà renderlo modificabile dalla console:
  che l'esecuzione sia già senza shell rende quel giorno meno pericoloso.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from platform_core.capabilities.registry import InMemoryStore
from platform_core.llm.accensione import (
    CHIAVE_RICHIESTA, CHIAVE_STATO, MotoreLocale, MotoreNonDisponibile,
    chiedi_accensione, stato_pubblicato,
)


class Comandi:
    """Esecutore finto: annota cosa è stato lanciato, non lancia nulla."""

    def __init__(self, codice: int | None = 0, output: str = "") -> None:
        self.lanciati: list[str] = []
        self._codice = codice
        self._output = output

    async def __call__(self, comando: str, attesa: float):
        self.lanciati.append(comando)
        return self._codice, self._output


class Sonda:
    """Salute finta: risponde no per `muta` giri, poi sì."""

    def __init__(self, muta: int = 0, sempre_no: bool = False) -> None:
        self._muta = muta
        self._sempre_no = sempre_no
        self.chiamate = 0

    async def __call__(self) -> bool:
        self.chiamate += 1
        if self._sempre_no:
            return False
        if self._muta > 0:
            self._muta -= 1
            return False
        return True


class Orologio:
    """Tempo finto: avanza quando glielo si dice."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def motore(**extra) -> tuple[MotoreLocale, Comandi, Sonda]:
    comandi = extra.pop("comandi", None) or Comandi()
    sonda = extra.pop("sonda", None) or Sonda()
    impostazioni = {
        "avvio": "avvia-il-motore",
        "arresto": "spegni-il-motore",
        "attesa_s": 30.0,
        "inattivita_s": 600.0,
        "esecutore": comandi,
        "sonda": sonda,
    }
    impostazioni.update(extra)
    return MotoreLocale(**impostazioni), comandi, sonda


class TestSenzaConfigurazione:
    """Senza comandi il worker si comporta esattamente come prima."""

    async def test_non_gestito_e_non_esegue_niente(self):
        comandi = Comandi()
        m = MotoreLocale(esecutore=comandi)

        assert not m.gestito
        async with m.in_uso():
            pass

        assert comandi.lanciati == []
        assert m.stato().stato == "non_gestito"
        assert m.stato().motivo, "deve dire perché non gestisce"

    async def test_lo_stato_dichiara_zero_inattivita(self):
        """Un tempo di spegnimento mostrato in console per un motore che
        nessuno spegne sarebbe una promessa falsa."""
        m = MotoreLocale(inattivita_s=600.0)
        assert m.stato().inattivita_s == 0.0


class TestAccensione:
    async def test_accende_e_attende_che_risponda(self):
        m, comandi, sonda = motore(sonda=Sonda(muta=1))

        async with m.in_uso(motivo="digestione"):
            assert m.stato().stato == "acceso"

        assert comandi.lanciati == ["avvia-il-motore"]
        assert sonda.chiamate >= 2, "ha smesso di chiedere prima di una risposta"

    async def test_se_risponde_gia_non_lo_riaccende(self):
        """Il caso comune: il modello è acceso da prima. Rilanciare l'avvio
        significherebbe, a seconda del launcher, un secondo processo sulla
        stessa GPU."""
        m, comandi, _ = motore()

        async with m.in_uso():
            pass

        assert comandi.lanciati == []
        assert not m.acceso_da_noi

    async def test_due_lavori_insieme_accendono_una_volta_sola(self):
        m, comandi, _ = motore(sonda=Sonda(muta=1))

        async def lavoro():
            async with m.in_uso():
                await asyncio.sleep(0)

        await asyncio.gather(lavoro(), lavoro(), lavoro())

        assert comandi.lanciati == ["avvia-il-motore"]

    async def test_se_non_risponde_il_lavoro_sa_perche(self):
        orologio = Orologio()
        m, _, _ = motore(
            sonda=Sonda(sempre_no=True), adesso=orologio,
            comandi=Comandi(codice=1, output="Permission denied (publickey)."),
        )

        async def avanza():
            # Fa scadere l'attesa senza dormire davvero.
            while True:
                await asyncio.sleep(0)
                orologio.t += 10

        corsa = asyncio.ensure_future(avanza())
        with pytest.raises(MotoreNonDisponibile) as errore:
            await m.accendi()
        corsa.cancel()

        assert "publickey" in str(errore.value), (
            "senza l'output del comando chi guarda la console non può sapere "
            "che è un problema di chiave"
        )
        assert m.stato().stato == "guasto"


class TestSpegnimento:
    async def test_non_spegne_mentre_un_lavoro_lo_usa(self):
        """La proprietà che protegge una digestione di tre ore."""
        orologio = Orologio()
        m, comandi, _ = motore(adesso=orologio, inattivita_s=60.0)

        async with m.in_uso():
            orologio.t += 10_000          # ampiamente oltre l'inattività
            await m._spegni_se_inattivo()
            assert comandi.lanciati == [], "ha spento un motore in uso"

    async def test_spegne_dopo_l_inattivita(self):
        orologio = Orologio()
        m, comandi, _ = motore(
            adesso=orologio, inattivita_s=60.0, sonda=Sonda(muta=1),
        )

        async with m.in_uso():
            pass
        orologio.t += 61
        await m._spegni_se_inattivo()

        assert comandi.lanciati == ["avvia-il-motore", "spegni-il-motore"]
        assert m.stato().stato == "spento"

    async def test_non_spegne_prima_dell_inattivita(self):
        orologio = Orologio()
        m, comandi, _ = motore(
            adesso=orologio, inattivita_s=600.0, sonda=Sonda(muta=1),
        )

        async with m.in_uso():
            pass
        orologio.t += 599
        await m._spegni_se_inattivo()

        assert comandi.lanciati == ["avvia-il-motore"]

    async def test_non_spegne_cio_che_non_ha_acceso(self):
        """Il modello era già acceso e nessun lavoro l'ha usato: spegnerlo
        significherebbe interrompere il lavoro di qualcun altro su quella
        macchina."""
        orologio = Orologio()
        m, comandi, _ = motore(adesso=orologio, inattivita_s=60.0)
        m._stato = "acceso"

        orologio.t += 10_000
        await m._spegni_se_inattivo()

        assert comandi.lanciati == []

    async def test_a_mano_non_spegne_un_lavoro_in_corso_ma_forzato_si(self):
        m, comandi, _ = motore()

        async with m.in_uso():
            assert await m.spegni() is False
            assert await m.spegni(forza=True) is True

        assert comandi.lanciati == ["spegni-il-motore"]


class TestStatoCondiviso:
    async def test_lo_stato_si_pubblica_con_scadenza(self):
        store = InMemoryStore()
        m, _, _ = motore(store=store, worker_id="w-1", sonda=Sonda(muta=1))

        async with m.in_uso():
            pass

        pubblicato = stato_pubblicato(store)
        assert pubblicato.stato == "acceso"
        assert pubblicato.worker_id == "w-1"
        assert pubblicato.inattivita_s == 600.0

    async def test_lo_stato_si_scrive_sempre_con_una_scadenza(self):
        """Senza scadenza, un worker morto lascerebbe in console un «acceso»
        che nessuno smentisce mai: un ricordo, non un'informazione."""
        scadenze: list = []

        class StoreSpia(InMemoryStore):
            def set(self, name, value, ex=None):
                scadenze.append(ex)
                return super().set(name, value, ex=ex)

        m, _, _ = motore(store=StoreSpia(), sonda=Sonda(muta=1))
        async with m.in_uso():
            pass

        assert scadenze and all(e for e in scadenze), (
            "uno stato pubblicato senza scadenza sopravvive al worker"
        )

    async def test_senza_worker_vivo_la_console_non_dice_acceso(self):
        store = InMemoryStore()   # la chiave è scaduta: non c'è più

        assert stato_pubblicato(store).stato == "non_gestito"
        assert "non risponde" in stato_pubblicato(store).motivo

    async def test_stato_illeggibile_non_fa_esplodere_la_console(self):
        store = InMemoryStore()
        store.set(CHIAVE_STATO, "{non è json", ex=60)

        assert stato_pubblicato(store).stato == "non_gestito"


class TestRichiesteDallaConsole:
    async def test_la_richiesta_viene_raccolta_ed_eseguita(self):
        store = InMemoryStore()
        m, comandi, _ = motore(store=store, sonda=Sonda(muta=1))

        chiedi_accensione(store, "accendi")
        await m._raccogli_richiesta()

        assert comandi.lanciati == ["avvia-il-motore"]
        assert store.get(CHIAVE_RICHIESTA) is None, (
            "una richiesta raccolta e non consumata verrebbe eseguita a ogni giro"
        )

    async def test_solo_accendi_e_spegni(self):
        """La console chiede un'azione, non *cosa* eseguire: è la riga che
        separa un pulsante da un'esecuzione di comandi arbitrari."""
        store = InMemoryStore()
        with pytest.raises(ValueError):
            chiedi_accensione(store, "rm -rf /")

        assert store.get(CHIAVE_RICHIESTA) is None

    async def test_una_richiesta_sconosciuta_viene_ignorata(self):
        store = InMemoryStore()
        store.set(CHIAVE_RICHIESTA, "riavvia-tutto", ex=60)
        m, comandi, _ = motore(store=store)

        await m._raccogli_richiesta()

        assert comandi.lanciati == []


class TestEsecuzioneDelComando:
    """Il comando vero, senza shell.

    Sincroni e su un ciclo scelto a mano: su Windows i sottoprocessi di
    asyncio vogliono il Proactor — quello che `asyncio.run` sceglie da sé in
    un processo normale — mentre sotto pytest il ciclo è un Selector, dove
    `create_subprocess_exec` non è implementato. Senza, qui fallirebbe il
    test e non il codice.
    """

    @staticmethod
    def _corri(coro):
        ciclo = (
            asyncio.ProactorEventLoop() if sys.platform == "win32"
            else asyncio.new_event_loop()
        )
        try:
            return ciclo.run_until_complete(coro)
        finally:
            ciclo.close()

    def test_niente_shell(self):
        """Con una shell di mezzo, `&&` e `;` nel comando eseguirebbero un
        secondo programma. Senza, l'intero testo resta un argomento."""
        from platform_core.llm.accensione import _esegui

        codice, output = self._corri(_esegui(
            f'{sys.executable} -c "import sys;print(sys.argv[1])" "uno && due"',
            attesa=30.0,
        ))

        assert codice == 0
        assert output.strip() == "uno && due"

    def test_i_percorsi_di_windows_restano_interi(self):
        """`shlex` in modo POSIX mangerebbe le barre rovesciate, e il comando
        cercherebbe `C:Projectsbonsaistart.ps1`: un errore che non somiglia
        alla sua causa."""
        from platform_core.llm.accensione import _argomenti

        pezzi = _argomenti(r'C:\Programmivvia.exe --file C:\dati\modello.gguf')

        if sys.platform == "win32":
            assert pezzi[0] == r"C:\Programmivvia.exe"
            assert pezzi[-1] == r"C:\dati\modello.gguf"
        else:
            assert pezzi[0].endswith("avvia.exe")

    def test_un_comando_che_non_ritorna_non_blocca(self):
        """Lo script che avvia il server resta in primo piano: aspettarne la
        fine significherebbe non accendere mai."""
        from platform_core.llm.accensione import _esegui

        codice, _ = self._corri(_esegui(
            f'{sys.executable} -c "import time;time.sleep(30)"', attesa=0.3,
        ))

        assert codice is None, "avrebbe aspettato la fine del processo"

    def test_un_comando_vuoto_e_un_errore_dichiarato(self):
        from platform_core.llm.accensione import _esegui

        with pytest.raises(MotoreNonDisponibile):
            self._corri(_esegui("   ", attesa=1.0))

    def test_il_ripiego_a_thread_esegue_lo_stesso(self):
        """Il ripiego non è per una nicchia: su Windows il backend sceglie il
        ciclo Selector perché psycopg 3 non sa usare l'altro, e lì i
        sottoprocessi asincroni non esistono — cioè proprio sulla macchina di
        sviluppo dove il modello e il worker stanno insieme."""
        from platform_core.llm.accensione import _esegui_in_thread

        codice, output = _esegui_in_thread(
            [sys.executable, "-c", "print('pronto')"], 30.0,
        )

        assert codice == 0
        assert output.strip() == "pronto"

    def test_il_ripiego_non_aspetta_chi_non_ritorna(self):
        from platform_core.llm.accensione import _esegui_in_thread

        codice, _ = _esegui_in_thread(
            [sys.executable, "-c", "import time;time.sleep(30)"], 0.3,
        )

        assert codice is None

    @pytest.mark.skipif(sys.platform != "win32", reason="il caso è di Windows")
    def test_su_un_ciclo_selector_ripiega_invece_di_fallire(self):
        """È il ciclo che il worker ha davvero su questa macchina."""
        from platform_core.llm.accensione import _esegui

        ciclo = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
        try:
            codice, output = ciclo.run_until_complete(_esegui(
                f'{sys.executable} -c "print(\'pronto\')"', attesa=30.0,
            ))
        finally:
            ciclo.close()

        assert codice == 0
        assert output.strip() == "pronto"
