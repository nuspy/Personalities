"""Percorsi canonici del pacchetto.

Esistono per una ragione imparata: i percorsi relativi alla directory di
lavoro funzionano finché si lancia tutto dalla radice del progetto, e
smettono nel momento in cui qualcosa parte da altrove — un worker avviato da
systemd, un test eseguito da una sottocartella, un container con `WORKDIR`
diverso. Il guasto non è un errore chiaro: è un file di configurazione che
«non c'è», e quindi un comportamento predefinito applicato in silenzio.

Qui i percorsi si ricavano dalla posizione di questo file, che è l'unica cosa
che non dipende da chi ha avviato il processo.
"""
from __future__ import annotations

import os
import pathlib

#: La cartella del pacchetto `platform_core`.
PACKAGE_DIR = pathlib.Path(__file__).resolve().parent

#: La radice del progetto.
PROJECT_ROOT = PACKAGE_DIR.parent

#: Dati di configurazione versionati insieme al codice: guardrail, profili,
#: personalità di esempio. Sovrascrivibile per il deploy, dove possono arrivare
#: da un volume montato.
DATA_DIR = pathlib.Path(
    os.getenv("PERSONA_DATA_DIR", str(PROJECT_ROOT / "data"))
).resolve()
