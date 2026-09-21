"""Il backend della piattaforma: API, worker e dominio condiviso.

Qui dentro c'è una sola riga di comportamento, e riguarda Windows.

**Perché.** Dal 3.8 Python usa su Windows il `ProactorEventLoop`, che psycopg 3
non sa usare in modalità asincrona: ogni connessione al database fallisce con
`InterfaceError`, e l'errore arriva al primo accesso ai dati — non all'avvio,
quando sarebbe ovvio a cosa attribuirlo. La policy va scelta **prima** che
asyncio crei il primo loop, quindi prima che uvicorn parta: l'import del
package è l'unico punto sicuramente precedente a tutto.

Il ripiego costa qualcosa — il selettore regge meno socket e non avvia
sottoprocessi asincroni — ma sono limiti che si incontrano in produzione, e la
produzione gira su Linux, dove questa riga non fa nulla. Se qualcuno ha già
scelto una policy, non gliela si cambia sotto i piedi.
"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    _politica = asyncio.get_event_loop_policy()
    if isinstance(_politica, asyncio.WindowsProactorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
