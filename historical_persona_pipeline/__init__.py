"""Historical Persona Pipeline.

Costruisce la personalita' di un personaggio — vocabolario, ragionamento,
espressioni, modo di parlare, valori — da documenti e fonti online.
"""
from __future__ import annotations

__version__ = "0.2.0"

# La verifica TLS viene agganciata all'archivio certificati del sistema
# all'importazione del package, non nei singoli moduli.
#
# Il motivo e' che il problema non appartiene a un modulo ma alla macchina:
# dove un proxy aziendale o un antivirus rifirma il traffico HTTPS, il bundle
# di `certifi` non contiene quella CA e *qualunque* richiesta fallisce con
# CERTIFICATE_VERIFY_FAILED — comprese quelle fatte da librerie di terze parti
# come `huggingface_hub`, che scarica i modelli e che non possiamo istruire
# modulo per modulo.
#
# L'alternativa diffusa (`verify=False`, o HF_HUB_DISABLE_SSL_VERIFICATION)
# disattiva proprio il controllo che protegge la connessione: qui invece la
# verifica resta piena, cambia solo l'archivio da cui si leggono le CA.
from .pipeline.utils.tls import enable_system_certificates as _enable_system_certificates

_enable_system_certificates()

# ATTENZIONE ALL'ORDINE DEGLI IMPORT, se si usa il package come libreria.
#
# L'iniezione sopra sostituisce la classe di contesto TLS di `ssl`, ma non
# tocca le sessioni HTTPS gia' costruite. `huggingface_hub` ne crea una al
# proprio import: se una libreria che lo tira dentro — Unsloth, transformers —
# viene importata PRIMA di questo package, quella sessione nasce con i
# certificati di `certifi` e i download dei modelli falliscono con
# CERTIFICATE_VERIFY_FAILED, anche se ogni altra richiesta funziona.
#
#     import historical_persona_pipeline   # per primo
#     import unsloth                       # poi, e prima di transformers
#
# Importando il package dai suoi moduli (`from historical_persona_pipeline...`)
# l'ordine e' garantito: questo file viene eseguito per primo comunque.
