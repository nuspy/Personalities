"""Rigenera `realm-personalities.prod.json` da quello di sviluppo.

    docker run -d --name kc-genera -p 8181:8080 -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \\
        -e KC_BOOTSTRAP_ADMIN_PASSWORD=genera -e KC_DB=dev-mem \\
        quay.io/keycloak/keycloak:26.0 start-dev
    python deploy/keycloak/strumenti/genera_realm_prod.py --keycloak http://localhost:8181 --password genera
    docker rm -f kc-genera

**Perché su un Keycloak vero e non scritto a mano.** Se il realm dichiara dei
flussi di autenticazione, Keycloak non crea più quelli predefiniti: un file
con il solo flusso 2FA importerebbe un realm senza login diretto, senza
registrazione, senza recupero della password. Qui i flussi si costruiscono
sull'istanza — copiando quello del browser, come si farebbe dalla console — e
poi si esporta il realm intero, che è completo per costruzione.

L'esportazione parziale maschera i segreti con `**********` e non contiene
chiavi private; lo script li sostituisce con segnaposto `${KC_...}`, che
Keycloak risolve dall'ambiente all'import. Un valore mascherato rimasto nel
file diventerebbe il segreto vero: lo script si rifiuta di scriverlo.

Un'istanza usa-e-getta, con il database in memoria: il realm di sviluppo non
viene toccato.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from urllib.parse import quote

import httpx

QUI = pathlib.Path(__file__).resolve().parent
SVILUPPO = QUI.parent / "realm-personalities.json"
PRODUZIONE = QUI.parent / "realm-personalities.prod.json"
REALM = "personalities"

BROWSER = "browser con 2FA amministratori"
DOPO_ESTERNO = "2FA amministratori dopo accesso esterno"
MASCHERA = "**********"


def realm_di_partenza() -> dict:
    """Il realm di sviluppo senza ciò che in produzione non deve esistere."""
    prod = copy.deepcopy(json.loads(SVILUPPO.read_text(encoding="utf-8")))
    prod.pop("users", None)
    prod["clients"] = [c for c in prod["clients"] if c["clientId"] != "persona-test"]
    for c in prod["clients"]:
        for campo in ("redirectUris", "webOrigins"):
            if campo in c:
                c[campo] = [u for u in c[campo] if "localhost" not in u]
    prod.update({
        "verifyEmail": True,
        "sslRequired": "external",
        "bruteForceProtected": True,
        "permanentLockout": False,
        "failureFactor": 8,
        "waitIncrementSeconds": 60,
        "maxFailureWaitSeconds": 900,
        "quickLoginCheckMilliSeconds": 1000,
        "minimumQuickLoginWaitSeconds": 60,
        "maxDeltaTimeSeconds": 43200,
        "passwordPolicy": "length(12) and notUsername and notEmail and passwordHistory(3)",
        "otpPolicyType": "totp",
        "otpPolicyAlgorithm": "HmacSHA1",
        "otpPolicyDigits": 6,
        "otpPolicyPeriod": 30,
        "otpPolicyLookAheadWindow": 1,
        "otpSupportedApplications": [
            "totpAppGoogleName", "totpAppMicrosoftAuthenticatorName", "totpAppFreeOTPName",
        ],
        "smtpServer": {
            "host": "${KC_SMTP_HOST}", "port": "${KC_SMTP_PORT}", "from": "${KC_SMTP_FROM}",
            "fromDisplayName": "AI Personality Platform", "auth": "true",
            "user": "${KC_SMTP_USER}", "password": "${KC_SMTP_PASSWORD}",
            "starttls": "true", "ssl": "false",
        },
        "identityProviders": [
            _provider("google", "Google", "google", {"defaultScope": "openid profile email"}),
            _provider("facebook", "Facebook", "facebook", {"defaultScope": "email public_profile"}),
            # Apple non ha un provider integrato: OIDC generico. Il suo
            # `client_secret` è un JWT firmato con la chiave dell'account
            # sviluppatore, e scade al più ogni sei mesi (vedi il README).
            _provider("apple", "Apple", "oidc", {
                "issuer": "https://appleid.apple.com",
                "authorizationUrl": "https://appleid.apple.com/auth/authorize?response_mode=form_post",
                "tokenUrl": "https://appleid.apple.com/auth/token",
                "jwksUrl": "https://appleid.apple.com/auth/keys",
                "useJwksUrl": "true",
                "validateSignature": "true",
                "clientAuthMethod": "client_secret_post",
                "defaultScope": "openid name email",
            }),
        ],
    })
    return prod


def _provider(alias: str, nome: str, tipo: str, config: dict) -> dict:
    # Spento finché non lo si accende dall'ambiente: all'import il segnaposto
    # diventa `false`, e un provider senza credenziali non compare nella
    # pagina di accesso.
    return {
        "alias": alias, "displayName": nome, "providerId": tipo, "enabled": False,
        "trustEmail": True, "firstBrokerLoginFlowAlias": "first broker login",
        "config": {
            "clientId": f"${{KC_{alias.upper()}_CLIENT_ID}}",
            "clientSecret": f"${{KC_{alias.upper()}_CLIENT_SECRET}}",
            "syncMode": "IMPORT", **config,
        },
    }


class Keycloak:
    def __init__(self, url: str, password: str) -> None:
        risposta = httpx.post(f"{url}/realms/master/protocol/openid-connect/token", data={
            "client_id": "admin-cli", "username": "admin", "password": password,
            "grant_type": "password",
        })
        risposta.raise_for_status()
        self.c = httpx.Client(
            base_url=f"{url}/admin/realms",
            headers={"Authorization": f"Bearer {risposta.json()['access_token']}"},
            timeout=30,
        )

    def ok(self, r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            print(r.request.method, r.request.url, r.status_code, r.text, file=sys.stderr)
            r.raise_for_status()
        return r

    def esecuzioni(self, flusso: str) -> list:
        return self.ok(self.c.get(f"/{REALM}/authentication/flows/{quote(flusso)}/executions")).json()

    def requisito(self, flusso: str, esecuzione: dict, valore: str) -> None:
        self.ok(self.c.put(f"/{REALM}/authentication/flows/{quote(flusso)}/executions",
                           json={**esecuzione, "requirement": valore}))

    def esecuzione(self, flusso: str, provider: str) -> dict:
        self.ok(self.c.post(f"/{REALM}/authentication/flows/{quote(flusso)}/executions/execution",
                            json={"provider": provider}))
        return [e for e in self.esecuzioni(flusso) if e.get("providerId") == provider][-1]

    def sottoflusso(self, flusso: str, alias: str, descrizione: str) -> dict:
        self.ok(self.c.post(f"/{REALM}/authentication/flows/{quote(flusso)}/executions/flow", json={
            "alias": alias, "type": "basic-flow", "provider": "registration-page-form",
            "description": descrizione,
        }))
        return next(e for e in self.esecuzioni(flusso) if e.get("displayName") == alias)

    def condizione_admin(self, flusso: str, alias_config: str, *, nega: bool) -> dict:
        e = self.esecuzione(flusso, "conditional-user-role")
        self.requisito(flusso, e, "REQUIRED")
        self.ok(self.c.post(f"/{REALM}/authentication/executions/{e['id']}/config", json={
            "alias": alias_config,
            "config": {"condUserRole": "admin", "negate": "true" if nega else "false"},
        }))
        return e


def configura(kc: Keycloak, partenza: dict) -> dict:
    if kc.c.get(f"/{REALM}").status_code == 200:
        kc.ok(kc.c.delete(f"/{REALM}"))
    kc.ok(kc.c.post("", json=partenza))

    # Il flusso del browser: copia di quello predefinito.
    kc.ok(kc.c.post(f"/{REALM}/authentication/flows/browser/copy", json={"newName": BROWSER}))
    tutte = kc.esecuzioni(BROWSER)
    moduli = next(e for e in tutte if e.get("authenticationFlow") and e["displayName"].endswith("forms"))
    facoltativo = next(e for e in tutte if e.get("authenticationFlow") and "Conditional OTP" in e["displayName"])

    # L'OTP facoltativo resta a chi l'ha attivato, ma non agli amministratori:
    # per loro c'è il sottoflusso obbligatorio, e l'OTP verrebbe chiesto due
    # volte. La condizione sta accanto all'altra, prima del modulo.
    nuova = kc.condizione_admin(facoltativo["displayName"], "non amministratore", nega=True)
    kc.ok(kc.c.post(f"/{REALM}/authentication/executions/{nuova['id']}/raise-priority"))

    # Obbligatorio per gli amministratori. Chi non l'ha ancora configurato lo
    # configura al primo accesso: è ciò che fa il modulo OTP quando è
    # REQUIRED e la credenziale manca.
    obbligatorio = kc.sottoflusso(moduli["displayName"], "2FA obbligatoria per gli amministratori",
                                  "OTP sempre richiesto a chi ha il ruolo admin")
    kc.requisito(moduli["displayName"], obbligatorio, "CONDITIONAL")
    kc.condizione_admin(obbligatorio["displayName"], "amministratore", nega=False)
    kc.requisito(obbligatorio["displayName"],
                 kc.esecuzione(obbligatorio["displayName"], "auth-otp-form"), "REQUIRED")
    kc.ok(kc.c.put(f"/{REALM}", json={"browserFlow": BROWSER}))

    # Dopo un accesso con Google, Facebook o Apple lo stesso vincolo: senza,
    # un amministratore che entra da un provider esterno salterebbe il
    # secondo fattore, perché il flusso del browser finisce al reindirizzamento.
    kc.ok(kc.c.post(f"/{REALM}/authentication/flows", json={
        "alias": DOPO_ESTERNO, "providerId": "basic-flow", "topLevel": True, "builtIn": False,
        "description": "OTP per gli amministratori anche dopo un accesso con un provider esterno",
    }))
    esterno = kc.sottoflusso(DOPO_ESTERNO, "2FA amministratori con provider esterno", "OTP per il ruolo admin")
    kc.requisito(DOPO_ESTERNO, esterno, "CONDITIONAL")
    kc.condizione_admin(esterno["displayName"], "amministratore esterno", nega=False)
    kc.requisito(esterno["displayName"], kc.esecuzione(esterno["displayName"], "auth-otp-form"), "REQUIRED")
    for alias in ("google", "facebook", "apple"):
        provider = kc.ok(kc.c.get(f"/{REALM}/identity-provider/instances/{alias}")).json()
        provider["postBrokerLoginFlowAlias"] = DOPO_ESTERNO
        kc.ok(kc.c.put(f"/{REALM}/identity-provider/instances/{alias}", json=provider))

    return kc.ok(kc.c.post(f"/{REALM}/partial-export",
                           params={"exportClients": "true", "exportGroupsAndRoles": "true"})).json()


def con_segnaposto(esportato: dict) -> str:
    for client in esportato["clients"]:
        if client["clientId"] == "persona-api":
            client["secret"] = "${KC_API_CLIENT_SECRET}"
        elif client.get("secret") == MASCHERA:
            client.pop("secret")
    for provider in esportato["identityProviders"]:
        alias = provider["alias"].upper()
        provider["config"]["clientSecret"] = f"${{KC_{alias}_CLIENT_SECRET}}"
        # Una stringa e non un booleano: Keycloak sostituisce il segnaposto
        # prima di leggere il JSON, e `false` è il valore se la variabile manca.
        provider["enabled"] = f"${{KC_{alias}_ENABLED:false}}"
    esportato["smtpServer"]["password"] = "${KC_SMTP_PASSWORD}"

    testo = json.dumps(esportato, ensure_ascii=False, indent=2)
    if MASCHERA in testo:
        raise SystemExit("È rimasto un valore mascherato: importato, diventerebbe il segreto vero.")
    return testo + "\n"


def main() -> None:
    argomenti = argparse.ArgumentParser()
    argomenti.add_argument("--keycloak", default="http://localhost:8181")
    argomenti.add_argument("--password", required=True, help="dell'admin dell'istanza usa-e-getta")
    a = argomenti.parse_args()

    esportato = configura(Keycloak(a.keycloak, a.password), realm_di_partenza())
    PRODUZIONE.write_text(con_segnaposto(esportato), encoding="utf-8")
    print(f"Scritto {PRODUZIONE}")


if __name__ == "__main__":
    main()
