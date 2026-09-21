"""Verifica dei token, con un realm finto ma crittografia vera.

Le chiavi sono generate qui e i token firmati sul momento: nessun Keycloak in
esecuzione, ma nemmeno una verifica simulata. Ciò che si prova è il codice
reale — la stessa `TokenVerifier` che gira in produzione — contro token
costruiti apposta per essere sbagliati in un modo preciso.

I casi negativi sono il punto. Un verificatore che accetta i token buoni è
facile; quello che conta è che rifiuti un token valido ma emesso per qualcun
altro, e che lo faccia per la ragione giusta.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from platform_core.auth.keycloak import (
    AuthenticationError, JwksCache, Principal, TokenVerifier,
)
from platform_core.settings import Settings

REALM = "personalities"
BASE = "http://keycloak-di-prova:8080"
ISSUER = f"{BASE}/realms/{REALM}"
KID = "chiave-di-prova"


@pytest.fixture(scope="module")
def chiave():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def pem(chiave) -> str:
    return chiave.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


@pytest.fixture(scope="module")
def jwks(chiave) -> Dict[str, Any]:
    numeri = chiave.public_key().public_numbers()

    def b64(n: int) -> str:
        import base64

        grezzo = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(grezzo).decode().rstrip("=")

    return {
        "keys": [{
            "kty": "RSA", "kid": KID, "use": "sig", "alg": "RS256",
            "n": b64(numeri.n), "e": b64(numeri.e),
        }]
    }


@pytest.fixture
def impostazioni() -> Settings:
    return Settings(
        keycloak_url=BASE,
        keycloak_realm=REALM,
        keycloak_client_id="persona-api",
        keycloak_accepted_audiences=("persona-frontend",),
    )


@pytest.fixture
def verificatore(impostazioni, jwks) -> TokenVerifier:
    """Verificatore con le chiavi già in memoria: nessuna chiamata di rete."""
    cache = JwksCache(impostazioni)
    cache._keys = {k["kid"]: k for k in jwks["keys"]}
    cache._fetched_at = time.time()
    return TokenVerifier(impostazioni, cache)


def emetti(pem: str, **sovrascrivi) -> str:
    adesso = int(time.time())
    claims = {
        "sub": "abc-123",
        "iss": ISSUER,
        "aud": "account",
        "azp": "persona-frontend",
        "exp": adesso + 900,
        "iat": adesso,
        "email": "utente@example.com",
        "name": "Utente Di Prova",
        "realm_access": {"roles": ["user", "offline_access"]},
    }
    claims.update(sovrascrivi)
    return jwt.encode(claims, pem, algorithm="RS256", headers={"kid": KID})


class TestTokenValidi:
    def test_un_token_buono_diventa_un_principal(self, verificatore, pem):
        principal = verificatore.verify(emetti(pem))

        assert principal.subject == "abc-123"
        assert principal.email == "utente@example.com"
        assert principal.display_name == "Utente Di Prova"

    def test_i_ruoli_tecnici_non_compaiono(self, verificatore, pem):
        """`offline_access` è di Keycloak, non un ruolo dell'applicazione.

        Lasciarlo confonderebbe un controllo scritto in buona fede: un elenco
        di ruoli pieno di voci tecniche invita a cercarci dentro la stringa
        sbagliata.
        """
        principal = verificatore.verify(emetti(pem))

        assert principal.roles == {"user"}
        assert not principal.is_admin

    def test_il_ruolo_di_amministratore_viene_riconosciuto(self, verificatore, pem):
        token = emetti(pem, realm_access={"roles": ["user", "admin"]})
        assert verificatore.verify(token).is_admin

    def test_i_ruoli_del_client_si_sommano_a_quelli_di_realm(self, verificatore, pem):
        """Keycloak può tenerli in due posti: leggerne uno solo non basta."""
        token = emetti(
            pem,
            realm_access={"roles": ["user"]},
            resource_access={"persona-api": {"roles": ["admin"]}},
        )
        assert verificatore.verify(token).roles == {"user", "admin"}

    def test_il_destinatario_puo_essere_il_client_dell_api(self, verificatore, pem):
        token = emetti(pem, aud="persona-api", azp="persona-api")
        assert verificatore.verify(token).subject == "abc-123"


class TestTokenRifiutati:
    def test_firma_manomessa(self, verificatore, pem):
        token = emetti(pem)[:-8] + "AAAAAAAA"
        with pytest.raises(AuthenticationError):
            verificatore.verify(token)

    def test_scaduto(self, verificatore, pem):
        token = emetti(pem, exp=int(time.time()) - 3600)
        with pytest.raises(AuthenticationError, match="non valido"):
            verificatore.verify(token)

    def test_emittente_diverso(self, verificatore, pem):
        """Un token di un altro realm è firmato bene, ma non è nostro."""
        token = emetti(pem, iss="http://keycloak-di-prova:8080/realms/altro")
        with pytest.raises(AuthenticationError):
            verificatore.verify(token)

    def test_destinatario_di_un_altra_applicazione(self, verificatore, pem):
        """Il controllo che si dimentica più spesso.

        Nello stesso realm possono vivere più client: senza questa verifica, il
        token di un'applicazione terza aprirebbe anche le nostre porte.
        """
        token = emetti(pem, aud="applicazione-terza", azp="applicazione-terza")
        with pytest.raises(AuthenticationError, match="emesso per"):
            verificatore.verify(token)

    def test_senza_indicazione_della_chiave(self, verificatore, pem):
        token = jwt.encode({"sub": "x", "iss": ISSUER}, pem, algorithm="RS256")
        with pytest.raises(AuthenticationError, match="quale chiave"):
            verificatore.verify(token)

    def test_firmato_con_una_chiave_ignota(self, verificatore, impostazioni):
        altra = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem_altra = altra.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        token = jwt.encode(
            {"sub": "x", "iss": ISSUER, "exp": int(time.time()) + 60},
            pem_altra, algorithm="RS256", headers={"kid": "chiave-sconosciuta"},
        )
        # Il rinnovo del JWKS proverebbe una chiamata di rete verso un host
        # inesistente: il fallimento arriva da lì, ed è comunque un rifiuto.
        with pytest.raises(AuthenticationError):
            verificatore.verify(token)

    def test_non_un_token(self, verificatore):
        with pytest.raises(AuthenticationError, match="intestazione"):
            verificatore.verify("questo-non-e-un-token")


class TestTolleranzaSugliOrologi:
    def test_un_token_appena_emesso_da_una_macchina_avanti(self, verificatore, pem):
        """Qualche secondo di sfasamento non deve produrre un 401.

        È il genere di guasto che si manifesta a intermittenza e solo su una
        macchina, e che si finisce per attribuire alla rete.
        """
        futuro = int(time.time()) + 10
        token = emetti(pem, iat=futuro, nbf=futuro)
        assert verificatore.verify(token).subject == "abc-123"


class TestPrincipal:
    def test_ha_ruolo(self):
        p = Principal(subject="x", roles={"user", "admin"})
        assert p.has_role("admin") and not p.has_role("supervisore")

    def test_senza_ruoli_non_e_amministratore(self):
        assert not Principal(subject="x").is_admin
