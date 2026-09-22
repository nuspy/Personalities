"""Verifica dei token emessi da Keycloak.

L'API **non convalida credenziali**: riceve un token già emesso e ne stabilisce
l'autenticità con la chiave pubblica del realm. È una differenza sostanziale —
qui non passa mai una password, quindi non c'è nulla da custodire.

Cosa viene controllato, e perché ciascuna cosa conta:

- **la firma**, con la chiave pubblica indicata dal `kid` dell'intestazione;
- **l'emittente**, perché un token perfettamente firmato da un altro realm — o
  da un altro Keycloak — resta un token di qualcun altro;
- **la scadenza**, con una tolleranza minima per lo sfasamento degli orologi;
- **il destinatario**, che è il controllo che si dimentica più spesso: senza,
  un token emesso per un client diverso, magari di un'applicazione terza che
  usa lo stesso realm, verrebbe accettato qui.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import httpx
from jose import jwt
from jose.exceptions import JWTError

from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: Per quanto tenere le chiavi pubbliche senza richiederle di nuovo. Keycloak
#: le ruota di rado, ma quando accade i token nuovi portano un `kid` ignoto:
#: quel caso è gestito da un aggiornamento immediato, non dall'attesa della
#: scadenza. Questo valore è solo il limite superiore.
JWKS_TTL_SECONDS = 3600

#: Tolleranza sullo sfasamento degli orologi fra chi emette e chi verifica.
#: Senza, un token appena emesso può risultare "non ancora valido" su una
#: macchina indietro di qualche secondo — un errore intermittente e inspiegabile.
CLOCK_SKEW_SECONDS = 30


class AuthenticationError(Exception):
    """Il token non è utilizzabile. Il messaggio è per i log, non per il client.

    All'esterno si risponde 401 senza dettagli: distinguere «firma non valida»
    da «scaduto» da «destinatario sbagliato» dice a chi prova quale pezzo
    cambiare al tentativo successivo.
    """


@dataclass(frozen=True)
class Principal:
    """Chi sta facendo la richiesta, secondo il token.

    Non è l'utente del database: è ciò che Keycloak afferma. Le due cose si
    incontrano in `users.keycloak_sub`, ed è un incontro deliberato — questo
    oggetto resta valido anche prima che una riga esista.
    """

    subject: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    locale: str = "it"
    raw_claims: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def has_role(self, role: str) -> bool:
        return role in self.roles


class JwksCache:
    """Le chiavi pubbliche del realm, riprese quando servono.

    Richiederle a ogni token significherebbe una chiamata HTTP per richiesta e
    legherebbe la disponibilità dell'API a quella di Keycloak per ogni singolo
    accesso, non solo al momento del login.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def jwks_url(self) -> str:
        base = self._settings.keycloak_url.rstrip("/")
        realm = self._settings.keycloak_realm
        return f"{base}/realms/{realm}/protocol/openid-connect/certs"

    @property
    def issuer(self) -> str:
        # Nel cluster le chiavi si leggono dal servizio interno, ma i token
        # portano l'indirizzo pubblico con cui il browser ha parlato a
        # Keycloak: sono due indirizzi, e confonderli farebbe rifiutare ogni
        # token con «emittente non valido».
        base = (self._settings.keycloak_issuer_url or self._settings.keycloak_url).rstrip("/")
        return f"{base}/realms/{self._settings.keycloak_realm}"

    def key_for(self, kid: str, *, allow_refresh: bool = True) -> Dict[str, Any]:
        """La chiave con quel `kid`, aggiornando l'insieme se è sconosciuto.

        L'aggiornamento su `kid` ignoto è ciò che rende indolore la rotazione
        delle chiavi: senza, per tutta la durata della cache i token nuovi
        verrebbero rifiutati — e il rifiuto sembrerebbe un problema di firma.
        """
        if self._scaduta():
            self._refresh()

        key = self._keys.get(kid)
        if key is None and allow_refresh:
            logger.info("Chiave %s sconosciuta: rilettura del JWKS", kid)
            self._refresh(force=True)
            key = self._keys.get(kid)

        if key is None:
            raise AuthenticationError(
                f"il token è firmato con una chiave ({kid}) che il realm non espone"
            )
        return key

    def _scaduta(self) -> bool:
        return time.time() - self._fetched_at > JWKS_TTL_SECONDS

    def _refresh(self, *, force: bool = False) -> None:
        with self._lock:
            # Un secondo controllo dentro il lock: mentre si aspettava, un
            # altro thread può avere già aggiornato. Senza, una raffica di
            # richieste con un `kid` nuovo produrrebbe una raffica di chiamate
            # identiche a Keycloak.
            if not force and not self._scaduta():
                return
            try:
                response = httpx.get(self.jwks_url, timeout=5.0)
                response.raise_for_status()
                documento = response.json()
            except Exception as exc:
                raise AuthenticationError(
                    f"chiavi pubbliche non recuperabili da {self.jwks_url}: {exc}"
                ) from exc

            self._keys = {k["kid"]: k for k in documento.get("keys", []) if "kid" in k}
            self._fetched_at = time.time()
            logger.info("JWKS aggiornato: %d chiavi", len(self._keys))

    def invalidate(self) -> None:
        self._keys = {}
        self._fetched_at = 0.0


class TokenVerifier:
    """Da token a `Principal`, o eccezione."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        jwks: Optional[JwksCache] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._jwks = jwks or JwksCache(self._settings)

    def verify(self, token: str) -> Principal:
        try:
            intestazione = jwt.get_unverified_header(token)
        except JWTError as exc:
            raise AuthenticationError(f"intestazione illeggibile: {exc}") from exc

        kid = intestazione.get("kid")
        if not kid:
            raise AuthenticationError("il token non dichiara quale chiave l'ha firmato")

        chiave = self._jwks.key_for(kid)

        try:
            claims = jwt.decode(
                token,
                chiave,
                algorithms=[intestazione.get("alg", "RS256")],
                issuer=self._jwks.issuer,
                # Keycloak mette il client negli `aud` solo se un mapper lo
                # prevede; quando non c'è, il destinatario si legge da `azp`.
                # La verifica avviene comunque, sotto, su entrambi i campi: qui
                # è disattivata solo perché jose fallirebbe sul caso legittimo
                # in cui `aud` contiene `account`.
                options={
                    "verify_aud": False,
                    "leeway": CLOCK_SKEW_SECONDS,
                },
            )
        except JWTError as exc:
            raise AuthenticationError(f"token non valido: {exc}") from exc

        self._verify_audience(claims)
        return self._to_principal(claims)

    def _verify_audience(self, claims: Dict[str, Any]) -> None:
        """Il token è stato emesso per noi?

        Questo è il controllo che separa «un token valido» da «un token valido
        per questo servizio». Nello stesso realm possono vivere più client, e
        senza questa verifica quello di un'applicazione terza aprirebbe anche
        le nostre porte.
        """
        atteso = self._settings.keycloak_client_id
        destinatari = claims.get("aud")
        if isinstance(destinatari, str):
            destinatari = [destinatari]
        destinatari = set(destinatari or [])

        parte_autorizzata = claims.get("azp")
        if parte_autorizzata:
            destinatari.add(parte_autorizzata)

        consentiti = {atteso, *self._settings.keycloak_accepted_audiences}
        if not destinatari & consentiti:
            raise AuthenticationError(
                f"token emesso per {sorted(destinatari) or 'nessuno'}, "
                f"non per {sorted(consentiti)}"
            )

    def _to_principal(self, claims: Dict[str, Any]) -> Principal:
        ruoli = self._extract_roles(claims)
        return Principal(
            subject=claims["sub"],
            email=claims.get("email"),
            display_name=claims.get("name") or claims.get("preferred_username"),
            roles=ruoli,
            locale=claims.get("locale") or "it",
            raw_claims=claims,
        )

    def _extract_roles(self, claims: Dict[str, Any]) -> Set[str]:
        """Ruoli di realm e ruoli del client, uniti.

        Keycloak li tiene in due posti diversi e la scelta dipende da come è
        configurato il realm. Leggerne uno solo significa che un giorno, dopo
        una modifica lato identità, tutti diventano utenti semplici senza che
        nulla segnali il perché.
        """
        ruoli: Set[str] = set()

        realm: List[str] = (claims.get("realm_access") or {}).get("roles", [])
        ruoli.update(realm)

        risorse: Dict[str, Any] = claims.get("resource_access") or {}
        for client in (self._settings.keycloak_client_id, "persona-frontend"):
            ruoli.update((risorse.get(client) or {}).get("roles", []))

        # Ruoli tecnici che Keycloak assegna da sé: tenerli confonderebbe un
        # controllo `has_role` scritto in buona fede.
        return {r for r in ruoli if r not in _RUOLI_TECNICI}


_RUOLI_TECNICI = {
    "offline_access",
    "uma_authorization",
    "default-roles-personalities",
    "manage-account",
    "manage-account-links",
    "view-profile",
}
