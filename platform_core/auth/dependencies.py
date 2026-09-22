"""Dipendenze di autenticazione per gli endpoint.

Tre livelli, in ordine di impegno crescente:

- `current_principal` — chi dice di essere il token. Non tocca il database.
- `current_user` — la riga corrispondente, creata al primo accesso.
- `require_role("admin")` — l'autorizzazione vera e propria.

Un endpoint che dichiara `user: CurrentUser` è autenticato per costruzione: non
c'è modo di dimenticarsene a metà, perché il parametro non sarebbe valorizzato.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.models import User
from ..domain.repositories import UserRepository
from ..domain.session import get_db_session
from ..settings import get_settings
from .keycloak import AuthenticationError, Principal, TokenVerifier

logger = logging.getLogger(__name__)

#: Identità usata quando l'autenticazione è disattivata in sviluppo. Ha un
#: `sub` riconoscibile a occhio: se compare in un database che non è di
#: sviluppo, si vede subito da dove è arrivata.
PRINCIPAL_DI_SVILUPPO = Principal(
    subject="dev-00000000-0000-0000-0000-000000000000",
    email="sviluppo@localhost",
    display_name="Utente di sviluppo",
    roles={"user", "admin"},
)


@lru_cache(maxsize=1)
def get_token_verifier() -> TokenVerifier:
    return TokenVerifier()


def reset_auth() -> None:
    """Dimentica il verificatore memorizzato. Solo per i test."""
    get_token_verifier.cache_clear()


async def current_principal(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
) -> Principal:
    """Verifica il token e restituisce chi lo presenta.

    In sviluppo, con `PERSONA_AUTH_DISABLED=true`, restituisce un'identità
    fittizia — utile per provare gli endpoint senza avviare Keycloak. Che quel
    flag non sopravviva alla produzione è verificato all'avvio da
    `Settings.validate_production()`: un controllo qui arriverebbe comunque
    troppo tardi, a servizio già in ascolto.
    """
    settings = get_settings()

    if settings.auth_disabled:
        return PRINCIPAL_DI_SVILUPPO

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticazione richiesta",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1].strip()
    try:
        principal = get_token_verifier().verify(token)
    except AuthenticationError as exc:
        # Il motivo resta nei log; al client va un 401 spoglio. Dire «firma
        # non valida» invece di «scaduto» aiuta chi sta provando a indovinare.
        logger.warning(
            "Token rifiutato (%s): %s", request.url.path, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticazione non valida",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return principal


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]


async def current_user(
    principal: CurrentPrincipal,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """L'utente del database, allineato al token."""
    try:
        utente = await UserRepository(session).ensure(principal)
        if getattr(utente, "appena_creato", False):
            # Il piano gratuito si apre qui e non dentro `ensure`: il dominio
            # degli utenti non deve sapere che esiste una tariffa, o le due
            # cose non si possono più cambiare separatamente.
            from ..billing.plans import GestoreAbbonamenti

            await GestoreAbbonamenti(session).apri_piano_base(utente.id)
            await session.commit()
        return utente
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc),
        ) from exc


CurrentUser = Annotated[User, Depends(current_user)]


def require_role(role: str):
    """Dipendenza che pretende un ruolo.

        @router.get("/admin/users", dependencies=[Depends(require_role("admin"))])

    Risponde 403 e non 404: nascondere l'esistenza di un endpoint
    amministrativo non protegge nulla — il percorso è nello schema OpenAPI — e
    rende incomprensibile il messaggio a un amministratore a cui manca davvero
    il ruolo.
    """

    async def _guard(principal: CurrentPrincipal) -> Principal:
        if not principal.has_role(role):
            logger.warning(
                "Accesso negato a %s: ruoli %s, richiesto %s",
                principal.subject[:8], sorted(principal.roles), role,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Richiede il ruolo '{role}'",
            )
        return principal

    return _guard


DbSession = Annotated[AsyncSession, Depends(get_db_session)]
