"""Identità e autorizzazione.

L'autenticazione vive in Keycloak; qui si verifica ciò che ha emesso e si
traduce in un utente della piattaforma.
"""
from .dependencies import (
    CurrentPrincipal, CurrentUser, DbSession, current_principal, current_user,
    require_role,
)
from .keycloak import AuthenticationError, JwksCache, Principal, TokenVerifier

__all__ = [
    "AuthenticationError",
    "CurrentPrincipal",
    "CurrentUser",
    "DbSession",
    "JwksCache",
    "Principal",
    "TokenVerifier",
    "current_principal",
    "current_user",
    "require_role",
]
