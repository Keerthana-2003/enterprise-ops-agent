"""Amazon Cognito JWT authentication for the API.

Enabled when COGNITO_USER_POOL_ID and COGNITO_CLIENT_ID are set (Terraform and CDK set them on
App Runner). Verifies the token signature against the user pool's JWKS, expiry, issuer,
token_use, and that the token was issued for this app client. With no pool configured the
API runs open, for local development.
"""
from __future__ import annotations

import os

import jwt
from jwt import PyJWKClient


class AuthError(Exception):
    pass


class CognitoVerifier:
    def __init__(self, region: str, user_pool_id: str, client_id: str, jwks_client=None):
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
        self.client_id = client_id
        self.jwks = jwks_client or PyJWKClient(f"{self.issuer}/.well-known/jwks.json", cache_keys=True)

    def verify(self, token: str) -> dict:
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.issuer,
                                options={"require": ["exp", "iss", "token_use"], "verify_aud": False})
        except jwt.PyJWTError as err:
            raise AuthError(f"Invalid token: {err}") from None
        use = claims.get("token_use")
        issued_for = claims.get("client_id") if use == "access" else claims.get("aud") if use == "id" else None
        if issued_for != self.client_id:
            raise AuthError("Token was not issued for this app client")
        return claims


def verifier_from_env() -> CognitoVerifier | None:
    pool, client = os.getenv("COGNITO_USER_POOL_ID"), os.getenv("COGNITO_CLIENT_ID")
    if not (pool and client):
        return None
    region = os.getenv("COGNITO_REGION") or pool.split("_")[0]
    return CognitoVerifier(region, pool, client)


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("Missing bearer token")
    return authorization.split(" ", 1)[1].strip()
