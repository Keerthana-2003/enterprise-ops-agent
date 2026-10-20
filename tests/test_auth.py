"""Cognito JWT verification with a locally generated RSA key standing in for the user pool JWKS."""
from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from api.auth import AuthError, CognitoVerifier, bearer_token

POOL, CLIENT, REGION = "us-east-1_TestPool", "client123", "us-east-1"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJWKS:
    class _K:
        key = KEY.public_key()

    def get_signing_key_from_jwt(self, token):
        return self._K()


def make_token(key=KEY, **overrides) -> str:
    claims = {"iss": ISSUER, "token_use": "access", "client_id": CLIENT, "sub": "user-1",
              "exp": int(time.time()) + 300, **overrides}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture()
def verifier():
    return CognitoVerifier(REGION, POOL, CLIENT, jwks_client=FakeJWKS())


def test_valid_access_and_id_tokens(verifier):
    assert verifier.verify(make_token())["sub"] == "user-1"
    id_token = make_token(token_use="id", aud=CLIENT, client_id=None)
    assert verifier.verify(id_token)["token_use"] == "id"


@pytest.mark.parametrize("bad", [
    {"exp": int(time.time()) - 10},
    {"iss": "https://evil.example.com"},
    {"client_id": "someone-else"},
    {"token_use": "refresh"},
])
def test_rejects_bad_claims(verifier, bad):
    with pytest.raises(AuthError):
        verifier.verify(make_token(**bad))


def test_rejects_wrong_signature(verifier):
    with pytest.raises(AuthError, match="Invalid token"):
        verifier.verify(make_token(key=OTHER_KEY))


def test_bearer_header_parsing():
    assert bearer_token("Bearer abc.def") == "abc.def"
    with pytest.raises(AuthError):
        bearer_token(None)
    with pytest.raises(AuthError):
        bearer_token("Basic xyz")
