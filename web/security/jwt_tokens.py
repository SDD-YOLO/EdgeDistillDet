from __future__ import annotations

import time

from jose import JWTError, jwt

from web.core.saas_settings import (
    get_access_token_expire_minutes,
    get_jwt_algorithm,
    get_jwt_secret,
    get_refresh_token_expire_days,
)


def create_tokens(subject_user_id: str) -> dict[str, str]:
    now_ts = int(time.time())
    secret = get_jwt_secret()
    alg = get_jwt_algorithm()
    access_min = get_access_token_expire_minutes()
    refresh_days = get_refresh_token_expire_days()

    access = jwt.encode(
        {
            "sub": subject_user_id,
            "typ": "access",
            "iat": now_ts,
            "exp": now_ts + access_min * 60,
        },
        secret,
        algorithm=alg,
    )
    refresh = jwt.encode(
        {
            "sub": subject_user_id,
            "typ": "refresh",
            "iat": now_ts,
            "exp": now_ts + refresh_days * 86400,
        },
        secret,
        algorithm=alg,
    )
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}


def decode_token(token: str) -> dict:
    return jwt.decode(token, get_jwt_secret(), algorithms=[get_jwt_algorithm()])


def decode_token_safe(token: str) -> dict | None:
    try:
        return decode_token(token)
    except JWTError:
        return None
