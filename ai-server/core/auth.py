"""
JWT Authentication for AI Server
"""

import jwt
import os
from fastapi import HTTPException, Request, status
from typing import Optional


def get_jwt_secret() -> str:
    """Get JWT secret from environment variable"""
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise ValueError("JWT_SECRET environment variable is not set")
    return secret


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        secret = get_jwt_secret()
        # Decode token without verification first to check algorithm
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256", "HS512"],  # Support both algorithms
            options={"verify_signature": True},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def extract_token_from_request(request: Request) -> Optional[str]:
    """
    Extract JWT token from request headers

    Args:
        request: FastAPI request object

    Returns:
        Token string if found, None otherwise
    """
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    # Check x-access-token header (alternative)
    token = request.headers.get("x-access-token")
    if token:
        return token

    return None


def verify_request_auth(request: Request) -> dict:
    """
    Verify request authentication and return user data

    Args:
        request: FastAPI request object

    Returns:
        Decoded token payload with user information

    Raises:
        HTTPException: If authentication fails
    """
    token = extract_token_from_request(request)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_token(token)
