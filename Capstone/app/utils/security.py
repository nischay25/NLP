import time
import os
from collections import defaultdict
from fastapi import Security, HTTPException, Request, status
from fastapi.security.api_key import APIKeyHeader
from backend.app.config import settings

# API Key security scheme using X-API-Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """
    Dependency to verify API keys for administrative or protected endpoints.
    Protects endpoint if settings.API_KEY is configured.
    """
    if not settings.API_KEY:
        return None  # Skip if auth key isn't configured
        
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. X-API-Key header is missing."
        )
        
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. Invalid X-API-Key provided."
        )
        
    return api_key


class InMemoryRateLimiter:
    """
    Simple in-memory IP-based rate limiter.
    """
    def __init__(self):
        self.requests = defaultdict(list)

    def check_rate_limit(self, request: Request):
        # We can disable rate limiting if config values are zero/negative
        if settings.RATE_LIMIT_MAX <= 0:
            return

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        
        # Clean up old timestamps outside window
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] 
            if now - t < settings.RATE_LIMIT_WINDOW
        ]
        
        if len(self.requests[client_ip]) >= settings.RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_MAX} requests per {settings.RATE_LIMIT_WINDOW} seconds are allowed."
            )
            
        self.requests[client_ip].append(now)

# Singleton rate limiter instance
rate_limiter = InMemoryRateLimiter()
