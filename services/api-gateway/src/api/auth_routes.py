"""
Authentication API Routes
Handles user login, registration, and token refresh
"""
import logging
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field

from src.core.config import settings
from shared.auth.jwt_handler import JWTHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Initialize JWT handler (singleton)
jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get or create JWT handler instance"""
    global jwt_handler
    if jwt_handler is None:
        jwt_handler = JWTHandler(
            secret_key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
            access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            refresh_token_expire_days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    return jwt_handler


# Request/Response Models
class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


# In-memory user store (for demo - use database in production!)
# Format: email -> {hashed_password, user_id, username, roles, ...}
user_store = {}


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    handler: JWTHandler = Depends(get_jwt_handler)
) -> TokenResponse:
    """
    Register a new user

    Args:
        request: Registration request with email, password, username

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If email already registered
    """
    # Check if user already exists
    if request.email in user_store:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password
    hashed_password = handler.hash_password(request.password)

    # Generate user ID
    user_id = f"user_{len(user_store) + 1}"

    # Store user (in production, save to database)
    user_store[request.email] = {
        "user_id": user_id,
        "email": request.email,
        "username": request.username,
        "full_name": request.full_name,
        "hashed_password": hashed_password,
        "roles": ["user"],  # Default role
        "is_active": True
    }

    logger.info(f"New user registered: {request.email} (ID: {user_id})")

    # Create tokens
    token_data = {
        "user_id": user_id,
        "email": request.email,
        "username": request.username,
        "roles": ["user"]
    }

    access_token = handler.create_access_token(token_data)
    refresh_token = handler.create_refresh_token({"user_id": user_id})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    handler: JWTHandler = Depends(get_jwt_handler)
) -> TokenResponse:
    """
    Login with email and password

    Args:
        request: Login request with email and password

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If credentials are invalid
    """
    # Get user from store
    user = user_store.get(request.email)

    if user is None:
        logger.warning(f"Login attempt for non-existent user: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password
    if not handler.verify_password(request.password, user["hashed_password"]):
        logger.warning(f"Invalid password for user: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    logger.info(f"User logged in: {request.email}")

    # Create tokens
    token_data = {
        "user_id": user["user_id"],
        "email": user["email"],
        "username": user["username"],
        "roles": user.get("roles", ["user"])
    }

    access_token = handler.create_access_token(token_data)
    refresh_token = handler.create_refresh_token({"user_id": user["user_id"]})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    handler: JWTHandler = Depends(get_jwt_handler)
) -> TokenResponse:
    """
    Refresh access token using refresh token

    Args:
        request: Refresh token request

    Returns:
        New access and refresh tokens

    Raises:
        HTTPException: If refresh token is invalid
    """
    # Verify refresh token
    payload = handler.verify_refresh_token(request.refresh_token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    user_id = payload.get("user_id")

    # Find user in store
    user = None
    for email, user_data in user_store.items():
        if user_data["user_id"] == user_id:
            user = user_data
            break

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    logger.info(f"Token refreshed for user: {user['email']}")

    # Create new tokens
    token_data = {
        "user_id": user["user_id"],
        "email": user["email"],
        "username": user["username"],
        "roles": user.get("roles", ["user"])
    }

    access_token = handler.create_access_token(token_data)
    new_refresh_token = handler.create_refresh_token({"user_id": user["user_id"]})

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me")
async def get_current_user_info(handler: JWTHandler = Depends(get_jwt_handler)):
    """
    Get current user information (requires authentication)

    This endpoint demonstrates how authentication would work.
    In production, use proper dependency injection for current user.
    """
    return {
        "message": "This endpoint requires authentication when ENABLE_AUTHENTICATION=True",
        "note": "Implement using AuthenticationMiddleware or FastAPI dependencies"
    }
