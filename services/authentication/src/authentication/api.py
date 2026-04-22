from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
import os

from authentication.service import AuthService

logger = structlog.get_logger(__name__)

# Initialize auth service
auth_service = AuthService()


# Request/Response models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    created_at: Optional[str] = None
    last_login: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    user: UserResponse
    message: str


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    error: str


def verify_token(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency to verify JWT token from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    logger.info("Authentication service starting...")
    yield
    logger.info("Authentication service shutting down...")


app = FastAPI(
    title="Authentication Service API",
    description="User authentication and management service",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/register", response_model=UserResponse, status_code=201)
async def register(request: RegisterRequest):
    """Register a new user"""
    result = auth_service.register_user(
        email=request.email,
        password=request.password,
        full_name=request.full_name
    )
    
    if "error" in result:
        status_code = result.get("status", 500)
        raise HTTPException(status_code=status_code, detail=result["error"])
    
    logger.info(f"User registered: {request.email}")
    user = result["user"]
    return UserResponse(
        id=user['id'],
        email=user['email'],
        full_name=user['full_name'],
        created_at=str(user['created_at']) if user.get('created_at') else None
    )


@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login user and receive JWT token"""
    result = auth_service.login_user(
        email=request.email,
        password=request.password
    )
    
    if "error" in result:
        status_code = result.get("status", 500)
        raise HTTPException(status_code=status_code, detail=result["error"])
    
    user = result["user"]
    return LoginResponse(
        token=result["token"],
        user=UserResponse(
            id=user['id'],
            email=user['email'],
            full_name=user['full_name']
        ),
        message=result["message"]
    )


@app.get("/me", response_model=UserResponse)
async def get_current_user(payload: Dict[str, Any] = Depends(verify_token)):
    """Get current user information"""
    user = auth_service.get_user(payload['user_id'])
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user['id'],
        email=user['email'],
        full_name=user['full_name'],
        created_at=str(user['created_at']) if user.get('created_at') else None,
        last_login=str(user['last_login']) if user.get('last_login') else None
    )


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)


@app.put("/me", response_model=UserResponse)
async def update_user(
    request: UpdateUserRequest,
    payload: Dict[str, Any] = Depends(verify_token)
):
    """Update current user information"""
    result = auth_service.update_user(
        user_id=payload['user_id'],
        full_name=request.full_name,
        password=request.password
    )
    
    if "error" in result:
        status_code = result.get("status", 500)
        raise HTTPException(status_code=status_code, detail=result["error"])
    
    user = result["user"]
    return UserResponse(
        id=user['id'],
        email=user['email'],
        full_name=user['full_name']
    )


@app.delete("/me", response_model=MessageResponse)
async def delete_user(payload: Dict[str, Any] = Depends(verify_token)):
    """Delete current user account"""
    result = auth_service.delete_user(payload['user_id'])
    
    if "error" in result:
        status_code = result.get("status", 500)
        raise HTTPException(status_code=status_code, detail=result["error"])
    
    return MessageResponse(message=result["message"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/verify-token")
async def verify_token_endpoint(authorization: Optional[str] = Header(None)):
    """Verify if a token is valid"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {
        "valid": True,
        "user_id": payload.get('user_id'),
        "email": payload.get('email')
    }
