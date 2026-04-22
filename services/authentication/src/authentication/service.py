from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import structlog
from database import UserRepository, get_session
from sqlalchemy.exc import SQLAlchemyError

logger = structlog.get_logger(__name__)

_tokens: dict[str, dict[str, Any]] = {}


class AuthService:
    def __init__(self) -> None:
        self.token_expiration_hours = int(os.getenv("TOKEN_EXPIRATION_HOURS", 24))

    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed_password.encode())

    def create_token(self, user_id: int, email: str) -> str:
        token = str(uuid.uuid4())
        _tokens[token] = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.now(tz=timezone.utc) + timedelta(hours=self.token_expiration_hours),
        }
        return token

    def verify_token(self, token: str) -> dict[str, Any] | None:
        payload = _tokens.get(token)
        if payload is None:
            logger.warning("Unknown token")
            return None
        if datetime.now(tz=timezone.utc) > payload["exp"]:
            _tokens.pop(token, None)
            logger.warning("Token has expired")
            return None
        return payload

    def register_user(self, email: str, password: str, full_name: str) -> dict[str, Any]:
        try:
            with get_session() as session:
                repo = UserRepository(session)
                if repo.get_by_email(email) is not None:
                    logger.warning("User already exists", email=email)
                    return {"error": "User already exists", "status": 409}

                user = repo.create(
                    email=email,
                    password_hash=self.hash_password(password),
                    full_name=full_name,
                )
                logger.info("User registered", email=email)
                return {
                    "status": 201,
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "full_name": user.full_name,
                        "created_at": user.created_at,
                    },
                    "message": "User registered successfully",
                }
        except SQLAlchemyError as e:
            logger.error("Database error during registration", error=str(e))
            return {"error": "Registration failed", "status": 500}

    def login_user(self, email: str, password: str) -> dict[str, Any]:
        try:
            with get_session() as session:
                repo = UserRepository(session)
                user = repo.get_by_email(email)

                if user is None:
                    logger.warning("Login attempt for non-existent user", email=email)
                    return {"error": "Invalid credentials", "status": 401}

                if not self.verify_password(password, user.password_hash):
                    logger.warning("Failed login attempt", email=email)
                    return {"error": "Invalid credentials", "status": 401}

                token = self.create_token(user.id, user.email)
                repo.update_last_login(user.id)

                logger.info("User logged in", email=email)
                return {
                    "status": 200,
                    "token": token,
                    "user": {"id": user.id, "email": user.email, "full_name": user.full_name},
                    "message": "Login successful",
                }
        except SQLAlchemyError as e:
            logger.error("Database error during login", error=str(e))
            return {"error": "Login failed", "status": 500}

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        try:
            with get_session() as session:
                user = UserRepository(session).get_by_id(user_id)
                if user is None:
                    return None
                return {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "created_at": user.created_at,
                    "last_login": user.last_login,
                }
        except SQLAlchemyError as e:
            logger.error("Database error fetching user", error=str(e))
            return None

    def update_user(
        self,
        user_id: int,
        full_name: str | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if full_name:
            updates["full_name"] = full_name
        if password:
            updates["password_hash"] = self.hash_password(password)
        if not updates:
            return {"error": "No updates provided", "status": 400}

        try:
            with get_session() as session:
                user = UserRepository(session).update(user_id, **updates)
                if user is None:
                    return {"error": "User not found", "status": 404}
                logger.info("User updated", user_id=user_id)
                return {
                    "status": 200,
                    "user": {"id": user.id, "email": user.email, "full_name": user.full_name},
                    "message": "User updated successfully",
                }
        except SQLAlchemyError as e:
            logger.error("Database error updating user", error=str(e))
            return {"error": "Update failed", "status": 500}

    def delete_user(self, user_id: int) -> dict[str, Any]:
        try:
            with get_session() as session:
                deleted = UserRepository(session).delete(user_id)
                if not deleted:
                    return {"error": "User not found", "status": 404}
                logger.info("User deleted", user_id=user_id)
                return {"status": 200, "message": "User deleted successfully"}
        except SQLAlchemyError as e:
            logger.error("Database error deleting user", error=str(e))
            return {"error": "Deletion failed", "status": 500}
