# Authentication Service

A FastAPI-based authentication service for user management with JWT token support.

## Features

- User registration with email validation
- User login with JWT token generation
- Password hashing using bcrypt (12 rounds)
- JWT token verification
- User profile management (get, update, delete)
- Database persistence with PostgreSQL
- Structured logging with structlog

## Setup

### Prerequisites

- Python 3.13+
- PostgreSQL database running
- Environment variables configured

### Environment Variables

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
MAIN_DB=your_database_name
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_EXPIRATION_HOURS=24
PORT=8001
WORKERS=1
```

### Installation

```bash
# Install dependencies
uv sync

# Create tables (if not already created)
cd ../../utils
python init_tables.py
```

## API Endpoints

### Authentication

#### Register
```
POST /register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "full_name": "John Doe"
}

Response (201):
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2026-04-18T10:30:00"
}
```

#### Login
```
POST /login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123"
}

Response (200):
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe"
  },
  "message": "Login successful"
}
```

#### Get Current User
```
GET /me
Authorization: Bearer <token>

Response (200):
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2026-04-18T10:30:00",
  "last_login": "2026-04-18T11:00:00"
}
```

#### Update User
```
PUT /me
Authorization: Bearer <token>
Content-Type: application/json

{
  "full_name": "John Updated",
  "password": "NewSecurePassword123"
}

Response (200):
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Updated"
}
```

#### Delete User
```
DELETE /me
Authorization: Bearer <token>

Response (200):
{
  "message": "User deleted successfully"
}
```

### Token Management

#### Verify Token
```
POST /verify-token
Authorization: Bearer <token>

Response (200):
{
  "valid": true,
  "user_id": 1,
  "email": "user@example.com"
}
```

### Health

#### Health Check
```
GET /health

Response (200):
{
  "status": "healthy"
}
```

## Database Schema

### Users Table

```sql
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   TEXT         NOT NULL,
    full_name       VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login      TIMESTAMPTZ,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE
);
```

## Running the Service

### Development Mode

```bash
cd /path/to/services/authentication
uv run authentication
```

The service will start on `http://0.0.0.0:8001`

### Docker

```bash
docker build -t authentication-service .
docker run -p 8001:8001 --env-file .env authentication-service
```

## Security Notes

1. **Change JWT_SECRET_KEY** in production to a strong, random value
2. **Use HTTPS** in production to protect token transmission
3. **Password Requirements**: Minimum 8 characters (enforced by API)
4. **Bcrypt Rounds**: Set to 12 for optimal security/performance balance
5. **Token Expiration**: Default 24 hours, configurable via JWT_EXPIRATION_HOURS

## Error Responses

### 400 Bad Request
```json
{
  "detail": "No updates provided"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid credentials"
}
```

### 404 Not Found
```json
{
  "detail": "User not found"
}
```

### 409 Conflict
```json
{
  "detail": "User already exists"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Login failed"
}
```

## Development

### Running Tests

```bash
# Tests coming soon
```

### Code Style

The project follows PEP 8 and uses structlog for logging.

## License

MIT
