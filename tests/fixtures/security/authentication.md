# Authentication Patterns

## JWT (JSON Web Tokens)

JWTs are stateless tokens for API authentication:

```python
import jwt
from datetime import datetime, timedelta

def create_token(user_id: int, secret: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    return jwt.encode(payload, secret, algorithm="HS256")

def verify_token(token: str, secret: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"])
```

Use short-lived access tokens (15min-1hr) with longer-lived refresh tokens. Store refresh tokens in httpOnly cookies, never in localStorage.

## OAuth 2.0 Authorization Code Flow

1. Client redirects user to authorization server
2. User authenticates and consents
3. Authorization server redirects back with authorization code
4. Client exchanges code for access token (server-to-server)
5. Client uses access token to access resources

## Password Hashing

Never store plaintext passwords. Use bcrypt or argon2:

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

hashed = pwd_context.hash("user_password")
is_valid = pwd_context.verify("user_password", hashed)
```
