from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets
import base64

from jose import jwt, JWTError
from passlib.context import CryptContext

from config import settings

# Initialize with pbkdf2 as primary, with bcrypt as optional fallback
try:
    # First try direct bcrypt import to check availability
    import bcrypt as bcrypt_lib
    # Test if bcrypt works
    bcrypt_lib.hashpw(b"test", bcrypt_lib.gensalt())
    
    # If we get here, bcrypt works - use it as primary
    pwd_context = CryptContext(
        schemes=["bcrypt", "pbkdf2_sha256"],
        deprecated="auto",
        bcrypt__default_rounds=12,
        pbkdf2_sha256__default_rounds=100000
    )
    BCRYPT_AVAILABLE = True
    print("SUCCESS: bcrypt initialized successfully")
    
except Exception as e:
    print(f"WARNING: bcrypt failed, using pbkdf2_sha256 as primary: {e}")
    pwd_context = CryptContext(
        schemes=["pbkdf2_sha256"],
        deprecated="auto",
        pbkdf2_sha256__default_rounds=100000,
        pbkdf2_sha256__min_rounds=50000,
        pbkdf2_sha256__max_rounds=200000
    )
    BCRYPT_AVAILABLE = False


def hash_password(password: str) -> str:
    """
    Hash a password with automatic fallback for compatibility issues.
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    try:
        # For bcrypt, handle 72-byte limit
        if BCRYPT_AVAILABLE and len(password.encode('utf-8')) > 72:
            # Use pbkdf2 for long passwords to avoid bcrypt truncation
            return pwd_context.hash(password, scheme="pbkdf2_sha256")
        else:
            return pwd_context.hash(password)
    except Exception as e:
        print(f"WARNING: Primary hashing failed, using emergency fallback: {e}")
        return emergency_hash_password(password)


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash with automatic fallback.
    """
    if not password or not hashed:
        return False
    
    try:
        # Let passlib detect the scheme automatically
        return pwd_context.verify(password, hashed)
    except Exception as e:
        print(f"WARNING: Primary verification failed, using emergency fallback: {e}")
        return emergency_verify_password(password, hashed)


def emergency_hash_password(password: str) -> str:
    """
    Emergency password hashing using built-in hashlib.
    """
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    combined = salt + ':' + base64.b64encode(password_hash).decode('ascii')
    return f"emergency:{combined}"


def emergency_verify_password(password: str, hashed: str) -> bool:
    """
    Emergency password verification using built-in hashlib.
    """
    try:
        if not hashed.startswith("emergency:"):
            return False
        
        _, combined = hashed.split(":", 1)
        salt, stored_hash = combined.split(":", 1)
        
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        password_hash_b64 = base64.b64encode(password_hash).decode('ascii')
        
        return secrets.compare_digest(stored_hash, password_hash_b64)
    except Exception:
        return False


def create_access_token(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = {"sub": str(user_id), "iat": int(datetime.utcnow().timestamp())}
    expire = datetime.utcnow() + (expires_delta or settings.ACCESS_TOKEN_EXPIRE)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        sub = payload.get("sub")
        return int(sub) if sub is not None else None
    except JWTError:
        return None
    except Exception:
        return None