import base64
import hashlib
import hmac
import json
import time
from typing import Literal

from fastapi import Depends, Header, HTTPException

from .config import DATA_DIR
from .database import get_db


SECRET_FILE = DATA_DIR / ".auth_secret"
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7


def _secret() -> bytes:
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SECRET_FILE.exists():
        SECRET_FILE.write_text(hashlib.sha256(str(time.time()).encode()).hexdigest(), encoding="utf-8")
    return SECRET_FILE.read_text(encoding="utf-8").encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def create_token(username: str, role: Literal["teacher", "student"]) -> str:
    payload = {"sub": username, "role": role, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    payload_text = _b64(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
    sig = hmac.new(_secret(), payload_text.encode(), hashlib.sha256).digest()
    return f"{payload_text}.{_b64(sig)}"


def verify_password(password: str, password_hash: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == password_hash


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录。")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload_text, sig_text = token.split(".", 1)
        expected = hmac.new(_secret(), payload_text.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(sig_text), expected):
            raise ValueError("bad signature")
        payload = json.loads(_unb64(payload_text))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
    except Exception as exc:
        raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录。") from exc

    with get_db() as conn:
        row = conn.execute("SELECT username, role, display_name FROM users WHERE username = ?", (payload.get("sub"),)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="账号不存在，请重新登录。")
    return {"username": row["username"], "role": row["role"], "name": row["display_name"]}


def require_teacher(user: dict = Depends(current_user)):
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师账号可执行该操作。")
    return user


def teacher_user(user: dict = Depends(current_user)):
    return require_teacher(user)
