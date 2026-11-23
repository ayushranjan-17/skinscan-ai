"""
SkinScan AI — Persistent SQLite Storage, Authentication & HITL Feedback Database
Built with Python's built-in sqlite3 and hashlib (zero external dependencies).
"""

import os
import sqlite3
import hashlib
import hmac
import secrets
import json
import time
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = os.environ.get(
    "SKINSCAN_DB_PATH",
    os.path.join(os.path.dirname(__file__), "skinscan.db")
)

# Signing secret for bearer tokens. REQUIRED in production. A missing or weak
# value is fatal: forged tokens would otherwise grant access to any user's
# scans and the MLOps export endpoints.
AUTH_SECRET = os.environ.get("SKINSCAN_AUTH_SECRET", "").strip()
_ALLOW_DEV_SECRET = os.environ.get("SKINSCAN_ALLOW_DEV_SECRET", "").lower() in ("1", "true", "yes")
if not AUTH_SECRET:
    if _ALLOW_DEV_SECRET:
        # Local development only: deterministic warning, random per-process secret so
        # tokens do not survive a restart.
        AUTH_SECRET = secrets.token_urlsafe(48)
        print(
            "[SkinScan] WARNING: SKINSCAN_AUTH_SECRET not set; using an ephemeral "
            "development secret. All issued tokens will be invalidated on restart."
        )
    else:
        raise RuntimeError(
            "SKINSCAN_AUTH_SECRET is not set. Generate one with "
            "`python -c 'import secrets; print(secrets.token_urlsafe(48))'` and export it "
            "before starting the server. For local development set SKINSCAN_ALLOW_DEV_SECRET=1 "
            "to use an ephemeral per-process secret."
        )
elif len(AUTH_SECRET) < 32:
    raise RuntimeError(
        "SKINSCAN_AUTH_SECRET is too short; use at least 32 characters of entropy."
    )

# Default token TTL. Shortened from 30 days to 8 hours; refresh on sign-in.
ACCESS_TOKEN_TTL_SECONDS = int(os.environ.get("SKINSCAN_TOKEN_TTL_SECONDS", str(8 * 3600)))

VALID_ROLES = {"clinician", "patient", "researcher"}
VALID_BIOPSY_LABELS = {"mel", "nv", "bcc", "akiec", "bkl", "df", "vasc", "wound"}

# Placeholder PBKDF2 digest used to equalize login time for unknown emails (H5).
# Computed lazily on first use.
_DUMMY_HASH: Optional[str] = None
_DUMMY_SALT: str = "0" * 32
PBKDF2_ITERATIONS = 600_000


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables for users, scans, and biopsy feedback queue."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_db_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL DEFAULT 'clinician',
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_age INTEGER,
                patient_sex TEXT,
                body_site TEXT,
                predicted_class TEXT,
                risk_stage TEXT,
                spectral_score REAL,
                malignant_probability REAL,
                icd10_code TEXT,
                dermasensor_assessment TEXT,
                result_json TEXT,
                confirmed_biopsy_label TEXT,
                queued_for_retrain INTEGER DEFAULT 0,
                clinician_notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scans_user_created 
            ON scans (user_id, created_at DESC);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scans_queued
            ON scans (queued_for_retrain) WHERE queued_for_retrain = 1;
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                revoked_at INTEGER NOT NULL,
                exp INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_revoked_tokens_exp
            ON revoked_tokens (exp);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_attempts (
                key TEXT NOT NULL,
                ts INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_auth_attempts_key_ts
            ON auth_attempts (key, ts);
        """)
        conn.commit()


# ─────────────────────────────────────────────────────────
# RATE LIMITING (sliding window, SQLite-backed)
# ─────────────────────────────────────────────────────────

def record_auth_attempt(key: str) -> int:
    """Log one auth attempt for `key` (e.g. 'login:ip:1.2.3.4' or 'login:email:a@b').

    Returns the number of attempts recorded for `key` within the last 15 minutes.
    """
    now = int(time.time())
    window_start = now - 15 * 60
    with get_db_connection() as conn:
        conn.execute("INSERT INTO auth_attempts (key, ts) VALUES (?, ?)", (key, now))
        conn.execute("DELETE FROM auth_attempts WHERE ts < ?", (window_start - 3600,))
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM auth_attempts WHERE key = ? AND ts >= ?",
            (key, window_start),
        ).fetchone()
        conn.commit()
    return int(row["n"]) if row else 0


def auth_attempt_count(key: str, window_seconds: int = 15 * 60) -> int:
    """Return attempts for `key` in the last `window_seconds` without recording a new one."""
    now = int(time.time())
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM auth_attempts WHERE key = ? AND ts >= ?",
            (key, now - window_seconds),
        ).fetchone()
    return int(row["n"]) if row else 0


# ─────────────────────────────────────────────────────────
# AUTHENTICATION & PASSWORD UTILITIES
# ─────────────────────────────────────────────────────────

def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> Tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, password_hash: str, salt: str, iterations: int = PBKDF2_ITERATIONS) -> bool:
    """Verify password against stored PBKDF2 hash and salt (constant-time compare)."""
    calc_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(calc_hash, password_hash)


def _get_dummy_hash() -> str:
    """Return a cached dummy PBKDF2 hash used to equalize login timing."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hashlib.pbkdf2_hmac(
            "sha256",
            b"dummy-password-for-timing-equalization",
            _DUMMY_SALT.encode("utf-8"),
            PBKDF2_ITERATIONS,
        ).hex()
    return _DUMMY_HASH


def create_access_token(user_id: str, email: str, role: str, expires_in_sec: Optional[int] = None) -> str:
    """Create a tamper-proof HMAC-SHA256 bearer token with a random jti for revocation."""
    import base64
    ttl = expires_in_sec if expires_in_sec is not None else ACCESS_TOKEN_TTL_SECONDS
    jti = secrets.token_hex(16)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": int(time.time()) + ttl,
        "jti": jti,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token signature and return payload if valid, unexpired, and not revoked."""
    import base64
    if not token or "." not in token:
        return None
    try:
        payload_b64, sig = token.split(".", 1)
        expected_sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None

        padding = (-len(payload_b64)) % 4
        if padding:
            payload_b64 += "=" * padding

        payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))

        if payload.get("exp", 0) < time.time():
            return None

        jti = payload.get("jti")
        if jti and is_token_revoked(jti):
            return None

        return payload
    except Exception:
        return None


def revoke_token(jti: str, exp: int) -> None:
    """Record a token's jti so later presentations are rejected (H6)."""
    if not jti:
        return
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at, exp) VALUES (?, ?, ?)",
            (jti, int(time.time()), int(exp)),
        )
        conn.commit()


def is_token_revoked(jti: str) -> bool:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?",
            (jti,),
        ).fetchone()
        return row is not None


def purge_expired_revocations() -> int:
    """Remove revocation rows whose underlying tokens are already past their exp."""
    now = int(time.time())
    with get_db_connection() as conn:
        cur = conn.execute("DELETE FROM revoked_tokens WHERE exp < ?", (now,))
        conn.commit()
        return cur.rowcount or 0


# ─────────────────────────────────────────────────────────
# USER OPERATIONS
# ─────────────────────────────────────────────────────────

def create_user(name: str, email: str, password: str, role: str = "clinician") -> Dict[str, Any]:
    """Register a new clinician or patient user."""
    clean_email = email.strip().lower()
    clean_name = name.strip()
    clean_role = role.strip().lower() if role.strip().lower() in VALID_ROLES else "clinician"

    if not clean_email or "@" not in clean_email:
        raise ValueError("Invalid email address")
    if len(password) < 10:
        raise ValueError("Password must be at least 10 characters")
    if not clean_name:
        raise ValueError("Name cannot be empty")

    pwd_hash, salt = hash_password(password)
    user_id = f"usr_{secrets.token_hex(8)}"

    with get_db_connection() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (id, name, email, role, password_hash, salt)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, clean_name, clean_email, clean_role, pwd_hash, salt)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"User with email '{clean_email}' already exists")

    return {
        "id": user_id,
        "name": clean_name,
        "email": clean_email,
        "role": clean_role
    }


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate credentials. Always runs PBKDF2 to avoid timing-based user enumeration (H5)."""
    clean_email = email.strip().lower()
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, name, email, role, password_hash, salt FROM users WHERE email = ?",
            (clean_email,)
        ).fetchone()

    if row is None:
        # Run a dummy KDF so the wall-clock cost matches a real login.
        verify_password(password, _get_dummy_hash(), _DUMMY_SALT)
        return None

    if not verify_password(password, row["password_hash"], row["salt"]):
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"],
    }


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch user profile by user_id."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, name, email, role, created_at FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)


# ─────────────────────────────────────────────────────────
# SCAN OPERATIONS & PERSISTENCE
# ─────────────────────────────────────────────────────────

def create_scan(user_id: str, scan_data: Dict[str, Any]) -> str:
    """
    Persist an inference scan result for an authenticated user.
    Strips raw base64 images to keep SQLite database lean and high-performance.
    """
    scan_id = f"scan_{secrets.token_hex(8)}"

    # Extract metadata fields
    patient_age = scan_data.get("metadata", {}).get("age") or scan_data.get("patient_age")
    patient_sex = scan_data.get("metadata", {}).get("sex") or scan_data.get("patient_sex")
    body_site = scan_data.get("metadata", {}).get("localization") or scan_data.get("body_site")
    
    predicted_class = (
        scan_data.get("name")
        or scan_data.get("shortName")
        or scan_data.get("predicted_class")
        or "Unknown"
    )
    risk_stage = str(scan_data.get("riskStage") or scan_data.get("risk_stage") or "")
    spectral_score = float(scan_data.get("spectralScore") or scan_data.get("spectral_score") or 0.0)
    
    cancer_ver = scan_data.get("cancer_verification") or {}
    mal_prob = float(cancer_ver.get("malignant_probability") or scan_data.get("malignant_probability") or 0.0)

    icd10_val = scan_data.get("icd10")
    if isinstance(icd10_val, dict):
        icd10_code = icd10_val.get("code", "")
    else:
        icd10_code = str(icd10_val or "")

    dermasensor_assessment = str(
        scan_data.get("dermasensorAssessment")
        or scan_data.get("dermasensor_assessment")
        or ""
    )

    # Clean result JSON without large base64 image strings
    clean_result = {
        k: v for k, v in scan_data.items()
        if k not in ("images", "hair_removed_image", "hair_mask_image", "gradcam_image")
    }

    # Hard cap on the persisted blob to prevent an attacker (or bug) from ballooning
    # the SQLite file with pathological payloads (M9).
    MAX_RESULT_JSON_BYTES = 256 * 1024  # 256 KiB
    result_json_str = json.dumps(clean_result)
    if len(result_json_str.encode("utf-8")) > MAX_RESULT_JSON_BYTES:
        # Fall back to a minimal shape so persistence does not fail outright.
        result_json_str = json.dumps({
            "_truncated": True,
            "predicted_class": predicted_class,
            "risk_stage": risk_stage,
            "spectral_score": spectral_score,
        })

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO scans (
                id, user_id, patient_age, patient_sex, body_site,
                predicted_class, risk_stage, spectral_score,
                malignant_probability, icd10_code, dermasensor_assessment,
                result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                user_id,
                patient_age,
                patient_sex,
                body_site,
                predicted_class,
                risk_stage,
                spectral_score,
                mal_prob,
                icd10_code,
                dermasensor_assessment,
                result_json_str,
            )
        )
        conn.commit()

    return scan_id


def get_user_scans(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all scans belonging to a user, newest first."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM scans
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

        scans = []
        for row in rows:
            d = dict(row)
            if d.get("result_json"):
                try:
                    d["result_json"] = json.loads(d["result_json"])
                except Exception:
                    pass
            scans.append(d)
        return scans


def get_scan_by_id(scan_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single scan by scan_id."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("result_json"):
            try:
                d["result_json"] = json.loads(d["result_json"])
            except Exception:
                pass
        return d


def delete_user_scan(user_id: str, scan_id: str) -> bool:
    """Delete a scan owned by the user."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM scans WHERE id = ? AND user_id = ?",
            (scan_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


# ─────────────────────────────────────────────────────────
# HITL BIOPSY FEEDBACK & MLOPS RETRAINING QUEUE
# ─────────────────────────────────────────────────────────

def update_scan_feedback(
    scan_id: str,
    confirmed_biopsy_label: str,
    clinician_notes: Optional[str] = None,
    user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Attach clinician histopathology biopsy label and queue scan for offline model fine-tuning.
    """
    clean_label = confirmed_biopsy_label.strip().lower()
    with get_db_connection() as conn:
        # Check scan existence
        if user_id:
            row = conn.execute("SELECT id FROM scans WHERE id = ? AND user_id = ?", (scan_id, user_id)).fetchone()
        else:
            row = conn.execute("SELECT id FROM scans WHERE id = ?", (scan_id,)).fetchone()

        if not row:
            return None

        conn.execute(
            """
            UPDATE scans
            SET confirmed_biopsy_label = ?,
                clinician_notes = ?,
                queued_for_retrain = 1
            WHERE id = ?
            """,
            (clean_label, clinician_notes or "", scan_id)
        )
        conn.commit()

    return get_scan_by_id(scan_id)


def get_retrain_queue(limit: int = 200) -> Dict[str, Any]:
    """Return all queued cases for offline MLOps retraining."""
    with get_db_connection() as conn:
        count_row = conn.execute(
            "SELECT COUNT(*) as count FROM scans WHERE queued_for_retrain = 1"
        ).fetchone()
        queued_count = count_row["count"] if count_row else 0

        rows = conn.execute(
            """
            SELECT id as scan_id, created_at, patient_age, patient_sex, body_site,
                   predicted_class, confirmed_biopsy_label, clinician_notes,
                   spectral_score, malignant_probability, icd10_code, dermasensor_assessment
            FROM scans
            WHERE queued_for_retrain = 1
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        cases = [dict(r) for r in rows]
        return {
            "queued_count": queued_count,
            "model_version": "v4 (frozen)",
            "cases": cases
        }


def export_retrain_manifest() -> List[Dict[str, Any]]:
    """Export an anonymized JSON manifest of verified biopsy cases for offline v5 training."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id as sample_id, created_at, patient_age, patient_sex, body_site,
                   predicted_class, confirmed_biopsy_label as ground_truth_label,
                   clinician_notes, spectral_score, malignant_probability, icd10_code
            FROM scans
            WHERE queued_for_retrain = 1 AND confirmed_biopsy_label IS NOT NULL
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]

