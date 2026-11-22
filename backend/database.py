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
AUTH_SECRET = os.environ.get("SKINSCAN_AUTH_SECRET", "skinscan-clinical-secret-key-salt-2026")
VALID_ROLES = {"clinician", "patient", "researcher"}
VALID_BIOPSY_LABELS = {"mel", "nv", "bcc", "akiec", "bkl", "df", "vasc", "wound"}


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
        conn.commit()


# ─────────────────────────────────────────────────────────
# AUTHENTICATION & PASSWORD UTILITIES
# ─────────────────────────────────────────────────────────

def hash_password(password: str) -> Tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 rounds and random salt."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verify password against stored PBKDF2 hash and salt."""
    calc_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()
    return hmac.compare_digest(calc_hash, password_hash)


def create_access_token(user_id: str, email: str, role: str, expires_in_sec: int = 86400 * 30) -> str:
    """Create a tamper-proof HMAC-SHA256 bearer token."""
    import base64
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": int(time.time()) + expires_in_sec,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token signature and return payload if valid and unexpired."""
    import base64
    if not token or "." not in token:
        return None
    try:
        payload_b64, sig = token.split(".", 1)
        expected_sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        
        # Add padding back if necessary
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding
            
        payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))
        
        if payload.get("exp", 0) < time.time():
            return None
            
        return payload
    except Exception:
        return None


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
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
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
    """Authenticate user credentials and return user info."""
    clean_email = email.strip().lower()
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, name, email, role, password_hash, salt FROM users WHERE email = ?",
            (clean_email,)
        ).fetchone()

        if not row:
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
                json.dumps(clean_result),
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

