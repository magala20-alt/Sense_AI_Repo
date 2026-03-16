import hashlib
import sqlite3

from ..core.config import APP_CONFIG


def _get_columns(cursor, table_name: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def ensure_users_security_columns(db_path: str | None = None) -> None:
    db_path = db_path or APP_CONFIG["db_path"]

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        columns = _get_columns(cursor, "users")

        if not columns:
            raise RuntimeError("Table 'users' was not found.")

        if "security_question" not in columns:
            cursor.execute(
                """
                ALTER TABLE users
                ADD COLUMN security_question TEXT NOT NULL DEFAULT ''
                """
            )

        if "security_answer_hash" not in columns:
            cursor.execute(
                """
                ALTER TABLE users
                ADD COLUMN security_answer_hash TEXT NOT NULL DEFAULT ''
                """
            )

        conn.commit()
        print("updated columns")
    finally:
        conn.close()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_answer(answer: str) -> str:
    normalized = answer.strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _prepare_password_for_storage(password: str) -> str:
    # Replace this if the project already hashes passwords elsewhere.
    return password


def set_security_qa(
    email: str,
    security_question: str,
    security_answer: str,
    db_path: str | None = None,
):
    db_path = db_path or APP_CONFIG["db_path"]
    ensure_users_security_columns(db_path)

    email_key = _normalize_email(email)
    question = security_question.strip()
    answer_hash = _hash_answer(security_answer)

    if not question or not security_answer.strip():
        return False, "Security question and answer are required."

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET security_question = ?, security_answer_hash = ?
            WHERE lower(email) = ?
            """,
            (question, answer_hash, email_key),
        )
        conn.commit()

        if cursor.rowcount == 0:
            return False, "Account not found."

        return True, "Security question saved."
    finally:
        conn.close()


def get_security_question(email: str, db_path: str | None = None):
    db_path = db_path or APP_CONFIG["db_path"]
    ensure_users_security_columns(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT security_question
            FROM users
            WHERE lower(email) = ?
            """,
            (_normalize_email(email),),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return row[0]
    finally:
        conn.close()


def verify_security_answer(email: str, answer: str, db_path: str | None = None) -> bool:
    db_path = db_path or APP_CONFIG["db_path"]
    ensure_users_security_columns(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT security_answer_hash
            FROM users
            WHERE lower(email) = ?
            """,
            (_normalize_email(email),),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return False

        return row[0] == _hash_answer(answer)
    finally:
        conn.close()


def update_password(email: str, new_password: str, db_path: str | None = None):
    db_path = db_path or APP_CONFIG["db_path"]

    if not new_password.strip():
        return False, "Password is required."

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET password = ?
            WHERE lower(email) = ?
            """,
            (_prepare_password_for_storage(new_password), _normalize_email(email)),
        )
        conn.commit()

        if cursor.rowcount == 0:
            return False, "Account not found."

        return True, "Password updated. Please sign in."
    finally:
        conn.close()

# if __name__ == "__main__":
#     ensure_users_security_columns()