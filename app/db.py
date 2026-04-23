from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings


def _connect() -> sqlite3.Connection:
    db_path = get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                input_path TEXT NOT NULL,
                pixel_spacing_mm REAL NOT NULL,
                result_path TEXT,
                error_text TEXT,
                meta_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(
    job_id: str,
    job_type: str,
    status: str,
    input_path: str,
    pixel_spacing_mm: float,
    meta: dict[str, Any] | None = None,
) -> None:
    now = _now_iso()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, job_type, status, input_path, pixel_spacing_mm,
                result_path, error_text, meta_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                status,
                input_path,
                pixel_spacing_mm,
                json.dumps(meta or {}, ensure_ascii=True),
                now,
                now,
            ),
        )
        connection.commit()


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    result_path: str | None = None,
    error_text: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    fields: list[str] = ["updated_at = ?"]
    params: list[Any] = [_now_iso()]
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if result_path is not None:
        fields.append("result_path = ?")
        params.append(result_path)
    if error_text is not None:
        fields.append("error_text = ?")
        params.append(error_text)
    if meta is not None:
        fields.append("meta_json = ?")
        params.append(json.dumps(meta, ensure_ascii=True))
    params.append(job_id)

    query = f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?"
    with _connect() as connection:
        connection.execute(query, params)
        connection.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    payload = dict(row)
    payload["meta"] = json.loads(payload.pop("meta_json") or "{}")
    return payload
