"""Append-only local outcomes. Feedback never changes the serving artifact."""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from .contracts import ContractError, dumps, label_index, loads


class Journal:
    def __init__(self, path, record_state=False):
        self.path, self.record_state = str(path), record_state
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS decisions (
                  id TEXT PRIMARY KEY, created REAL NOT NULL, state TEXT,
                  questions TEXT NOT NULL, response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                  id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decisions(id),
                  created REAL NOT NULL, verdict TEXT NOT NULL, labels TEXT NOT NULL,
                  reviewer TEXT NOT NULL, note TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS immutable_decisions BEFORE UPDATE ON decisions BEGIN SELECT RAISE(ABORT,'append_only'); END;
                CREATE TRIGGER IF NOT EXISTS retain_decisions BEFORE DELETE ON decisions BEGIN SELECT RAISE(ABORT,'append_only'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_feedback BEFORE UPDATE ON feedback BEGIN SELECT RAISE(ABORT,'append_only'); END;
                CREATE TRIGGER IF NOT EXISTS retain_feedback BEFORE DELETE ON feedback BEGIN SELECT RAISE(ABORT,'append_only'); END;
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, state, questions, response):
        with self._connect() as db:
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)", (response["decision_id"], time.time(), dumps(state) if self.record_state else None, dumps(questions), dumps(response)))

    def feedback(self, decision_id, verdict, labels, reviewer, note=""):
        if verdict not in {"success", "failure", "unknown"} or type(reviewer) is not str or not reviewer.strip() or type(note) is not str or len(note) > 16_384:
            raise ContractError("feedback_requires_verdict_and_reviewer")
        with self._connect() as db:
            row = db.execute("SELECT questions FROM decisions WHERE id=?", (decision_id,)).fetchone()
            if row is None:
                raise ContractError("unknown_decision_id")
            qs = loads(row[0])
            if type(labels) is not dict or (verdict != "unknown" and set(labels) != set(qs)) or set(labels) - set(qs):
                raise ContractError("feedback_requires_verified_labels_for_each_field")
            for key, value in labels.items():
                label_index(qs[key], value)
            event = {"id": str(uuid.uuid4()), "decision_id": decision_id, "created": time.time(),
                     "verdict": verdict, "labels": labels, "reviewer": reviewer, "note": note}
            db.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)", (event["id"], decision_id, event["created"], verdict, dumps(labels), reviewer, note))
        return event

    def export_verified(self):
        # Latest explicit review wins for training export; all older events remain.
        # Deduplication and split leakage checks run again during training.
        with self._connect() as db:
            rows = db.execute("""SELECT d.id,d.state,f.labels,f.verdict FROM decisions d
              JOIN feedback f ON f.decision_id=d.id
              WHERE f.rowid=(SELECT MAX(f2.rowid) FROM feedback f2 WHERE f2.decision_id=d.id)
              ORDER BY f.rowid""").fetchall()
        result = []
        for ident, saved_state, target, verdict in rows:
            if saved_state is not None and verdict != "unknown":
                result.append({"id": ident, "state": loads(saved_state), "labels": loads(target)})
        return result
