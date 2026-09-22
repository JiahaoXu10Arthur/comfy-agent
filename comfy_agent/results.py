"""Every render and every verdict, one sqlite file. history reads this, never renderdb."""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS render(
  render_id TEXT PRIMARY KEY, arm TEXT, run INTEGER, task_id INTEGER, n INTEGER,
  factors TEXT, claim TEXT, png TEXT, seconds REAL,
  scores TEXT, verdict TEXT, stop_reason TEXT, missing TEXT);
CREATE INDEX IF NOT EXISTS i_scope ON render(arm, run, task_id, n);
"""


class Results:
    def __init__(self, path):
        self.db = sqlite3.connect(str(Path(path)))
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def add_render(self, render_id, arm, run, task_id, n, factors, claim, png, seconds):
        self.db.execute("INSERT INTO render(render_id,arm,run,task_id,n,factors,claim,png,seconds) VALUES(?,?,?,?,?,?,?,?,?)",
                        (render_id, arm, run, task_id, n, json.dumps(factors), claim, png, seconds))
        self.db.commit()

    def add_tags(self, render_id, scores):
        self.db.execute("UPDATE render SET scores=? WHERE render_id=?", (json.dumps(scores), render_id))
        self.db.commit()

    def set_conclusion(self, render_id, verdict, stop_reason, missing):
        self.db.execute("UPDATE render SET verdict=?, stop_reason=?, missing=? WHERE render_id=?",
                        (verdict, stop_reason, missing, render_id))
        self.db.commit()

    def _row(self, r):
        cols = ["render_id", "arm", "run", "task_id", "n", "factors", "claim", "png", "seconds",
                "scores", "verdict", "stop_reason", "missing"]
        d = dict(zip(cols, r))
        d["factors"] = json.loads(d["factors"] or "[]")
        d["scores"] = {k: tuple(v) for k, v in json.loads(d["scores"] or "{}").items()}
        d["all_landed"] = bool(d["scores"]) and all(v[0] == "LANDED" for v in d["scores"].values())
        d["min_score"] = min((v[1] for v in d["scores"].values()), default=None)
        return d

    def get(self, render_id):
        r = self.db.execute("SELECT * FROM render WHERE render_id=?", (render_id,)).fetchone()
        return self._row(r) if r else None

    def renders(self, arm, run, task_id):
        rows = self.db.execute("SELECT * FROM render WHERE arm=? AND run=? AND task_id=? ORDER BY n",
                               (arm, run, task_id)).fetchall()
        return [self._row(r) for r in rows]

    def history(self, arm, run, exclude_task, query, limit=20):
        rows = self.db.execute("SELECT * FROM render WHERE arm=? AND run=? AND task_id<>? ORDER BY task_id, n",
                               (arm, run, exclude_task)).fetchall()
        out = []
        q = query.lower()
        for r in map(self._row, rows):
            hay = " ".join(r["factors"] + list(r["scores"])).lower()
            if q in hay:
                out.append(r)
            if len(out) >= limit:
                break
        return out
