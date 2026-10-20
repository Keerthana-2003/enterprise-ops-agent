"""Ontology / context layer.

Resolves a free-text mention ("Fabrikam", "northwind.com") to one canonical entity,
returns the identifiers that entity has in every source system, and walks its
relationships. This is what lets the agent join data across fragmented systems.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path


class Ontology:
    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text())
        self.entity_types: dict = data["entity_types"]
        self.systems: dict = data["systems"]
        self.entities: dict[str, dict] = {e["id"]: e for e in data["entities"]}
        self.relations: list[dict] = data["relations"]

    def resolve(self, mention: str, entity_type: str | None = None, min_score: float = 0.6) -> list[dict]:
        """Return candidate entities for a mention, best first, with a match score."""
        q = mention.strip().lower()
        scored = []
        for ent in self.entities.values():
            if entity_type and ent["type"] != entity_type:
                continue
            names = [ent["name"].lower(), *ent.get("aliases", [])]
            score = max(
                1.0 if q == n else 0.9 if q in n or n in q else SequenceMatcher(None, q, n).ratio()
                for n in names
            )
            if score >= min_score:
                scored.append((score, ent))
        scored.sort(key=lambda x: -x[0])
        return [{**ent, "match_score": round(score, 2)} for score, ent in scored[:3]]

    def neighbors(self, entity_id: str) -> list[dict]:
        out = []
        for r in self.relations:
            if r["from"] == entity_id:
                out.append({"rel": r["rel"], "entity": self.entities.get(r["to"], {"id": r["to"]})["name"]})
            elif r["to"] == entity_id:
                out.append({"rel": f"INVERSE_{r['rel']}", "entity": self.entities.get(r["from"], {"id": r["from"]})["name"]})
        return out

    def schema_summary(self) -> str:
        types = "; ".join(f"{k}: {v}" for k, v in self.entity_types.items())
        systems = "; ".join(f"{k}: {v}" for k, v in self.systems.items())
        return f"Entity types: {types}\nSource systems: {systems}"
