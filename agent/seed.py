"""Creates the sample enterprise data the agent works over.

The data is deliberately fragmented, like real enterprise systems:
  * CRM (SQLite)      keys customers as integers:            1, 2, 3 ...
  * Billing (SQLite)  keys customers by an external code:    "ACCT-7781" ...
  * Support (JSON)    identifies customers by email domain:  "northwind.com" ...
The ontology (data/ontology.json) is what ties these identifiers together.

Run:  python -m agent.seed
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

CUSTOMERS = [
    # crm_id, name, tier, region, account_manager, billing_code, domain, product
    (1, "Northwind Traders", "Enterprise", "US-East", "Priya Shah", "ACCT-7781", "northwind.com", "Analytics Suite"),
    (2, "Contoso Health", "Enterprise", "US-West", "Marcus Lee", "ACCT-1043", "contosohealth.com", "Claims Automation"),
    (3, "Fabrikam Logistics", "Mid-Market", "EU", "Priya Shah", "ACCT-5520", "fabrikam.eu", "Analytics Suite"),
    (4, "Adventure Works", "SMB", "US-East", "Dana Cruz", "ACCT-3307", "adventure-works.com", "Document AI"),
    (5, "Tailspin Airlines", "Enterprise", "APAC", "Marcus Lee", "ACCT-9912", "tailspin.aero", "Claims Automation"),
    (6, "Wide World Importers", "Mid-Market", "US-West", "Dana Cruz", "ACCT-2210", "wideworld.com", "Document AI"),
]

INVOICES = [
    # invoice_id, billing_code, amount_usd, status, due_date
    ("INV-1001", "ACCT-7781", 48000, "paid", "2026-07-01"),
    ("INV-1002", "ACCT-7781", 48000, "overdue", "2026-09-01"),
    ("INV-1003", "ACCT-1043", 125000, "paid", "2026-08-15"),
    ("INV-1004", "ACCT-5520", 22000, "overdue", "2026-08-20"),
    ("INV-1005", "ACCT-5520", 22000, "overdue", "2026-09-20"),
    ("INV-1006", "ACCT-3307", 6000, "paid", "2026-09-05"),
    ("INV-1007", "ACCT-9912", 98000, "open", "2026-10-30"),
    ("INV-1008", "ACCT-2210", 15000, "paid", "2026-09-12"),
]

TICKETS = [
    {"ticket_id": "T-501", "domain": "northwind.com", "priority": "P1", "status": "open",
     "subject": "Dashboard exports failing for finance team", "opened": "2026-10-08"},
    {"ticket_id": "T-502", "domain": "northwind.com", "priority": "P3", "status": "closed",
     "subject": "SSO login question", "opened": "2026-09-14"},
    {"ticket_id": "T-503", "domain": "fabrikam.eu", "priority": "P2", "status": "open",
     "subject": "Data sync delayed by 6 hours", "opened": "2026-10-10"},
    {"ticket_id": "T-504", "domain": "fabrikam.eu", "priority": "P1", "status": "open",
     "subject": "Reports missing EU warehouse data", "opened": "2026-10-12"},
    {"ticket_id": "T-505", "domain": "tailspin.aero", "priority": "P2", "status": "open",
     "subject": "Claims model flagging too many false positives", "opened": "2026-10-11"},
    {"ticket_id": "T-506", "domain": "contosohealth.com", "priority": "P3", "status": "closed",
     "subject": "Request for new user seats", "opened": "2026-09-30"},
]

DOCS = {
    "escalation_policy.md": """# Escalation Policy
Any Enterprise customer with an open P1 ticket must be escalated to the account manager within 4 hours.
If a customer has an open P1 ticket and an overdue invoice, do not send collections notices; escalate to the account manager first.
Mid-Market customers with two or more open tickets are reviewed weekly by the support lead.
""",
    "refund_policy.md": """# Refund and Credit Policy
Service credits are capped at 10% of the customer's most recent invoice amount.
Credits require account manager approval and cannot be issued on overdue invoices.
SMB customers are eligible for credits only after a P1 incident lasting longer than 24 hours.
""",
    "sla.md": """# Service Level Agreement
P1 tickets: first response within 1 hour, resolution target 24 hours.
P2 tickets: first response within 4 hours, resolution target 3 business days.
P3 tickets: first response within 1 business day.
Enterprise customers receive 99.9% uptime; Mid-Market and SMB receive 99.5%.
""",
    "collections_policy.md": """# Collections Policy
Invoices more than 30 days overdue trigger a collections notice.
Customers with two or more overdue invoices are placed on credit hold pending finance review.
""",
}


def build_ontology() -> dict:
    """Entity and relationship graph that resolves identifiers across systems."""
    entities, relations = [], []
    for crm_id, name, tier, region, am, code, domain, product in CUSTOMERS:
        cid = f"customer:{crm_id}"
        entities.append({
            "id": cid, "type": "Customer", "name": name,
            "aliases": [name.lower(), name.split()[0].lower(), domain],
            "identifiers": {"crm": crm_id, "billing": code, "support_domain": domain},
            "attributes": {"tier": tier, "region": region},
        })
        relations.append({"from": cid, "rel": "MANAGED_BY", "to": f"person:{am}"})
        relations.append({"from": cid, "rel": "SUBSCRIBES_TO", "to": f"product:{product}"})
    for am in sorted({c[4] for c in CUSTOMERS}):
        entities.append({"id": f"person:{am}", "type": "AccountManager", "name": am, "aliases": [am.lower()],
                         "identifiers": {}, "attributes": {}})
    for product in sorted({c[7] for c in CUSTOMERS}):
        entities.append({"id": f"product:{product}", "type": "Product", "name": product,
                         "aliases": [product.lower()], "identifiers": {}, "attributes": {}})
    return {
        "entity_types": {
            "Customer": "A company that buys our products. Known by different keys in each system.",
            "AccountManager": "Employee who owns the customer relationship.",
            "Product": "A product line a customer subscribes to.",
        },
        "systems": {
            "crm": "SQLite table customers(id, name, tier, region, account_manager)",
            "billing": "SQLite table invoices(invoice_id, billing_code, amount_usd, status, due_date)",
            "support": "JSON tickets keyed by customer email domain",
        },
        "entities": entities,
        "relations": relations,
    }


def seed(data_dir: Path = DATA) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "docs").mkdir(exist_ok=True)

    for name, sql_create, rows, sql_insert in [
        ("crm.db",
         "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, tier TEXT, region TEXT, account_manager TEXT)",
         [(c[0], c[1], c[2], c[3], c[4]) for c in CUSTOMERS],
         "INSERT INTO customers VALUES (?,?,?,?,?)"),
        ("billing.db",
         "CREATE TABLE invoices (invoice_id TEXT PRIMARY KEY, billing_code TEXT, amount_usd REAL, status TEXT, due_date TEXT)",
         INVOICES,
         "INSERT INTO invoices VALUES (?,?,?,?,?)"),
    ]:
        path = data_dir / name
        path.unlink(missing_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(sql_create)
            conn.executemany(sql_insert, rows)

    (data_dir / "tickets.json").write_text(json.dumps(TICKETS, indent=2))
    (data_dir / "ontology.json").write_text(json.dumps(build_ontology(), indent=2))
    for fname, text in DOCS.items():
        (data_dir / "docs" / fname).write_text(text)
    return data_dir


if __name__ == "__main__":
    print(f"Seeded sample enterprise data in {seed()}")
