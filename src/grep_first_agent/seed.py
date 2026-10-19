"""Fictional Acme Workspace data for two users. Used by tests and the demo.

Every project-scoped record carries a `project_id`. Alice's backend response deliberately includes
records from project `p7`, which is NOT in her hierarchy (the real API's hierarchy tree is mixed),
plus one violation with no project at all. The agent must hide both.
"""

from __future__ import annotations

TOKENS = {"token-alice": "alice", "token-bob": "bob"}

# Projects each user may see. Anything outside this set must never reach the model.
HIERARCHY = {"alice": ["p1", "p2", "p3", "p4"], "bob": ["p9"]}

PROJECT_MEMBERS = {"p1": ["u1", "u2"], "p2": ["u1", "u3"], "p3": ["u1", "u2"], "p4": ["u1"], "p9": ["u4", "u5"]}

SEED: dict[str, dict] = {
    "alice": {
        "profile": {"name": "Alice Park", "role": "Operations lead", "timezone": "Europe/London"},
        "users": [
            {"id": "u1", "name": "Alice Park", "email": "alice.park@acme.example"},
            {"id": "u2", "name": "Dana Reyes", "email": "dana.reyes@acme.example"},
            {"id": "u3", "name": "Sam Ito", "email": "sam.ito@acme.example"},
        ],
        "space": [
            {"id": "s1", "name": "Product", "project_ids": ["p1", "p2"]},
            {"id": "s2", "name": "Operations", "project_ids": ["p3", "p4"]},
        ],
        "project": [
            {"id": "p1", "name": "Website Relaunch", "description": "New marketing site"},
            {"id": "p2", "name": "Mobile App Beta", "description": "Closed beta of the app"},
            {"id": "p3", "name": "Operations", "description": "Day-to-day running of the office"},
            {"id": "p4", "name": "Marketing", "description": "Campaigns and brand"},
        ],
        "task": [
            {"id": "t1", "project_id": "p4", "name": "Q4 featherfootwear campaign brief", "description": "Brief for the autumn shoe launch", "status": "open", "path": "Marketing > Q4 featherfootwear campaign brief"},
            {"id": "t2", "project_id": "p1", "name": "Launch plan", "description": "Milestones for the site go-live", "status": "open", "path": "Website Relaunch > Launch plan"},
            {"id": "t3", "project_id": "p2", "name": "Launch plan", "description": "Beta rollout steps", "status": "open", "path": "Mobile App Beta > Launch plan"},
            {"id": "t4", "project_id": "p2", "name": "Fix login bug", "description": "Users logged out after 5 minutes", "status": "done", "path": "Mobile App Beta > Fix login bug"},
            {"id": "t5", "project_id": "p3", "name": "Update OPS Manual section 4", "description": "Add the new escalation steps", "status": "open", "path": "Operations > Update OPS Manual section 4"},
        ],
        "document": [
            {"id": "d1", "project_id": "p3", "name": "OPS Manual", "content": "Incidents are triaged by the operations team. On-call engineers handle escalation first. The manual is reviewed every Friday."},
            {"id": "d2", "project_id": "p4", "name": "Brand guidelines", "content": "Logo usage and colours are fixed. Tone of voice is friendly."},
            {"id": "d3", "project_id": "p7", "name": "Merger term sheet", "content": "Merger terms are confidential. Offer price is under negotiation."},
        ],
        "email": [
            {"id": "e1", "name": "Q3 numbers", "sender": "Dana Reyes", "address": "dana.reyes@acme.example", "body": "Hi, quick note on the Q3 numbers: profit margins improved by two points, mostly from lower shipping costs. Write to dana.reyes@acme.example if unsure."},
            {"id": "e2", "name": "Team lunch on Friday", "sender": "Sam Ito", "address": "sam.ito@acme.example", "body": "Pizza at 1pm, tell me if you have allergies."},
        ],
        "meeting": [
            {"id": "m1", "project_id": "p3", "name": "OPS Manual review", "notes": "Walk through section 4 changes", "attendees": "Alice, Dana"},
            {"id": "m2", "project_id": "p1", "name": "Weekly sync", "notes": "Round table", "attendees": "Everyone"},
        ],
        "group_chat": [
            {"id": "g1", "name": "Launch war room", "last_message": "Go-live moved to Thursday", "members": "Alice, Dana, Sam"},
            {"id": "g2", "name": "Ops on-call", "last_message": "Pager handover at 6pm", "members": "Alice, Dana"},
        ],
        "kpi": [
            {"id": "k1", "project_id": "p3", "name": "Incident response time", "description": "Median minutes to first response", "value": "42 min"},
            {"id": "k2", "project_id": "p1", "name": "Site conversion rate", "description": "Share of visits that sign up", "value": "3.1%"},
            {"id": "k3", "project_id": "p2", "name": "Beta crash-free sessions", "description": "Sessions without a crash", "value": "99.2%"},
        ],
        "violation": [
            {"id": "v1", "project_id": "p3", "name": "Late safety inspection", "description": "Monthly inspection overdue by 9 days", "severity": "high"},
            {"id": "v2", "project_id": "p1", "name": "Unapproved brand colour", "description": "Landing page uses an off-palette blue", "severity": "medium"},
            {"id": "v3", "project_id": "p7", "name": "Missing sign-off on supplier contract", "description": "Contract signed without legal review", "severity": "high"},
            {"id": "v4", "name": "Missing access review", "description": "No project recorded for this finding", "severity": "low"},
        ],
        "workflow": [
            {"id": "w1", "project_id": "p1", "name": "Weekly status digest", "description": "Friday summary", "source_prompt": "Every Friday summarise open tasks per project and email the team", "source_document": ""},
            {"id": "w2", "project_id": "p3", "name": "Incident escalation", "description": "Pages the on-call engineer", "source_prompt": "", "source_document": "OPS Manual"},
            {"id": "w3", "project_id": "p3", "name": "New hire onboarding", "description": "Accounts and welcome pack", "source_prompt": "Create accounts and send the welcome pack to every new hire", "source_document": ""},
            {"id": "w4", "project_id": "p4", "name": "Invoice reminder", "description": "Chases late invoices", "source_prompt": "", "source_document": ""},
            {"id": "w5", "project_id": "p4", "name": "Asset review", "description": "Checks new assets", "source_prompt": "", "source_document": "Brand guidelines"},
            {"id": "w6", "project_id": "p7", "name": "Merger data room sync", "description": "Out of hierarchy", "source_prompt": "", "source_document": ""},
        ],
    },
    "bob": {
        "profile": {"name": "Bob Lee", "role": "Corporate development", "timezone": "America/New_York"},
        "users": [
            {"id": "u4", "name": "Bob Lee", "email": "bob.lee@acme.example"},
            {"id": "u5", "name": "Legal Desk", "email": "legal@acme.example"},
        ],
        "space": [{"id": "s9", "name": "Corporate Development", "project_ids": ["p9"]}],
        "project": [{"id": "p9", "name": "Acquisition", "description": "Confidential target review"}],
        "task": [{"id": "t9", "project_id": "p9", "name": "Acquisition due diligence", "description": "Check the target's books", "status": "open", "path": "Acquisition > Acquisition due diligence"}],
        "document": [{"id": "d9", "project_id": "p9", "name": "Board memo", "content": "Draft terms for the acquisition. Offer price is under negotiation."}],
        "email": [{"id": "e9", "name": "Offer letter", "sender": "Legal", "address": "legal@acme.example", "body": "Attached is the draft offer."}],
        "meeting": [{"id": "m9", "project_id": "p9", "name": "Deal room", "notes": "Price discussion", "attendees": "Bob"}],
        "group_chat": [{"id": "g9", "name": "Deal room chat", "last_message": "Term sheet v3 is out", "members": "Bob, Legal"}],
        "kpi": [{"id": "k9", "project_id": "p9", "name": "Diligence completion", "description": "Share of checklist done", "value": "60%"}],
        "violation": [{"id": "v9", "project_id": "p9", "name": "Budget overrun", "description": "Advisor fees above plan", "severity": "medium"}],
        "workflow": [{"id": "w9", "project_id": "p9", "name": "Deal digest", "description": "Daily summary", "source_prompt": "Summarise deal room activity every day", "source_document": ""}],
    },
}
