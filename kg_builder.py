#!/usr/bin/env python3
"""
High-fidelity BCM/BCP/Risk graph builder (Docling + hybrid chunking + synonyms + evidence)

- Parses PDFs with Docling → Markdown
- Hybrid chunking preserves tables/lists so rows don't split
- Strict JSON extraction with evidence for every fact
- Synonym normalization (BIA, RTO, MTPD, RPO, Vendor, Person, etc.)
- Process metrics normalized to hours: p.rto_hours, p.mtpd_hours, p.rpo_hours
- Typed dependencies, alternate sites, full ordered call trees
- :Evidence nodes with exact quotes + provenance, linked by :SUPPORTED_BY
- :Chunk {file, idx, text, embedding} and vector index "text_embeddings" (cosine, 3072 dims)

Run: python kg_build_bcm_docling_fact_strict_syn.py
"""

import os, re, json, time, glob
from typing import List, Dict, Any, Tuple, Optional

from pathlib import Path
from neo4j import GraphDatabase
import requests
from docling.document_converter import DocumentConverter

# -------------------- CONFIG --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY environment variable.")

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "devaansh"

# >>> Put your exact files (no wildcards) here <<<
CANDIDATE_SOURCES = [
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\Regulatory Submission Continuity Plan_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\Compliance & Legal Operations Continuity & Regulatory Readiness Test_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\Compliance & Legal Regulatory Escalation Call Tree_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\REGULATORY REPORTING_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\Branch Banking Operations Continuity & Alternate Site Test_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\Branch Cash Replenishment Continuity Plan_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\BRANCH CASH REPLENISHMENT PROCESS RISK ASSESSMENT_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\BRANCH CASH REPLENISHMENT PROCESS_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\Cash Management Operations Emergency Call Tree_1.01.pdf",
]

EMBED_MODEL = "text-embedding-3-large"  # 3072 dims

# -------------------- SCHEMA --------------------
node_labels = [
    # Org & people
    "Organization","BusinessUnit","Department","Team","Role","Person",
    # Parties & authorities
    "Vendor","ThirdParty","Regulator","ComplianceRequirement","Regulation","Policy",
    # Geography & facilities
    "Location","Site","AlternateSite","Facility","Region","Country",
    # Documents & structure
    "Document","Section","Table","TableRow","Figure","Checklist","Procedure","WorkInstruction","SOP",
    # Evidence & ingestion
    "Chunk","Evidence",
    # Plans & analysis
    "BCP","BIA","RiskAssessment","RiskRegister","Strategy","Scenario","RecoveryPhase","Capability",
    # Processes & delivery
    "Process","SubProcess","Service","SubService","Activity","Product",
    # Metrics, priority & performance
    "Tier","Criticality","SLA","KPI","Capacity","WorkWindow",
    # Continuity metrics
    "RTO","RPO","MTPD",
    # Risk domain
    "Risk","Threat","Vulnerability","Control","ControlTest","ControlEffectiveness",
    "Impact","ImpactRating","Likelihood","RiskRating","ResidualRisk","Treatment","MitigationPlan",
    # Assets & dependencies
    "Application","Database","API","Dataset","VitalRecord","Record",
    "ITInfrastructure","Network","Server","VM","Storage","EndpointDevice","Tool","FacilityAsset",
    # Comms
    "CallTree","Notification","Message","Channel","Recipient","EscalationPolicy","ContactMethod",
    # Events, testing & ops
    "Exercise","TestPlan","TestReport","TestResult","Issue","IssueLog",
    "Incident","Outage","Change","Exception","Waiver","Review","Audit",
    # Governance & lifecycle
    "Owner","Approval","Approver","Status","Version","EffectivePeriod","TimePeriod","DateTag","Tag"
]

rel_types = [
    # Org structure & responsibility
    "HAS_UNIT","HAS_DEPARTMENT","HAS_TEAM","REPORTS_TO",
    "OWNS","OWNED_BY","ACCOUNTABLE_FOR","RESPONSIBLE_FOR","CONSULTED_FOR","INFORMED_ABOUT",
    # Governance & lifecycle
    "HAS_STATUS","HAS_VERSION","APPROVED_BY","EFFECTIVE_DURING","SUPERSEDED_BY","REVIEWED_BY",
    # Documentation & evidence
    "DESCRIBED_IN","HAS_SECTION","HAS_TABLE","HAS_ROW","HAS_FIGURE","FROM_CHUNK",
    "SUPPORTED_BY","CITES","DERIVED_FROM",
    # Plans & analysis
    "OWNS_BCP","OWNS_BIA","OWNS_RISK_ASSESSMENT",
    "INCLUDES_PROCESS","COVERS_SERVICE","REALIZES_CAPABILITY",
    "DEFINES_SCENARIO","USES_STRATEGY","HAS_RECOVERY_PHASE",
    # Processes & performance
    "HAS_TIER","HAS_SLA","HAS_KPI","HAS_CAPACITY","HAS_WORK_WINDOW",
    # Continuity metrics
    "SETS_RTO","SETS_RPO","SETS_MTPD","WITHIN_THRESHOLD","BEYOND_THRESHOLD",
    # Risk & control
    "IDENTIFIES_RISK","IDENTIFIES_THREAT","HAS_VULNERABILITY",
    "MITIGATED_BY","CONTROLS","HAS_CONTROL_TEST","HAS_CONTROL_EFFECTIVENESS",
    "HAS_IMPACT","HAS_LIKELIHOOD","HAS_RISK_RATING","RESULTS_IN_RESIDUAL_RISK","TREATED_BY",
    # Dependencies (typed)
    "DEPENDS_ON","HAS_INTERNAL_DEPENDENCY","HAS_EXTERNAL_DEPENDENCY",
    "REQUIRES_APPLICATION","REQUIRES_DATA","REQUIRES_DATASET","REQUIRES_VITAL_RECORD",
    "REQUIRES_INFRASTRUCTURE","REQUIRES_NETWORK","REQUIRES_FACILITY","REQUIRES_TOOL",
    "HOSTED_ON","CONNECTS_TO","SUPPLIED_BY","USES_VENDOR","INTEGRATES_WITH",
    # Assets linkage
    "BACKED_UP_BY","REPLICATES_TO","EXPORTS_TO","IMPORTS_FROM",
    # Location & facilities
    "LOCATED_AT","OPERATES_AT","PRIMARY_SITE","ALTERNATE_LOCATION","FAILS_OVER_TO","SERVES_REGION",
    # Communications & call trees
    "HAS_CALL_TREE","ISSUED_NOTIFICATION","USES_CHANNEL","HAS_RECIPIENT","ESCALATES_TO","CONTACT_FOR",
    # Events & testing
    "HAS_TEST_PLAN","HAS_TEST_REPORT","PRODUCED_TEST_RESULT","FOUND_ISSUE","HAS_ISSUE_LOG",
    "TRIGGERED_BY","RESPONDS_TO","AFFECTS_PROCESS","RELATED_TO_INCIDENT","CAUSES_OUTAGE","RESOLVED_BY",
    # Compliance
    "COMPLIES_WITH","GOVERNED_BY","MANDATED_BY","REQUIRES_REPORT_TO","ALIGNS_WITH"
]

# -------------------- SYNONYMS --------------------
NODE_SYNONYMS = {
    # Analysis & plans
    "BIA":"BIA", "Business Impact Analysis":"BIA", "Impact Analysis":"BIA", "BusinessImpactAssessment":"BIA",
    "BCP":"BCP", "Business Continuity Plan":"BCP", "Continuity Plan":"BCP",

    # Continuity metrics
    "RTO":"RTO", "Recovery Time Objective":"RTO", "RecoveryTimeObjective":"RTO", "Target Recovery Time":"RTO",
    "MTPD":"MTPD", "Maximum Tolerable Period of Disruption":"MTPD", "Max Tolerable Downtime":"MTPD",
    "RPO":"RPO", "Recovery Point Objective":"RPO", "Target Data Loss":"RPO",

    # Risk
    "Risk Assessment":"RiskAssessment","RiskAnalysis":"RiskAssessment","Threat Assessment":"RiskAssessment",
    "Threat":"Threat","Hazard":"Threat","Risk Event":"Threat",

    # Roles/people/vendors
    "Person":"Person","Individual":"Person","Employee":"Person","Staff":"Person",
    "Role":"Role","Position":"Role","Function":"Role",
    "Vendor":"Vendor","Supplier":"Vendor","Provider":"Vendor","Third Party":"Vendor","ThirdParty":"Vendor",
}

def normalize_label(label: str) -> str:
    label = (label or "").strip()
    return NODE_SYNONYMS.get(label, label)

def normalize_name(name: str) -> str:
    if not name: return name
    # keep acronyms (RTO/MTPD) as upper; title-case the rest
    if name.isupper(): return name.strip()
    return re.sub(r"\s+", " ", name.strip()).strip()

# -------------------- DOCLING --------------------
def doc_to_markdown(path: str) -> str:
    conv = DocumentConverter()
    res = conv.convert(Path(path))
    md = res.document.export_to_markdown()
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md

# -------------------- HYBRID CHUNKER --------------------
class HybridChunker:
    def __init__(self, max_chars=1800, overlap=300):
        self.max_chars = max_chars; self.overlap = overlap
    def _is_table(self, line): return "|" in line and not line.strip().startswith("```")
    def _is_sep(self, line): return re.fullmatch(r"\s*\|?\s*:?-{2,}(:?\s*\|+\s*:?-{2,})+\s*\|?\s*", line) is not None
    def _is_heading(self, line): return line.lstrip().startswith("#")
    def _is_list(self, line): return bool(re.match(r"^\s*([-*+]|\d+\.)\s+", line))
    def split(self, md: str) -> List[str]:
        lines = md.splitlines(); chunks=[]; cur=[]; cur_len=0; i=0
        def flush():
            nonlocal cur, cur_len
            if cur:
                t="\n".join(cur).strip()
                if t: chunks.append(t)
                cur=[]; cur_len=0
        while i < len(lines):
            line = lines[i]

            # TABLE
            if self._is_table(line):
                tbl=[line]; i+=1
                while i<len(lines) and (self._is_table(lines[i]) or self._is_sep(lines[i])):
                    tbl.append(lines[i]); i+=1
                tbl_text="\n".join(tbl).strip()
                if len(tbl_text) > self.max_chars:
                    # split by rows, repeat header
                    header, rows=[],[]
                    for j, tl in enumerate(tbl):
                        if self._is_sep(tl):
                            header = tbl[:j]; rows = tbl[j+1:]; break
                    if not header: header=tbl[:1]; rows=tbl[1:]
                    pack=[]; curp=header.copy()
                    for r in rows:
                        cand="\n".join(curp+[r])
                        if len(cand) > self.max_chars and len(curp)>len(header):
                            pack.append("\n".join(curp))
                            curp=header.copy()+curp[-3:]  # slight overlap
                        curp.append(r)
                    if curp: pack.append("\n".join(curp))
                    flush()
                    chunks.extend(pack)
                else:
                    if cur_len + len(tbl_text) + 2 > self.max_chars:
                        flush()
                    cur.append(tbl_text); cur_len += len(tbl_text)+1
                continue

            # LIST
            if self._is_list(line):
                block=[line]; i+=1
                while i<len(lines) and (self._is_list(lines[i]) or not lines[i].strip()):
                    block.append(lines[i]); i+=1
                txt="\n".join(block).strip()
                if cur_len + len(txt) + 2 > self.max_chars:
                    flush()
                cur.append(txt); cur_len += len(txt)+1
                continue

            # HEADING as soft boundary
            if self._is_heading(line) and cur_len > self.max_chars*0.6:
                flush()

            # NORMAL
            if cur_len + len(line) + 1 > self.max_chars:
                prev="\n".join(cur)
                chunks.append(prev)
                cur=[prev[-self.overlap:]] if prev else []
                cur_len=len("\n".join(cur))
            cur.append(line); cur_len += len(line)+1; i+=1

        if cur: chunks.append("\n".join(cur))
        return [c for c in chunks if c.strip()]

# -------------------- OPENAI --------------------
def openai_chat(messages, model="gpt-4o-mini", response_format=None, temperature=0):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
        json={"model": model, "messages": messages, "temperature": temperature,
              **({"response_format": response_format} if response_format else {})},
        timeout=180
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def get_embedding(text: str) -> List[float]:
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
        json={"model": EMBED_MODEL, "input": text[:12000]},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

# -------------------- PROMPT --------------------
PROMPT = """
You are an expert BCM/BCP/Risk graph extractor. The chunk is Markdown (tables/lists). 
Return ONLY a single JSON object with these arrays:

1) entities: [{ id, label, name, properties, evidence }]
   - label MUST be chosen from the provided label list (no new labels).
   - Normalize synonyms in your output:
     * “Business Impact Analysis”, “Impact Analysis” → label=BIA
     * “Recovery Time Objective”, “Target Recovery Time” → label=RTO
     * “Maximum Tolerable Period of Disruption”, “Max Tolerable Downtime” → label=MTPD
     * “Recovery Point Objective”, “Target Data Loss” → label=RPO
     * “Supplier”, “Provider”, “Third Party” → label=Vendor
     * “Employee”, “Staff”, “Individual” → label=Person
   - evidence: exact quote (≤300 chars) supporting the entity.

2) relations: [{ type, source_id, target_id, properties, evidence }]
   - type MUST be chosen from the provided relation type list.
   - evidence: exact quote (≤300 chars) supporting the relation.
   - If the relation is implied by a table row, quote that row.

3) process_props: [{ process, rto_hours, mtpd_hours, rpo_hours, evidence }]
   - HOURS ONLY (normalize: 2d→48, 30min→0.5, 90m→1.5).
   - If a row has Process + (RTO|MTPD|RPO), you MUST output them.
   - evidence: row text snippet.

4) process_dependencies: [{ process, type, name, evidence }]
   - Dependencies are systems, apps, vendors, portals, telecom, infra, other processes/services.
   - DO NOT include threats here (earthquake, flood, strike, pandemic, app failure are threats).
   - type ∈ {InternalDependency, ExternalDependency, Dependency, Application, ITInfrastructure, Vendor, Process, Service, SubService, Dataset, VitalRecord}.
   - evidence: row/list snippet.

5) alternate_locations: [{ process, site, evidence }]

6) call_tree: [{ call_tree, notification, recipients:[{ name, role, order }], evidence }]
   - Include ALL recipients, IN ORDER, from the table/list.
   - evidence: table/list snippet covering recipients set.

Rules:
- Use only labels/types from the provided lists.
- Keep names concise; put qualifiers in properties.
- Do NOT invent values. If missing, omit the field.
- evidence strings should be copied from the chunk.

Output JSON only.
Chunk:
{chunk}
"""

# -------------------- JSON GUARD --------------------
def safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {
        "entities": [], "relations": [], "process_props": [],
        "process_dependencies": [], "alternate_locations": [], "call_tree": []
    }

def extract_chunk(chunk: str, filename: str) -> Dict[str, Any]:
    msg = [
        {"role":"system","content":"Extract a strict JSON graph for BCM/BCP/Risk with evidence and synonyms normalized."},
        {"role":"user","content":PROMPT.replace("{chunk}", chunk[:6000])}
    ]
    content = openai_chat(msg, model="gpt-4o-mini", response_format={"type":"json_object"}, temperature=0)
    data = safe_json(content)
    for key in ["entities","relations","process_props","process_dependencies","alternate_locations","call_tree"]:
        data.setdefault(key, [])
    # clamp evidence lengths
    def clip(x): 
        if not x: return ""
        x = re.sub(r"\s+", " ", x.strip())
        return x[:300]
    for e in data["entities"]:
        e["label"] = normalize_label(e.get("label",""))
        e["name"]  = normalize_name(e.get("name",""))
        e.setdefault("properties", {}); e["properties"].setdefault("source", filename)
        e["evidence"] = clip(e.get("evidence",""))
    for r in data["relations"]:
        r["type"] = r.get("type","")
        r.setdefault("properties", {})
        r["evidence"] = clip(r.get("evidence",""))
    for p in data["process_props"]:
        p["process"] = normalize_name(p.get("process",""))
        for k in ["rto_hours","mtpd_hours","rpo_hours"]:
            if p.get(k) is not None:
                try: p[k] = float(p[k])
                except Exception: p[k] = None
        p["evidence"] = clip(p.get("evidence",""))
    for d in data["process_dependencies"]:
        d["process"] = normalize_name(d.get("process",""))
        d["type"] = (d.get("type") or "").strip()
        d["name"] = normalize_name(d.get("name",""))
        d["evidence"] = clip(d.get("evidence",""))
    for a in data["alternate_locations"]:
        a["process"] = normalize_name(a.get("process",""))
        a["site"] = normalize_name(a.get("site",""))
        a["evidence"] = clip(a.get("evidence",""))
    for c in data["call_tree"]:
        c["call_tree"] = normalize_name(c.get("call_tree",""))
        c["notification"] = normalize_name(c.get("notification",""))
        c["evidence"] = clip(c.get("evidence",""))
        for rcp in c.get("recipients", []):
            rcp["name"] = normalize_name(rcp.get("name",""))
            rcp["role"] = normalize_name(rcp.get("role",""))
            try: rcp["order"] = int(rcp.get("order")) if rcp.get("order") is not None else None
            except Exception: rcp["order"] = None
    return data

# -------------------- NEO4J HELPERS --------------------
def ensure_constraints_and_indexes(driver):
    with driver.session() as s:
        # minimal unique-ish anchors
        for lbl in set(node_labels) - {"Chunk","Evidence"}:
            s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.name IS NOT NULL")
        # vector index for :Chunk
        s.run("""
        CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS { indexConfig: { `vector.dimensions`: 3072, `vector.similarity_function`: 'cosine' } }
        """)

def merge_entity(tx, label: str, name: str, props: Dict[str,Any]) -> None:
    if not label or not name: return
    tx.run(f"MERGE (n:{label} {{name:$name}}) SET n += $props", name=name, props=props or {})

def create_evidence(tx, file: str, chunk_idx: int, quote: str, where: str) -> str:
    rec = tx.run("""
        MERGE (e:Evidence {file:$f, chunk:$i, quote:$q})
        ON CREATE SET e.where=$w, e.ts=timestamp()
        RETURN elementId(e) AS id
    """, f=file, i=chunk_idx, q=quote, w=where).single()
    return rec["id"]

def link_supported_by(tx, label: str, name: str, evidence_eid: str):
    tx.run(f"""
        MATCH (n:{label} {{name:$name}})
        MATCH (e) WHERE elementId(e)=$eid
        MERGE (n)-[:SUPPORTED_BY]->(e)
    """, name=name, eid=evidence_eid)

def merge_relation_with_evidence(tx, rtype: str, sl: str, sn: str, tl: str, tn: str, props: Dict[str,Any], file: str, chunk_idx: int, quote: str):
    # store evidence on rel properties
    tx.run(f"""
        MATCH (a:{sl} {{name:$src}}), (b:{tl} {{name:$tgt}})
        MERGE (a)-[r:{rtype}]->(b)
        SET r += $props,
            r.evidence_file = $file,
            r.evidence_chunk = $idx,
            r.evidence_quote = $quote
    """, src=sn, tgt=tn, props=props or {}, file=file, idx=chunk_idx, quote=quote)

def update_process_metrics(tx, process: str, rto: Optional[float], mtpd: Optional[float], rpo: Optional[float], file: str, chunk_idx: int, quote: str):
    tx.run("""
        MERGE (p:Process {name:$name})
        SET p.rto_hours = coalesce($rto, p.rto_hours),
            p.mtpd_hours = coalesce($mtpd, p.mtpd_hours),
            p.rpo_hours = coalesce($rpo, p.rpo_hours),
            p.metrics_last_source = $file,
            p.metrics_last_chunk  = $idx,
            p.metrics_last_quote  = $quote
    """, name=process, rto=rto, mtpd=mtpd, rpo=rpo, file=file, idx=chunk_idx, quote=quote)

def link_process_dependency(tx, process: str, dep_type: str, dep_name: str, file: str, chunk_idx: int, quote: str):
    map_tbl = {
        "InternalDependency": ("InternalDependency", "HAS_INTERNAL_DEPENDENCY"),
        "ExternalDependency": ("ExternalDependency", "HAS_EXTERNAL_DEPENDENCY"),
        "Dependency":         ("Dependency",         "DEPENDS_ON"),
        "Application":        ("Application",        "REQUIRES_APPLICATION"),
        "ITInfrastructure":   ("ITInfrastructure",   "REQUIRES_INFRASTRUCTURE"),
        "Vendor":             ("Vendor",             "SUPPLIED_BY"),
        "Process":            ("Process",            "DEPENDS_ON"),
        "Service":            ("Service",            "DEPENDS_ON"),
        "SubService":         ("SubService",         "DEPENDS_ON"),
        "Dataset":            ("Dataset",            "REQUIRES_DATASET"),
        "VitalRecord":        ("VitalRecord",        "REQUIRES_VITAL_RECORD"),
    }
    label, rel = map_tbl.get(dep_type, ("Dependency","DEPENDS_ON"))
    tx.run(f"""
        MERGE (p:Process {{name:$p}})
        MERGE (d:{label} {{name:$d}})
        MERGE (p)-[r:{rel}]->(d)
        SET r.evidence_file=$file, r.evidence_chunk=$idx, r.evidence_quote=$quote
    """, p=process, d=dep_name, file=file, idx=chunk_idx, quote=quote)

def link_alternate_location(tx, process: str, site: str, file: str, chunk_idx: int, quote: str):
    tx.run("""
        MERGE (p:Process {name:$p})
        MERGE (s:Site {name:$s})
        MERGE (p)-[r:ALTERNATE_LOCATION]->(s)
        SET r.evidence_file=$file, r.evidence_chunk=$idx, r.evidence_quote=$quote
    """, p=process, s=site, file=file, idx=chunk_idx, quote=quote)

def upsert_chunk(tx, file: str, idx: int, text: str, embedding: List[float]):
    tx.run("""
        MERGE (c:Chunk {file:$f, idx:$i})
        SET c.text=$t, c.embedding=$e
    """, f=file, i=idx, t=text, e=embedding)

def link_entity_to_chunk(tx, label: str, name: str, file: str, idx: int):
    tx.run(f"""
        MATCH (e:{label} {{name:$n}}) MATCH (c:Chunk {{file:$f, idx:$i}})
        MERGE (e)-[:FROM_CHUNK]->(c)
    """, n=name, f=file, i=idx)

def write_call_tree(tx, ct_name: str, notif: str, recipients: List[Dict[str,Any]], file: str, idx: int, quote: str):
    tx.run("MERGE (ct:CallTree {name:$n})", n=ct_name or "Call Tree")
    tx.run("""
        MATCH (ct:CallTree {name:$n})
        MERGE (no:Notification {name:$m})
        MERGE (ct)-[:ISSUED_NOTIFICATION]->(no)
        SET no.evidence_file=$f, no.evidence_chunk=$i, no.evidence_quote=$q
    """, n=ct_name or "Call Tree", m=notif or "Emergency Notification", f=file, i=idx, q=quote)
    for rec in recipients:
        nm = (rec.get("name") or "").strip()
        if not nm: continue
        role = (rec.get("role") or "").strip()
        ordv = rec.get("order")
        tx.run("""
            MERGE (r:Recipient {name:$name})
            SET r.role=$role
            WITH r
            MATCH (no:Notification {name:$notif})
            MERGE (no)-[hr:HAS_RECIPIENT]->(r)
            SET hr.order=$ord, hr.evidence_file=$f, hr.evidence_chunk=$i
        """, name=nm, role=role, notif=notif or "Emergency Notification", ord=ordv, f=file, i=idx)

# -------------------- PIPELINE --------------------
def discover_pdfs(paths: List[str]) -> List[str]:
    out=[]
    for p in paths:
        if p.lower().endswith(".pdf") and os.path.exists(p):
            out.append(p)
    # stable dedupe
    return list(dict.fromkeys(out))

def build_chunks(md: str) -> List[str]:
    return HybridChunker(max_chars=1800, overlap=300).split(md)

def process_file(driver, pdf_path: str):
    fname = os.path.basename(pdf_path)
    print(f"[parse] {fname}")
    md = doc_to_markdown(pdf_path)
    chunks = build_chunks(md)
    print(f"  - chunks: {len(chunks)}")

    # extract
    results=[]
    for i, ch in enumerate(chunks):
        try:
            data = extract_chunk(ch, fname)
        except Exception as e:
            print(f"    [warn] extraction chunk {i}: {e}")
            data = {"entities": [], "relations": [], "process_props": [],
                    "process_dependencies": [], "alternate_locations": [], "call_tree": []}
        results.append(data)
        time.sleep(0.1)

    # write
    with driver.session() as s:
        # make Chunk nodes with embeddings
        for i, ch in enumerate(chunks):
            emb = get_embedding(ch)
            s.execute_write(upsert_chunk, fname, i, ch, emb)

        # Entities + Evidence + FROM_CHUNK
        id_map: Dict[str, Tuple[str,str]] = {}
        for i, part in enumerate(results):
            for e in part.get("entities", []):
                label = normalize_label(e.get("label",""))
                name  = normalize_name(e.get("name",""))
                if not label or not name or label not in node_labels: continue
                s.execute_write(merge_entity, label, name, e.get("properties") or {})

                # evidence node
                evid = s.execute_write(create_evidence, fname, i, e.get("evidence",""), "entity")
                s.execute_write(link_supported_by, label, name, evid)
                s.execute_write(link_entity_to_chunk, label, name, fname, i)

                if e.get("id"):
                    id_map[e["id"]] = (label, name)

        # Relations (with evidence on rel props)
        for i, part in enumerate(results):
            for r in part.get("relations", []):
                rtype=r.get("type","")
                sid=r.get("source_id"); tid=r.get("target_id")
                if rtype not in rel_types or not sid or not tid: continue
                if sid not in id_map or tid not in id_map: continue
                sl, sn = id_map[sid]; tl, tn = id_map[tid]
                s.execute_write(
                    merge_relation_with_evidence,
                    rtype, sl, sn, tl, tn, r.get("properties") or {},
                    fname, i, r.get("evidence","")
                )

        # Process metrics → properties
        for i, part in enumerate(results):
            for p in part.get("process_props", []):
                proc = p.get("process")
                s.execute_write(update_process_metrics,
                                proc, p.get("rto_hours"), p.get("mtpd_hours"), p.get("rpo_hours"),
                                fname, i, p.get("evidence",""))

        # Dependencies (typed)
        for i, part in enumerate(results):
            for d in part.get("process_dependencies", []):
                if not d.get("process") or not d.get("name"): continue
                s.execute_write(link_process_dependency,
                                d["process"], d["type"], d["name"],
                                fname, i, d.get("evidence",""))

        # Alternate sites
        for i, part in enumerate(results):
            for a in part.get("alternate_locations", []):
                if not a.get("process") or not a.get("site"): continue
                s.execute_write(link_alternate_location,
                                a["process"], a["site"], fname, i, a.get("evidence",""))

        # Call trees (ordered)
        for i, part in enumerate(results):
            for ct in part.get("call_tree", []):
                s.execute_write(write_call_tree,
                                ct.get("call_tree",""), ct.get("notification",""),
                                ct.get("recipients") or [], fname, i, ct.get("evidence",""))

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as s: s.run("RETURN 1").single()
    ensure_constraints_and_indexes(driver)

    pdfs = discover_pdfs(CANDIDATE_SOURCES)
    if not pdfs:
        raise SystemExit("No PDFs found. Check CANDIDATE_SOURCES list.")
    print(f"[info] Found {len(pdfs)} PDF(s).")

    for k, pdf in enumerate(pdfs, 1):
        print(f"[{k}/{len(pdfs)}] {os.path.basename(pdf)}")
        try:
            process_file(driver, pdf)
            print("  - loaded ✓")
        except Exception as e:
            print(f"  - load failed: {e}")

    driver.close()
    print("\n[done] Ingestion complete.")

if __name__ == "__main__":
    main()
