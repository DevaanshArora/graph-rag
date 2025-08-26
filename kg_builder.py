#!/usr/bin/env python3
"""
BCM/BCP/Risk KG builder -> Neo4j with Docling parsing + hybrid chunking
- Preserves tables/lists/sections so metrics & call trees aren't split
- Extracts: entities/relations + Process metrics as properties (rto_hours, mtpd_hours, rpo_hours)
- Extracts: ordered Call Tree recipients (HAS_RECIPIENT {order})
- Creates :Chunk {file, idx, text, embedding} + [:FROM_CHUNK]
- Vector index: text_embeddings (cosine, 3072 dims)
"""

import os, re, json, time, glob
from typing import List, Dict, Any, Tuple

from neo4j import GraphDatabase
import requests
from docling.document_converter import DocumentConverter
from pathlib import Path

# ------------ CONFIG ------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY env var.")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "devaansh"

CANDIDATE_SOURCES = [
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\Regulatory Submission Continuity Plan_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\Compliance & Legal Operations Continuity & Regulatory Readiness Test_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\Compliance & Legal Regulatory Escalation Call Tree_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Compliance & Legal_Statutory Compliance_Reporting & Analytics_Aluva_India\REGULATORY REPORTING_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\Branch Banking Operations Continuity & Alternate Site Test_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\Branch Cash Replenishment Continuity Plan_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\BRANCH CASH REPLENISHMENT PROCESS RISK ASSESSMENT_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\BRANCH CASH REPLENISHMENT PROCESS_1.01.pdf",
    r"C:\Users\DevaanshArora\Downloads\AutoBCM_Mashreq\AutoBCM_Mashreq\BCM_Branch Banking_Cash Management_Treasury Operations_Thane_India\Cash Management Operations Emergency Call Tree_1.01.pdf"
]


# ------------ SCHEMA ------------
NODE_LABELS = [
    "Organization","Group","Division","BusinessUnit","Site","Location","Status","Version","Approval",
    "BCP","BIA","RiskAssessment","Scenario","Strategy","RecoveryPhase","Instruction",
    "Process","Product","SubProduct","Service","SubService",
    "TestPlan","TestReport","TestResult","IssueLog",
    "Person","Role","CallTree","Notification","Recipient",
    "RTO","MTPD","RPO","Impact","ImpactRating","Likelihood","RiskRating","ResidualRisk",
    "Dependency","InternalDependency","ExternalDependency","Vendor",
    "Application","ITInfrastructure","VitalRecord","Record","ResourceRequirement","SpecialRequirement",
    "Threat","Control","ControlEffectiveness"
]

REL_TYPES = [
    "BELONGS_TO","HAS_GROUP","HAS_DIVISION","HAS_BUSINESS_UNIT",
    "OWNS_BCP","OWNS_BIA","OWNS_RISK_ASSESSMENT","INCLUDES_PROCESS","DEFINES_SCENARIO","USES_STRATEGY","HAS_RECOVERY_PHASE",
    "HAS_TEST_PLAN","HAS_TEST_REPORT","PRODUCED_TEST_RESULT","HAS_ISSUE_LOG","PASSED","FAILED",
    "HAS_ROLE","ASSIGNED_TO","HAS_CALL_TREE","ISSUED_NOTIFICATION","HAS_RECIPIENT","CONTACT_FOR",
    "LOCATED_AT","ALTERNATE_LOCATION",
    "DEPENDS_ON","HAS_INTERNAL_DEPENDENCY","HAS_EXTERNAL_DEPENDENCY","SUPPLIED_BY",
    "REQUIRES_APPLICATION","REQUIRES_INFRASTRUCTURE","REQUIRES_VITAL_RECORD",
    "HAS_RESOURCE_REQUIREMENT","HAS_SPECIAL_REQUIREMENT",
    "SETS_RTO","SETS_MTPD","SETS_RPO","HAS_IMPACT","HAS_LIKELIHOOD","HAS_RISK_RATING","RESULTS_IN_RESIDUAL_RISK",
    "IDENTIFIES_THREAT","MITIGATED_BY","HAS_CONTROL_EFFECTIVENESS","WITHIN_THRESHOLD","BEYOND_THRESHOLD",
    "HAS_VERSION","HAS_STATUS","APPROVED_BY","INSTRUCTS","RESPONSIBLE_FOR"
]

# ------------ DOC LING ------------
def docling_read_markdown(path: str) -> str:
    conv = DocumentConverter()
    res = conv.convert(Path(path))
    md = res.document.export_to_markdown()
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md

# ------------ HYBRID CHUNKER ------------
class HybridChunker:
    """
    Structure-aware chunker for Docling Markdown:
    - Keeps TABLES intact (never splits a row)
    - Keeps LIST blocks intact
    - Respects HEADINGS (#, ##, ###) as soft boundaries
    - Falls back to size-based splitting for very large blocks
    """
    def __init__(self, max_chars=1600, overlap=300):
        self.max_chars = max_chars
        self.overlap = overlap

    @staticmethod
    def _is_table_line(line: str) -> bool:
        # consider as part of a table if it has pipes and isn't just code
        return "|" in line and not line.strip().startswith("```")

    @staticmethod
    def _is_table_sep(line: str) -> bool:
        # typical markdown table separator: |---|---|
        return re.fullmatch(r"\s*\|?\s*:?-{2,}(:?\s*\|+\s*:?-{2,})+\s*\|?\s*", line) is not None

    @staticmethod
    def _is_heading(line: str) -> bool:
        return line.lstrip().startswith("#")

    @staticmethod
    def _is_list(line: str) -> bool:
        return bool(re.match(r"^\s*([-*+]|\d+\.)\s+", line))

    def _flush(self, chunks: List[str], cur: List[str]):
        if cur:
            text = "\n".join(cur).strip()
            if text:
                chunks.append(text)
            cur.clear()

    def split(self, md: str) -> List[str]:
        lines = md.splitlines()
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # ----- TABLE BLOCK -----
            if self._is_table_line(line):
                # capture full contiguous table (header, sep, rows)
                table_lines = [line]
                i += 1
                while i < len(lines) and (self._is_table_line(lines[i]) or self._is_table_sep(lines[i])):
                    table_lines.append(lines[i]); i += 1

                table_text = "\n".join(table_lines).strip()

                # If table too large, split by ROWS without breaking a row
                if len(table_text) > self.max_chars:
                    # keep header + separator together
                    header = []
                    rows = []
                    # identify header+sep
                    for j, tl in enumerate(table_lines):
                        if self._is_table_sep(tl):
                            header = table_lines[:j]  # header(s) up to sep
                            rows = table_lines[j+1:]  # remaining rows
                            break
                    if not header:
                        header = table_lines[:1]; rows = table_lines[1:]

                    # pack rows into chunks with header repeated, respecting max_chars
                    pack: List[str] = []
                    cur_pack = header.copy()
                    for r in rows:
                        candidate = "\n".join(cur_pack + [r])
                        if len(candidate) > self.max_chars and len(cur_pack) > len(header):
                            pack.append("\n".join(cur_pack))
                            # overlap a few last rows to keep continuity
                            overlap_rows = cur_pack[-3:] if len(cur_pack) > 3 else cur_pack
                            cur_pack = header.copy() + overlap_rows[len(header):]
                        cur_pack.append(r)
                    if cur_pack:
                        pack.append("\n".join(cur_pack))
                    for p in pack:
                        self._flush(chunks, cur)
                        chunks.append(p)
                else:
                    # add entire table as a single unit
                    if cur_len + len(table_text) + 2 > self.max_chars:
                        # create overlapping boundary
                        if cur:
                            # overlap tail of current chunk
                            tail = "\n".join(cur)[-self.overlap:]
                            chunks.append("\n".join(cur))
                            cur = [tail] if tail else []
                            cur_len = len("\n".join(cur))
                    cur.append(table_text); cur_len += len(table_text) + 1
                continue  # continue loop without i += 1 (already moved)

            # ----- LIST BLOCK -----
            if self._is_list(line):
                list_block = [line]
                i += 1
                while i < len(lines) and (self._is_list(lines[i]) or not lines[i].strip()):
                    list_block.append(lines[i]); i += 1
                list_text = "\n".join(list_block).strip()
                if cur_len + len(list_text) + 2 > self.max_chars:
                    self._flush(chunks, cur); cur_len = 0
                cur.append(list_text); cur_len += len(list_text) + 1
                continue

            # ----- HEADING (soft boundary) -----
            if self._is_heading(line) and cur_len > self.max_chars * 0.6:
                # close current chunk at heading to keep sections coherent
                self._flush(chunks, cur); cur_len = 0

            # ----- NORMAL LINE -----
            if cur_len + len(line) + 1 > self.max_chars:
                # finalize current chunk with overlap
                text = "\n".join(cur)
                chunks.append(text)
                cur = [text[-self.overlap:]] if text else []
                cur_len = len("\n".join(cur))
            cur.append(line); cur_len += len(line) + 1
            i += 1

        self._flush(chunks, cur)
        return [c for c in chunks if c.strip()]

# ------------ OPENAI ------------
EMBED_MODEL = "text-embedding-3-large"

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

# ------------ PROMPT (same as props version) ------------
PROMPT = f"""
You are an expert BCM/BCP/Risk graph extractor. The chunk is Markdown and may include tables.
Extract ONLY facts explicitly supported by the text. Output a single JSON object with FOUR arrays:

1) "entities": nodes using ONLY labels:
   OneOf({", ".join(NODE_LABELS)})

2) "relations": edges using ONLY types:
   OneOf({", ".join(REL_TYPES)})

3) "process_props": numeric metrics from tables/lines (hours):
   [{{"process":"...", "rto_hours":2, "mtpd_hours":4, "rpo_hours":1}}]

4) "call_tree": all recipients IN ORDER:
   [{{"call_tree":"...", "notification":"...", "recipients":[{{"name":"...", "role":"...", "order":1}}]}}]

Normalize time to HOURS (2d→48). Do NOT invent values. Return ONLY JSON.

Chunk:
{{chunk}}
"""

def safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {"entities": [], "relations": [], "process_props": [], "call_tree": []}

def extract_chunk(chunk: str, filename: str) -> Dict[str, Any]:
    msgs = [
        {"role":"system","content":"You extract BCM/BCP/Risk knowledge graphs as strict JSON."},
        {"role":"user","content":PROMPT.replace("{chunk}", chunk[:6000])}
    ]
    content = openai_chat(msgs, model="gpt-4o-mini", response_format={"type":"json_object"}, temperature=0)
    data = safe_json(content)
    for e in data.get("entities", []):
        e.setdefault("properties", {})
        e["properties"].setdefault("source", filename)
        e["properties"]["description"] = (e["properties"].get("description") or "")[:160]
    for r in data.get("relations", []):
        r.setdefault("properties", {})
    data.setdefault("process_props", []); data.setdefault("call_tree", [])
    return data

# ------------ Neo4j helpers ------------
def ensure_constraints_and_indexes(driver):
    numericish = {"RTO","MTPD","RPO","Impact","ImpactRating","Likelihood","RiskRating","ResidualRisk"}
    with driver.session() as s:
        for label in NODE_LABELS:
            if label not in numericish:
                s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS NOT NULL")
        s.run("""
        CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS { indexConfig: { `vector.dimensions`: 3072, `vector.similarity_function`: 'cosine' } }
        """)

def merge_entity(tx, label: str, name: str, props: Dict[str, Any]):
    tx.run(f"MERGE (n:{label} {{name:$name}}) SET n += $props", name=name, props=props)

def merge_relation(tx, rel_type: str, sl: str, sn: str, tl: str, tn: str, props: Dict[str, Any]):
    tx.run(
        f"MATCH (a:{sl} {{name:$src}}), (b:{tl} {{name:$tgt}}) "
        f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
        src=sn, tgt=tn, props=props
    )

def update_process_metrics(tx, process: str, rto, mtpd, rpo):
    tx.run("""
        MERGE (p:Process {name:$name})
        SET p.rto_hours = coalesce($rto, p.rto_hours),
            p.mtpd_hours = coalesce($mtpd, p.mtpd_hours),
            p.rpo_hours = coalesce($rpo, p.rpo_hours)
    """, name=process, rto=rto, mtpd=mtpd, rpo=rpo)

def merge_chunk_and_links(tx, file: str, idx: int, text: str,
                          entities_for_chunk: List[Tuple[str,str]],
                          embedding: List[float]):
    tx.run("MERGE (c:Chunk {file:$file, idx:$idx}) SET c.text=$text, c.embedding=$embedding",
           file=file, idx=idx, text=text, embedding=embedding)
    for (label, name) in entities_for_chunk:
        tx.run(
            f"MATCH (e:{label} {{name:$name}}) MATCH (c:Chunk {{file:$file, idx:$idx}}) "
            "MERGE (e)-[:FROM_CHUNK]->(c)",
            name=name, file=file, idx=idx
        )

def write_call_tree(tx, ct_name: str, notif: str, recipients: List[Dict[str,Any]]):
    tx.run("MERGE (ct:CallTree {name:$n})", n=ct_name or "Call Tree")
    tx.run("MERGE (ct:CallTree {name:$n})-[:ISSUED_NOTIFICATION]->(no:Notification {name:$m})",
           n=ct_name or "Call Tree", m=notif or "Emergency Notification")
    for rec in recipients:
        name = (rec.get("name") or "").strip()
        role = rec.get("role","")
        order = rec.get("order", None)
        if not name: continue
        tx.run("""
            MERGE (r:Recipient {name:$name})
            SET r.role = $role
            WITH r
            MATCH (ct:CallTree {name:$ct}), (no:Notification {name:$no})
            MERGE (no)-[hr:HAS_RECIPIENT]->(r)
            SET hr.order = $ord
        """, name=name, role=role, ord=order, ct=ct_name or "Call Tree", no=notif or "Emergency Notification")

def load_to_neo4j(driver, filename: str, chunk_results: List[Dict[str, Any]], chunks_text: List[str]):
    id_map: Dict[str, Tuple[str,str]] = {}
    entities_accum: List[Tuple[str,str,Dict[str,Any]]] = []
    per_chunk_entities: List[List[Tuple[str,str]]] = [[] for _ in range(len(chunk_results))]
    process_props_accum: List[Tuple[str,Any,Any,Any]] = []
    calltree_accum: List[Tuple[str,str,List[Dict[str,Any]]]] = []

    for ci, part in enumerate(chunk_results):
        for e in part.get("entities", []):
            label = (e.get("label") or "").strip()
            name  = (e.get("name") or "").strip()
            if not label or not name or label not in NODE_LABELS: continue
            entities_accum.append((label, name[:512], e.get("properties") or {}))
            per_chunk_entities[ci].append((label, name[:512]))
            if e.get("id"): id_map[e["id"]] = (label, name[:512])

        for pr in part.get("process_props", []):
            process = (pr.get("process") or "").strip()
            if not process: continue
            process_props_accum.append((
                process[:512], pr.get("rto_hours"), pr.get("mtpd_hours"), pr.get("rpo_hours")
            ))

        for ct in part.get("call_tree", []):
            calltree_accum.append((ct.get("call_tree",""), ct.get("notification",""), ct.get("recipients") or []))

    with driver.session() as s:
        for (label, name, props) in entities_accum:
            s.execute_write(merge_entity, label, name, props)

    with driver.session() as s:
        for part in chunk_results:
            for r in part.get("relations", []):
                rtype = r.get("type"); sid, tid = r.get("source_id"), r.get("target_id")
                if rtype not in REL_TYPES or not sid or not tid: continue
                if sid not in id_map or tid not in id_map: continue
                (sl, sn) = id_map[sid]; (tl, tn) = id_map[tid]
                s.execute_write(merge_relation, rtype, sl, sn, tl, tn, r.get("properties") or {})

    with driver.session() as s:
        for (proc, rto, mtpd, rpo) in process_props_accum:
            s.execute_write(update_process_metrics, proc, rto, mtpd, rpo)

    with driver.session() as s:
        for idx, text in enumerate(chunks_text):
            if not text.strip(): continue
            emb = get_embedding(text)
            s.execute_write(merge_chunk_and_links, filename, idx, text, per_chunk_entities[idx], emb)

    with driver.session() as s:
        for (ct_name, notif, recs) in calltree_accum:
            s.execute_write(write_call_tree, ct_name, notif, recs)

# ------------ Discovery ------------
def discover_pdfs(items: List[str]) -> List[str]:
    out = []
    for x in items:
        if "*" in x or "?" in x:
            out.extend(glob.glob(x, recursive=True))
        elif os.path.isdir(x):
            out.extend(glob.glob(os.path.join(x, "**/*.pdf"), recursive=True))
        elif x.lower().endswith(".pdf") and os.path.exists(x):
            out.append(x)
    seen, res = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); res.append(p)
    return res

# ------------ MAIN ------------
def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as s: s.run("RETURN 1").single()
    ensure_constraints_and_indexes(driver)

    pdfs = discover_pdfs(CANDIDATE_SOURCES)
    if not pdfs: raise SystemExit("No PDFs found.")

    chunker = HybridChunker(max_chars=1600, overlap=300)

    print(f"[info] Found {len(pdfs)} PDF(s).")
    for i, pdf in enumerate(pdfs, 1):
        fname = os.path.basename(pdf)
        print(f"[{i}/{len(pdfs)}] {fname}")
        try:
            md = docling_read_markdown(pdf)
        except Exception as e:
            print(f"  [parse error] {e}"); continue

        chunks = chunker.split(md)
        print(f"  - chunks: {len(chunks)}")
        chunk_results = []
        for j, ch in enumerate(chunks, 1):
            try:
                data = extract_chunk(ch, fname)
                if not isinstance(data, dict):
                    data = {"entities": [], "relations": [], "process_props": [], "call_tree": []}
                ents = [e for e in data.get("entities", []) if e.get("label") in NODE_LABELS and e.get("name")]
                rels = [r for r in data.get("relations", []) if r.get("type") in REL_TYPES]
                chunk_results.append({
                    "entities": ents[:60],
                    "relations": rels[:120],
                    "process_props": data.get("process_props", [])[:40],
                    "call_tree": data.get("call_tree", [])[:15],
                })
            except Exception as e:
                print(f"    [warn] chunk {j}: {e}")
            time.sleep(0.12)

        try:
            load_to_neo4j(driver, fname, chunk_results, chunks_text=chunks)
            print("  - loaded ✓")
        except Exception as e:
            print(f"  - load failed: {e}")

    driver.close()
    print("\n[done] Ingestion complete.")

if __name__ == "__main__":
    main()
