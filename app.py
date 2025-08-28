import os, textwrap, requests, json, re
import streamlit as st
from neo4j import GraphDatabase

# ---------------------- Config UI ----------------------
st.set_page_config(page_title="GraphRAG (Direct)", layout="wide")
st.title("🔎 GraphRAG — Direct Neo4j + OpenAI (no neo4j-graphrag)")

with st.sidebar:
    st.header("⚙️ Settings")
    uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
    user = st.text_input("Neo4j User", "neo4j")
    pwd  = st.text_input("Neo4j Password", "devaansh", type="password")
    index_name = st.text_input("Vector Index Name", "text_embeddings")
    openai_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    embed_model = st.text_input("Embedding model", "text-embedding-3-large")  # 3072 dims
    gen_model   = st.text_input("LLM model", "gpt-4o")
    top_k = st.slider("Top-K chunks", 1, 12, 5)
    show_context = st.checkbox("Show retrieved context", True)

query = st.text_area("Question", height=120, value=
"Summarize the Branch Cash Replenishment Process: key steps, RTO/MTPD, dependencies, and controls."
)

# ---------------------- OpenAI helpers ----------------------
def _require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Provide it in the sidebar.")

def get_embedding(text: str, model: str):
    _require_openai_key()
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}",
                 "Content-Type":"application/json"},
        json={"model": model, "input": text[:12000]},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def chat_complete(system: str, user_msg: str, model: str):
    _require_openai_key()
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}",
                 "Content-Type":"application/json"},
        json={"model": model, "temperature": 0,
              "messages":[{"role":"system","content":system},
                          {"role":"user","content":user_msg}]},
        timeout=180
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

SYS_PROMPT = "Answer strictly from the provided context. Do not speculate."

# ---------------------- Neo4j helpers ----------------------
@st.cache_resource(show_spinner=False)
def get_driver(uri, user, pwd):
    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    with drv.session() as s:
        s.run("RETURN 1").single()
    return drv

# Vector search on :Chunk(embedding)
VEC_QUERY = """
CALL db.index.vector.queryNodes($index, $topK, $embedding)
YIELD node, score
RETURN elementId(node) AS id, node.file AS file, node.idx AS idx, node.text AS text, score
"""

# ---------------------- GRAPH CONTEXT (Neo4j 5 safe, fanout 1..3) ----------------------
GRAPH_CONTEXT_QUERY = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk) WHERE elementId(c)=cid

// 1) Fanout relationships excluding FROM_CHUNK (expand to 1..3 hops)
OPTIONAL MATCH pth=(c)<-[:FROM_CHUNK]-(e)-[r*1..3]-(nb)
WHERE ALL(rel IN r WHERE type(rel) <> 'FROM_CHUNK')
WITH c, apoc.coll.toSet(REDUCE(acc=[], rels IN r | acc + rels)) AS rels
WITH collect(DISTINCT c) AS chunks, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) AS rels

// 2) Process metrics adjacent to these chunks
OPTIONAL MATCH (proc:Process)-[:FROM_CHUNK]->(c2:Chunk)
WHERE c2 IN chunks AND (
    proc.rto_hours IS NOT NULL OR
    proc.mtpd_hours IS NOT NULL OR
    proc.rpo_hours IS NOT NULL
)
WITH chunks, rels, collect(DISTINCT proc) AS procs

// 3) Alternate sites (explicit matches; no pattern in WHERE)
OPTIONAL MATCH (proc2:Process)-[:ALTERNATE_LOCATION]->(site:Site)
OPTIONAL MATCH (proc2)-[:FROM_CHUNK]->(c3:Chunk)
WHERE c3 IN chunks
WITH chunks, rels, procs, collect(DISTINCT {p:proc2.name, s:site.name}) AS altSites

// 4) Call tree recipients (global; scope by chunk via subqueries if required)
OPTIONAL MATCH (ct:CallTree)-[:ISSUED_NOTIFICATION]->(no:Notification)-[hr:HAS_RECIPIENT]->(rec:Recipient)
WITH chunks, rels, procs, altSites,
     collect(DISTINCT {ct:coalesce(ct.name,'Call Tree'), no:coalesce(no.name,'Notification'),
                       recipient:rec.name, role:rec.role, ord:hr.order}) AS recs

RETURN
  apoc.text.join([c IN chunks | c.text], '\n') + '\n' +
  apoc.text.join([rel IN rels |
    coalesce(startNode(rel).name, head(labels(startNode(rel)))) + ' - ' + type(rel) +
    coalesce(' ' + coalesce(rel.details,''), '') + ' -> ' +
    coalesce(endNode(rel).name, head(labels(endNode(rel))))
  ], '\n') + '\n' +
  CASE WHEN size(procs)>0 THEN
    '\n# Process Metrics (hours)\n' +
    apoc.text.join([p IN procs |
      p.name + ': ' +
      coalesce('RTO=' + toString(p.rto_hours) + ' ', '') +
      coalesce('MTPD=' + toString(p.mtpd_hours) + ' ', '') +
      coalesce('RPO=' + toString(p.rpo_hours) + ' ', '')
    ], '\n')
  ELSE '' END + '\n' +
  CASE WHEN size(altSites)>0 THEN
    '\n# Alternate Sites\n' +
    apoc.text.join([a IN altSites | a.p + ' → ' + a.s], '\n')
  ELSE '' END + '\n' +
  CASE WHEN size(recs)>0 THEN
    '\n# Call Tree (ordered)\n' +
    apoc.text.join([r IN apoc.coll.sortMulti(recs, ['ct','no','ord']) |
      r.ct + ' / ' + r.no + ' : ' + coalesce(toString(r.ord),'') + '. ' +
      r.recipient + coalesce(' ('+coalesce(r.role,'')+')','')
    ], '\n')
  ELSE '' END
  AS info
"""

# ---------------------- PURE GRAPH FACTS (Neo4j 5 safe) ----------------------
PURE_GRAPH_QUERY = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk) WHERE elementId(c)=cid

// Entities linked to this chunk
OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
WITH c, collect(DISTINCT e) AS ents

// Relations among those entities (excluding FROM_CHUNK) + evidence
OPTIONAL MATCH (e1)-[rel]->(e2)
WHERE e1 IN ents AND e2 IN ents AND type(rel) <> 'FROM_CHUNK'
WITH c, ents, collect(DISTINCT {
  src:e1.name, typ:type(rel), dst:e2.name,
  ev_file:rel.evidence_file, ev_chunk:rel.evidence_chunk, ev_quote:rel.evidence_quote
}) AS rels

// Process metrics for processes tied to this chunk
OPTIONAL MATCH (p:Process)-[:FROM_CHUNK]->(c)
WHERE p.rto_hours IS NOT NULL OR p.mtpd_hours IS NOT NULL OR p.rpo_hours IS NOT NULL
WITH c, ents, rels,
     collect(DISTINCT {name:p.name, rto:p.rto_hours, mtpd:p.mtpd_hours, rpo:p.rpo_hours}) AS procs

// Typed dependencies for processes tied to this chunk (explicit match)
OPTIONAL MATCH (p2:Process)-[d:HAS_INTERNAL_DEPENDENCY|HAS_EXTERNAL_DEPENDENCY|DEPENDS_ON|
                                  REQUIRES_APPLICATION|REQUIRES_INFRASTRUCTURE|REQUIRES_VITAL_RECORD|
                                  SUPPLIED_BY|REQUIRES_DATASET|REQUIRES_VITAL_RECORD]->(dep),
               (p2)-[:FROM_CHUNK]->(c)
WITH c, ents, rels, procs,
     collect(DISTINCT {proc:p2.name, rel:type(d), dep:coalesce(dep.name, head(labels(dep)))}) AS deps

// Alternate sites (explicit match)
OPTIONAL MATCH (p3:Process)-[:ALTERNATE_LOCATION]->(s:Site),
               (p3)-[:FROM_CHUNK]->(c)
WITH c, ents, rels, procs, deps, collect(DISTINCT {proc:p3.name, site:s.name}) AS alts

// Call tree rows scoped to this chunk with subqueries (ct OR notification tied to chunk)
CALL {
  WITH c
  OPTIONAL MATCH (ct:CallTree)-[:ISSUED_NOTIFICATION]->(no:Notification)-[hr:HAS_RECIPIENT]->(r:Recipient),
                 (ct)-[:FROM_CHUNK]->(c)
  RETURN collect(DISTINCT {ct:coalesce(ct.name,'Call Tree'), no:coalesce(no.name,'Notification'),
                           recipient:r.name, role:r.role, ord:hr.order}) AS rows1
}
CALL {
  WITH c
  OPTIONAL MATCH (ct2:CallTree)-[:ISSUED_NOTIFICATION]->(no2:Notification)-[hr2:HAS_RECIPIENT]->(r2:Recipient),
                 (no2)-[:FROM_CHUNK]->(c)
  RETURN collect(DISTINCT {ct:coalesce(ct2.name,'Call Tree'), no:coalesce(no2.name,'Notification'),
                           recipient:r2.name, role:r2.role, ord:hr2.order}) AS rows2
}
WITH c, ents, rels, procs, deps, alts, rows1 + rows2 AS callrows

RETURN {
  chunk: elementId(c),
  entities: [x IN ents WHERE x IS NOT NULL | {name:x.name, labels:labels(x)}],
  relations: rels,
  processes: [x IN procs WHERE x.name IS NOT NULL],
  dependencies: deps,
  alternate_sites: alts,
  call_tree: callrows
} AS result
"""

# ---------------------- Retrieval helpers ----------------------
def vector_search(driver, index_name, qtext, top_k, embed_model):
    emb = get_embedding(qtext, embed_model)
    with driver.session() as s:
        rows = s.run(VEC_QUERY, index=index_name, topK=top_k, embedding=emb).data()
    return rows, [r["id"] for r in rows]

def build_vector_and_graph_context(driver, chunk_ids):
    vec_context = ""
    graph_context = ""
    if chunk_ids:
        with driver.session() as s:
            vec_rows = s.run(
                "MATCH (c:Chunk) WHERE elementId(c) IN $ids RETURN c.text AS text ORDER BY c.idx",
                ids=chunk_ids
            ).data()
            vec_context = "\n".join([r["text"] for r in vec_rows])

        with driver.session() as s:
            g = s.run(GRAPH_CONTEXT_QUERY, chunkIds=chunk_ids).single()
            graph_context = g["info"] if g and g.get("info") else vec_context
    return vec_context, graph_context

def query_pure_graph_facts(driver, chunk_ids):
    if not chunk_ids:
        return [], ""
    with driver.session() as s:
        data = s.run(PURE_GRAPH_QUERY, chunkIds=chunk_ids).data()
    # Pretty markdown and also return structured list
    md_lines = []
    for row in data:
        r = row["result"]
        md_lines.append(f"### Chunk {r['chunk']}")
        if r["processes"]:
            md_lines.append("**Processes & Metrics (hours):**")
            for p in r["processes"]:
                bits = []
                if p.get("rto") is not None:  bits.append(f"RTO={p['rto']}")
                if p.get("mtpd") is not None: bits.append(f"MTPD={p['mtpd']}")
                if p.get("rpo") is not None:  bits.append(f"RPO={p['rpo']}")
                md_lines.append(f"- {p['name']}: " + (" ".join(bits) if bits else "—"))
        if r["dependencies"]:
            md_lines.append("**Dependencies (typed):**")
            for d in r["dependencies"]:
                md_lines.append(f"- {d['proc']} - {d['rel']} -> {d['dep']}")
        if r["alternate_sites"]:
            md_lines.append("**Alternate Sites:**")
            for a in r["alternate_sites"]:
                md_lines.append(f"- {a['proc']} → {a['site']}")
        if r["call_tree"]:
            md_lines.append("**Call Tree (ordered):**")
            rows = sorted([x for x in r["call_tree"] if x.get("recipient")], key=lambda z: (z.get("ct",""), z.get("no",""), z.get("ord") or 0))
            for ctrow in rows:
                order = f"{ctrow['ord']}." if ctrow.get("ord") is not None else "-"
                role  = f" ({ctrow['role']})" if ctrow.get("role") else ""
                md_lines.append(f"- {ctrow['ct']} / {ctrow['no']} : {order} {ctrow['recipient']}{role}")
        if r["entities"]:
            md_lines.append("**Entities (in chunk):** " + ", ".join([f"{e['name']} ({'/'.join(e['labels'])})" for e in r["entities"]]))
        if r["relations"]:
            md_lines.append("**Relations (within chunk):**")
            for rel in r["relations"]:
                ev = []
                if rel.get("ev_file"): ev.append(f"file={rel['ev_file']}")
                if rel.get("ev_chunk"): ev.append(f"chunk={rel['ev_chunk']}")
                if rel.get("ev_quote"): ev.append(f"quote=\"{rel['ev_quote']}\"")
                evtxt = f"  — {'; '.join(ev)}" if ev else ""
                md_lines.append(f"- {rel['src']} - {rel['typ']} -> {rel['dst']}{evtxt}")
        md_lines.append("")
    return data, "\n".join(md_lines).strip()

# -------- Graph-only LLM context builder (no chunks, only facts) --------
def build_graph_only_context_from_results(results_list):
    """
    Convert the PURE_GRAPH_QUERY results into a compact, evidence-carrying factual context.
    """
    lines = []
    for row in results_list:
        r = row["result"]
        # Processes & metrics
        for p in r.get("processes", []):
            parts = []
            if p.get("rto") is not None:  parts.append(f"RTO={p['rto']}h")
            if p.get("mtpd") is not None: parts.append(f"MTPD={p['mtpd']}h")
            if p.get("rpo") is not None:  parts.append(f"RPO={p['rpo']}h")
            if p.get("name"):
                lines.append(f"PROCESS: {p['name']} | " + (" ".join(parts) if parts else ""))
        # Dependencies
        for d in r.get("dependencies", []):
            lines.append(f"DEPENDENCY: {d['proc']} - {d['rel']} -> {d['dep']}")
        # Alternate sites
        for a in r.get("alternate_sites", []):
            lines.append(f"ALTERNATE_SITE: {a['proc']} -> {a['site']}")
        # Call tree
        ordered = sorted([x for x in r.get("call_tree", []) if x.get("recipient")],
                         key=lambda z: (z.get("ct",""), z.get("no",""), z.get("ord") or 0))
        for ctrow in ordered:
            ordtxt = f"{ctrow['ord']}" if ctrow.get("ord") is not None else "-"
            role   = f" ({ctrow['role']})" if ctrow.get("role") else ""
            lines.append(f"CALLTREE: {ctrow['ct']} / {ctrow['no']} | {ordtxt}. {ctrow['recipient']}{role}")
        # Intra-chunk entity relations + evidence
        for rel in r.get("relations", []):
            ev = []
            if rel.get("ev_file"): ev.append(f"file={rel['ev_file']}")
            if rel.get("ev_chunk"): ev.append(f"chunk={rel['ev_chunk']}")
            if rel.get("ev_quote"): ev.append(f"quote=\"{rel['ev_quote']}\"")
            evtxt = f" | evidence: {'; '.join(ev)}" if ev else ""
            lines.append(f"REL: {rel['src']} - {rel['typ']} -> {rel['dst']}{evtxt}")
        # Entities (light)
        names = [e['name'] for e in r.get("entities", []) if e.get('name')]
        if names:
            lines.append("ENTITIES: " + ", ".join(names))
    text = "\n".join(lines).strip()
    return text if text else "NO_GRAPH_FACTS"

# --------- Simple consensus / reconciliation ----------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s:.%-]", "", s)  # keep simple punctuation
    return s

def consensus(vector_ans: str, graph_ans: str, graph_only_ans: str):
    va, ga, go = _norm(vector_ans), _norm(graph_ans), _norm(graph_only_ans)
    agree_vg  = (va == ga) or (va in ga) or (ga in va)
    agree_vgo = (va == go) or (ga == go) or (go in va) or (go in ga)
    all_same  = va == ga == go
    out = {"all_same": all_same, "agree_vg": agree_vg, "agree_any_with_graph_only": agree_vgo}
    if all_same:
        out["summary"] = "✅ All three answers agree."
    elif agree_vg and not agree_vgo:
        out["summary"] = "⚠️ Vector-only and Vector+Graph agree, but Graph-only differs (check facts/evidence)."
    elif not agree_vg and agree_vgo:
        out["summary"] = "⚠️ Graph-only matches at least one of the other answers; the remaining one differs."
    else:
        out["summary"] = "❌ Answers disagree. Inspect contexts and graph facts."
    out["answers"] = {
        "vector_only": vector_ans,
        "vector_plus_graph": graph_ans,
        "graph_only_on_facts": graph_only_ans
    }
    return out

# ---------------------- Run ----------------------
if st.button("Search", type="primary") and query.strip():
    try:
        driver = get_driver(uri, user, pwd)

        with st.spinner("Retrieving…"):
            vec_rows, chunk_ids = vector_search(driver, index_name, query, top_k, embed_model)
            vec_ctx, graph_ctx  = build_vector_and_graph_context(driver, chunk_ids)
            pure_json, pure_md  = query_pure_graph_facts(driver, chunk_ids)

        # ----- Answers with LLM (vector-only / graph-expanded / graph-only) -----
        vector_answer = chat_complete(SYS_PROMPT, textwrap.dedent(f"""
            Answer the Question using ONLY this Context:
            # Question:
            {query}

            # Context:
            {vec_ctx}

            # Answer:
        """), gen_model)

        graph_answer = chat_complete(SYS_PROMPT, textwrap.dedent(f"""
            Answer the Question using ONLY this Context:
            # Question:
            {query}

            # Context:
            {graph_ctx}

            # Answer:
        """), gen_model)

        graph_only_ctx = build_graph_only_context_from_results(pure_json)
        graph_only_answer = chat_complete(SYS_PROMPT, textwrap.dedent(f"""
            Answer the Question using ONLY this Context (facts from the graph; include only what is present; if unknown, say so):
            # Question:
            {query}

            # Graph Facts:
            {graph_only_ctx}

            # Answer:
        """), gen_model)

        # Layout: 4 columns (3 answers + consensus)
        c1, c2, c3, c4 = st.columns([1,1,1,1])

        with c1:
            st.subheader("Vector-only Answer")
            st.write(vector_answer)
            if show_context:
                with st.expander("Show vector-only context"):
                    st.code(vec_ctx)

        with c2:
            st.subheader("Vector+Graph Answer")
            st.write(graph_answer)
            if show_context:
                with st.expander("Show vector+graph context"):
                    st.code(graph_ctx)

        with c3:
            st.subheader("Graph-only Answer (LLM on facts)")
            st.write(graph_only_answer)
            with st.expander("Show graph facts (markdown)"):
                st.markdown(pure_md if pure_md else "_No graph facts._")
            with st.expander("Show graph facts (raw JSON)"):
                st.code(json.dumps([r["result"] for r in pure_json], indent=2))

        with c4:
            st.subheader("Consensus Answer")
            summary = consensus(vector_answer, graph_answer, graph_only_answer)
            st.write(summary["summary"])
            with st.expander("Show all answers compared"):
                st.json(summary["answers"])

        with st.expander("Diagnostics"):
            st.write({
                "index_name": index_name, "top_k": top_k,
                "embed_model": embed_model, "gen_model": gen_model, "neo4j": uri,
                "chunk_ids": chunk_ids
            })

    except Exception as e:
        st.error(str(e))
else:
    st.info("Set connection details, enter a question, then click **Search**.")
