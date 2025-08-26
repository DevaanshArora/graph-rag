import os, textwrap, requests
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
def get_embedding(text: str, model: str):
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

# Graph fan-out from retrieved chunks (exclude FROM_CHUNK inside expansion)
GRAPH_CONTEXT_QUERY = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk) WHERE elementId(c)=cid
OPTIONAL MATCH p=(c)<-[:FROM_CHUNK]-(e)-[r*1..2]-(nb)
WHERE ALL(rel IN r WHERE type(rel) <> 'FROM_CHUNK')
WITH c, apoc.coll.toSet(REDUCE(acc=[], rels IN r | acc + rels)) AS rels
WITH collect(DISTINCT c) AS chunks, apoc.coll.flatten(collect(rels)) AS rels_flat
WITH chunks, apoc.coll.toSet(rels_flat) AS rels
RETURN
  apoc.text.join([c IN chunks | c.text], '\n') + '\n' +
  apoc.text.join([rel IN rels |
    coalesce(startNode(rel).name, head(labels(startNode(rel)))) + ' - ' + type(rel) +
    coalesce(' ' + rel.details, '') + ' -> ' +
    coalesce(endNode(rel).name, head(labels(endNode(rel))))
  ], '\n') AS info
"""

def vector_search_and_context(driver, index_name, qtext, top_k, embed_model):
    emb = get_embedding(qtext, embed_model)
    with driver.session() as s:
        rows = s.run(VEC_QUERY, index=index_name, topK=top_k, embedding=emb).data()
    chunk_ids = [r["id"] for r in rows]
    vec_context = "\n".join([r["text"] for r in rows])

    # graph expansion
    with driver.session() as s:
        g = s.run(GRAPH_CONTEXT_QUERY, chunkIds=chunk_ids).single()
    graph_context = g["info"] if g else vec_context
    return vec_context, graph_context

# ---------------------- Run ----------------------
if st.button("Search", type="primary") and query.strip():
    try:
        driver = get_driver(uri, user, pwd)
        with st.spinner("Retrieving…"):
            vec_ctx, graph_ctx = vector_search_and_context(driver, index_name, query, top_k, embed_model)

        # Answers
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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vector-only Answer")
            st.write(vector_answer)
            if show_context:
                with st.expander("Show vector-only context"):
                    st.code(vec_ctx)
        with col2:
            st.subheader("Vector+Graph Answer")
            st.write(graph_answer)
            if show_context:
                with st.expander("Show vector+graph context"):
                    st.code(graph_ctx)

        with st.expander("Diagnostics"):
            st.write({"index_name": index_name, "top_k": top_k,
                      "embed_model": embed_model, "gen_model": gen_model, "neo4j": uri})

    except Exception as e:
        st.error(str(e))
else:
    st.info("Set connection details, enter a question, then click **Search**.")
