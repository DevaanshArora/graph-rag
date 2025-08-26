# kg_retriever.py
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

URI  = "bolt://localhost:7687"
AUTH = ("neo4j", "devaansh")
INDEX_NAME = "text_embeddings"  # your :Chunk(embedding) vector index

# Use the same embedding family as you indexed with (text-embedding-3-large, 3072 dims)
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0.0})

rag_template = RagTemplate(
    template='''
Answer the Question using ONLY the following Context. Do not speculate.

# Question:
{query_text}

# Context:
{context}

# Answer:
''',
    expected_inputs=['query_text', 'context']
)

# Vector-only retriever: constructor is (driver, index_name, embedder=?, return_properties=?, ...)
# Do NOT pass node_label or text_property – they’re inferred from the index.
def make_vector_retriever(driver):
    return VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder,
        # return_properties=["file","idx","text"]  # optional, if you want specific props returned
    )

# Vector+Graph retriever: signature is (driver, index_name, retrieval_query, embedder=?, ...)
RETRIEVAL_QUERY = """
WITH node AS chunk
MATCH p=(chunk)<-[:FROM_CHUNK]-(e)-[r*1..2]-(nb)
WHERE ALL(rel IN r WHERE type(rel) <> 'FROM_CHUNK')
WITH chunk, apoc.coll.toSet(REDUCE(acc=[], rels IN r | acc + rels)) AS rels
WITH collect(DISTINCT chunk) AS chunks, apoc.coll.flatten(collect(rels)) AS rels_flat
WITH chunks, apoc.coll.toSet(rels_flat) AS rels
RETURN
  apoc.text.join([c IN chunks | c.text], '\n') + '\n' +
  apoc.text.join([rel IN rels |
    startNode(rel).name + ' - ' + type(rel) +
    coalesce(' ' + rel.details, '') + ' -> ' + endNode(rel).name
  ], '\n') AS info
"""

def make_graph_retriever(driver):
    return VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        retrieval_query=RETRIEVAL_QUERY,
        embedder=embedder,
    )

if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        vector_retriever = make_vector_retriever(driver)
        graph_retriever  = make_graph_retriever(driver)

        vector_rag = GraphRAG(retriever=vector_retriever, llm=llm, prompt_template=rag_template)
        graph_rag  = GraphRAG(retriever=graph_retriever,  llm=llm, prompt_template=rag_template)
        q1 = "What are the RTO and MTPD for the Branch Cash Replenishment process? Answer only with numbers and units."
       # q1 = "Summarize the Branch Cash Replenishment Process: key steps, RTO/MTPD, dependencies, and controls."
        q2 = "Who is notified in the Cash Management Operations Emergency call tree and what is the escalation?"

        print("\nVECTOR-ONLY:\n", vector_rag.search(query_text=q1, retriever_config={'top_k': 5}).answer)
        print("\nVECTOR+GRAPH:\n", graph_rag.search(query_text=q1,  retriever_config={'top_k': 5}).answer)
        print("\nVECTOR+GRAPH (call tree):\n", graph_rag.search(query_text=q2, retriever_config={'top_k': 5}).answer)
    finally:
        driver.close()
