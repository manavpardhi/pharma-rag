from src.retrieval.vector_store import PharmaVectorStore

def test_query():
    store = PharmaVectorStore()
    
    # A specific "Pharma Logic" question
    query = "What is the indication for Mekinist in melanoma?"
    
    print(f"Query: {query}\n")
    results = store.search(query, k=2)
    print(results['ids'])
    # get first result
    ids = results['ids'][0]
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for i, doc in enumerate(docs):
        meta = metadatas[i]
        print(f"--- Result {i+1} (ID: {ids[i]}) ---")
        print(f"Drug: {meta.get('drug_name')} | Section: {meta.get('section')}")
        print(f"Snippet: {doc[:300]}...\n")

if __name__ == "__main__":
    test_query()