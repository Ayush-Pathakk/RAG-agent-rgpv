import numpy as np
from typing import List, Dict
from embedder import Embedder
from vector_store_builder import VectorStore

class Retriever:
    """Retrieves relevant chunks from vector store."""
    
    def __init__(self, vector_store_path: str = "data/processed"):
        """
        Args:
            vector_store_path: Path to saved FAISS index
        """
        print("🔧 Initializing retriever...")
        
        # Load embedder
        self.embedder = Embedder()
        
        # Load vector store
        self.vector_store = VectorStore()
        self.vector_store.load(vector_store_path)
        
        print("✅ Retriever ready")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        score_threshold: float = 1.5
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question
            top_k: Number of results to return
            score_threshold: Max distance (lower = more similar)
                            1.5 is good default for filtering
        
        Returns:
            List of relevant documents with scores
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Convert query to embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=top_k)
        
        # Filter by threshold
        filtered = [r for r in results if r['score'] < score_threshold]
        
        print(f"✅ Found {len(filtered)} relevant results (from {len(results)} total)")
        
        return filtered


# BUILD VECTOR STORE FIRST (run once)
def build_index():
    """Build vector store from PDFs."""
    print("🏗️  BUILDING VECTOR STORE\n")
    
    from pdf_loader import PDFLoader
    from text_splitter import TextSplitter
    
    # Load PDFs
    loader = PDFLoader()
    docs = loader.load_pdfs()
    
    if not docs:
        print("❌ No documents to index")
        return
    
    # Chunk texts
    splitter = TextSplitter()
    chunks = splitter.split_documents(docs)
    
    # Generate embeddings
    embedder = Embedder()
    texts = [c['text'] for c in chunks]
    embeddings = embedder.embed_documents(texts)
    
    # Build vector store
    store = VectorStore()
    store.add_documents(embeddings, chunks)
    store.save()
    
    print("\n✅ Vector store built and saved!")


# TEST
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # Build index
        build_index()
    else:
        # Test retrieval
        print("🧪 TESTING RETRIEVER\n")
        
        # Check if index exists
        from pathlib import Path
        if not Path("data/processed/faiss.index").exists():
            print("⚠️  Vector store not found!")
            print("   Run: python src/retriever.py build")
            sys.exit(1)
        
        # Create retriever
        retriever = Retriever()
        
        # Test queries
        queries = [
            "What is a data structure?",
            "Explain arrays",
            "What is time complexity?"
        ]
        
        for query in queries:
            results = retriever.retrieve(query, top_k=3)
            
            print(f"\n📄 Results for: '{query}'")
            print("-" * 50)
            
            if results:
                for i, r in enumerate(results, 1):
                    print(f"{i}. Score: {r['score']:.3f}")
                    print(f"   Source: {r['source']} (Page {r['page']})")
                    print(f"   Text: {r['text'][:150]}...\n")
            else:
                print("   ❌ No relevant results found\n")