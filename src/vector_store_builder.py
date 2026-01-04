import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict

class VectorStore:
    """FAISS vector store for similarity search."""
    
    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Embedding dimension (384 for MiniLM)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add documents to vector store.
        
        Args:
            embeddings: Document embeddings
            documents: Document metadata
        """
        print(f"💾 Adding {len(embeddings)} documents to vector store...")
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        print(f"✅ Vector store now has {self.index.ntotal} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            List of documents with scores
        """
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(distances[0][i])
                results.append(doc)
        
        return results
    
    def save(self, directory: str = "data/processed"):
        """Save vector store to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{directory}/faiss.index")
        
        # Save documents
        with open(f"{directory}/documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"💾 Vector store saved to {directory}/")
    
    def load(self, directory: str = "data/processed"):
        """Load vector store from disk."""
        self.index = faiss.read_index(f"{directory}/faiss.index")
        
        with open(f"{directory}/documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"📂 Loaded {self.index.ntotal} documents from {directory}/")


# TEST
if __name__ == "__main__":
    print("🧪 TESTING VECTOR STORE\n")
    
    from embedder import Embedder
    
    # Sample data
    texts = ["Arrays store data", "Linked lists use nodes"]
    docs = [
        {'text': texts[0], 'source': 'test.pdf', 'page': 1},
        {'text': texts[1], 'source': 'test.pdf', 'page': 2}
    ]
    
    # Generate embeddings
    embedder = Embedder()
    embeddings = embedder.embed_documents(texts)
    
    # Build vector store
    store = VectorStore()
    store.add_documents(embeddings, docs)
    
    # Test search
    query_emb = embedder.embed_query("What is an array?")
    results = store.search(query_emb, k=2)
    
    print("\n🔍 Search results:")
    for r in results:
        print(f"   Score: {r['score']:.3f} | {r['text']}")
    
    # Test save/load
    store.save()
    
    new_store = VectorStore()
    new_store.load()
    print(f"\n✅ Test passed!")