from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model for embeddings
        """
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Model loaded")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        print(f"🧠 Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Embeddings generated: shape {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.model.encode([query])[0]


# TEST
if __name__ == "__main__":
    print("🧪 TESTING EMBEDDER\n")
    
    # Test texts
    texts = [
        "Arrays are contiguous memory locations",
        "Linked lists use pointers to connect nodes",
        "Stacks follow LIFO principle"
    ]
    
    # Create embedder
    embedder = Embedder()
    
    # Generate embeddings
    embeddings = embedder.embed_documents(texts)
    
    print(f"\n✅ Test passed!")
    print(f"   Input: {len(texts)} texts")
    print(f"   Output: {embeddings.shape}")