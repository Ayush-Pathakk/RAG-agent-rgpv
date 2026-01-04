from typing import List, Dict
import re

class TextSplitter:
    """
    Smart text splitter for RAG system.
    Splits documents into chunks while maintaining context.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Target size of each chunk (in words)
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of documents from PDFLoader
                      Each doc has: {'text': str, 'source': str, 'page': int}
        
        Returns:
            List of chunked documents with metadata
        """
        all_chunks = []
        
        print(f"📝 Splitting {len(documents)} documents into chunks...")
        print(f"   Chunk size: ~{self.chunk_size} words")
        print(f"   Overlap: {self.chunk_overlap} words")
        print("-" * 50)
        
        for doc in documents:
            # Split this document into chunks
            chunks = self._split_text(doc['text'])
            
            # Add metadata to each chunk
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    'text': chunk_text,
                    'source': doc['source'],
                    'page': doc['page'],
                    'chunk_id': i + 1
                })
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} pages")
        print("=" * 50)
        
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split into sentences (simple approach)
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                # Join sentences into chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few sentences for context
                overlap_words = 0
                overlap_sentences = []
                
                # Go backwards to collect overlap
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_words + sent_words <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_words += sent_words
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_word_count = overlap_words
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Simple approach using regex.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence endings (., !, ?)
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


# ==========================================
# TEST CODE
# ==========================================

if __name__ == "__main__":
    """
    Test the text splitter with PDF loader output.
    Run: python src/text_splitter.py
    """
    
    print("🧪 TESTING TEXT SPLITTER")
    print("=" * 50)
    
    # First, load PDFs
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import pdf_loader
    sys.path.append(str(Path(__file__).parent))
    
    from pdf_loader import PDFLoader
    
    # Load documents
    print("\n📚 Step 1: Loading PDFs...")
    loader = PDFLoader(data_dir="data/raw")
    documents = loader.load_pdfs()
    
    if not documents:
        print("❌ No documents to split. Add PDFs to data/raw/")
        sys.exit(1)
    
    # Split into chunks
    print("\n✂️  Step 2: Splitting into chunks...")
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Show results
    print("\n📊 RESULTS:")
    print("-" * 50)
    print(f"Original pages: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunks per page: {len(chunks)/len(documents):.1f}")
    
    # Show sample chunks
    print("\n📋 SAMPLE CHUNKS:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
        print(f"\nChunk {i}:")
        print(f"  Source: {chunk['source']}")
        print(f"  Page: {chunk['page']}")
        print(f"  Chunk ID: {chunk['chunk_id']}")
        print(f"  Word count: {len(chunk['text'].split())}")
        print(f"  Preview: {chunk['text'][:200]}...")
    
    # Show chunk size distribution
    print("\n📈 CHUNK SIZE DISTRIBUTION:")
    word_counts = [len(chunk['text'].split()) for chunk in chunks]
    print(f"  Min words: {min(word_counts)}")
    print(f"  Max words: {max(word_counts)}")
    print(f"  Average words: {sum(word_counts)/len(word_counts):.0f}")