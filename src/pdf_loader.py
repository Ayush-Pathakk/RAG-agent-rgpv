import os
from pathlib import Path
from PyPDF2 import PdfReader
from typing import List, Dict

class PDFLoader:
    """
    Simple PDF loader for RGPV RAG system.
    Loads PDFs from data/raw/ folder and extracts text.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the PDF loader.
        
        Args:
            data_dir: Path to folder containing PDFs
        """
        self.data_dir = Path(data_dir)
        
        # Create directory if it doesn't exist
        if not self.data_dir.exists():
            print(f"⚠️  Creating directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pdfs(self) -> List[Dict[str, any]]:
        """
        Load all PDFs from the data directory.
        
        Returns:
            List of documents with structure:
            [
                {
                    'text': 'extracted text...',
                    'source': 'filename.pdf',
                    'page': 1
                },
                ...
            ]
        """
        documents = []
        
        # Find all PDF files
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {self.data_dir}")
            print(f"   Please add PDFs to this folder and try again.")
            return documents
        
        print(f"📚 Found {len(pdf_files)} PDF file(s)")
        print("-" * 50)
        
        # Process each PDF
        for pdf_path in pdf_files:
            print(f"\n📄 Processing: {pdf_path.name}")
            
            try:
                # Extract text from this PDF
                pdf_docs = self._extract_text_from_pdf(pdf_path)
                documents.extend(pdf_docs)
                
                print(f"   ✅ Extracted {len(pdf_docs)} pages")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                print(f"   ⚠️  Skipping this file...")
                continue
        
        print("\n" + "=" * 50)
        print(f"✅ Total pages extracted: {len(documents)}")
        print("=" * 50)
        
        return documents
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        # Read the PDF
        reader = PdfReader(str(pdf_path))
        
        # Extract text from each page
        for page_num, page in enumerate(reader.pages, start=1):
            # Extract text
            text = page.extract_text()
            
            # Only save if text exists and is not empty
            if text and text.strip():
                documents.append({
                    'text': text.strip(),
                    'source': pdf_path.name,
                    'page': page_num
                })
        
        return documents


# ==========================================
# TEST CODE (Run this file to test)
# ==========================================

if __name__ == "__main__":
    """
    Test the PDF loader with your data.
    Run: python src/pdf_loader.py
    """
    
    print("🧪 TESTING PDF LOADER")
    print("=" * 50)
    
    # Create loader
    loader = PDFLoader(data_dir="data/raw")
    
    # Load all PDFs
    docs = loader.load_pdfs()
    
    # Show results
    if docs:
        print("\n📊 SAMPLE OUTPUT:")
        print("-" * 50)
        
        # Show first document
        sample = docs[0]
        print(f"Source: {sample['source']}")
        print(f"Page: {sample['page']}")
        print(f"Text length: {len(sample['text'])} characters")
        print(f"\nFirst 300 characters:")
        print(sample['text'][:300])
        print("...")
        
        # Show stats
        print("\n📈 STATISTICS:")
        print(f"   Total documents: {len(docs)}")
        
        # Count unique sources
        sources = set(d['source'] for d in docs)
        print(f"   Unique PDFs: {len(sources)}")
        
        for source in sources:
            count = sum(1 for d in docs if d['source'] == source)
            print(f"      - {source}: {count} pages")
    
    else:
        print("\n⚠️  No documents loaded!")
        print("   Make sure you have PDFs in data/raw/ folder")