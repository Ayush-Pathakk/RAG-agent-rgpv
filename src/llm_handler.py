import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

class LLMHandler:
    """Handle Groq API calls for answer generation."""
    
    def __init__(self):
        """Initialize Groq API."""
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=api_key)
        print("✅ Groq API initialized")
    
    def generate_answer(self, query: str, context: str, marks: int = 5) -> str:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User question
            context: Retrieved text from PDFs
            marks: Answer length (default 5 marks)
        
        Returns:
            Generated answer
        """
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Retry logic
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Call Groq
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an RGPV exam assistant. Answer questions using ONLY the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                return response.choices[0].message.content
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if rate limit error
                if "429" in error_msg or "rate" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "❌ Rate limit exceeded. Please try again in a moment."
                else:
                    return f"❌ Error generating answer: {str(e)}"
        
        return "❌ Failed after multiple retries."
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM (5 marks format)."""
        
        prompt = f"""You are an RGPV exam assistant helping students prepare for exams.

CONTEXT FROM STUDY MATERIAL:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Write a clear, exam-style answer in 5 marks format (100-150 words)
- Use ONLY the information provided in the context above
- Structure your answer in a single well-organized paragraph
- Include examples if they are mentioned in the context
- Write naturally as if explaining to a student
- Do NOT say "information not available" - if the context has relevant info, use it to answer
- If the context truly has nothing relevant, then say "This topic is not covered in the provided material"

ANSWER:"""
        
        return prompt


# TEST
if __name__ == "__main__":
    print("🧪 TESTING GROQ API\n")
    
    try:
        llm = LLMHandler()
        
        # Test with sample context
        context = """
        Recursion is a programming technique where a function calls itself.
        It requires a base case to stop recursion and a recursive case.
        Example: Factorial function f(n) = n * f(n-1) with base case f(0) = 1.
        """
        
        query = "What is recursion? Explain with example."
        
        print(f"Query: {query}\n")
        print('='*60)
        print("📝 5 MARKS ANSWER:")
        print('='*60)
        
        answer = llm.generate_answer(query, context)
        print(answer)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. .env file exists in project root")
        print("2. GROQ_API_KEY is set in .env")
        print("3. API key is valid")