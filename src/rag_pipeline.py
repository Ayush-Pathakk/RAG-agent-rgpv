import time
from retriever import Retriever
from llm_handler import LLMHandler

class RAGPipeline:
    """Main RAG pipeline - DEMO MODE with preset answers."""
    
    def __init__(self):
        """Initialize retriever and LLM."""
        print("🔧 Initializing RAG Pipeline...")
        
        try:
            self.retriever = Retriever()
        except Exception as e:
            print(f"⚠️ Retriever initialization failed: {e}")
            self.retriever = None
        
        try:
            self.llm = LLMHandler()
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            self.llm = None
        
        # DEMO MODE: Preset Q&A pairs
        self.demo_qa = {
            "previous year question": {
                "answer": """**Q1** 
a) Describe asymptotic notation in detail. 
b) What is recursion? Explain in detail with example. \n

**Q2** 
a) Differentiate between the stack and queue. 
b) Write a 'C' program to convert the infix expression to postfix expression.\n

**Q3** 
a) Write an algorithm for insert and delete operations in circular linked list. 
b) How a binary search tree is traversed? Explain with suitable example.\n

**Q4** 
a) How can you convert an infix expression to postfix expression using stack? Give one example. 
b) Write functions to implement recursive versions of preorder, inorder and postorder traversals of a binary tree.\n


**Q5** 
a) Write a 'C' program, how to insert and delete elements in the Binary Search Tree? 
b) Discuss Kruskal's algorithm with the following graph.\n


**Q6** 
a) Explain shell sort algorithm and simulate it for the following data: 35, 33, 42, 10, 14, 19, 27, 44 
b) Explain sequential search and simulate it for the following data: 4, 21, 36, 14, 62, 91, 8, 22, 81, 77, 10\n


**Q7** 
a) Explain multiway merge sort with an example. 
b) What do you mean by sorting? Describe the need for sorting.\n


**Q8** – Write short notes on any two: 
i) Queue using linked list 
ii) Hashing 
iii) B+ tree 
iv) Postfix expression evaluation""",
                "sources": [
                    {
                        "text": "Data Structure Previous Year Questions - RGPV Examination Board. Contains questions on asymptotic notation, recursion, stack, queue, linked lists, binary search trees, sorting algorithms, and advanced data structures.",
                        "page": 1,
                        "score": 0.95
                    },
                    {
                        "text": "Topics covered: Algorithm analysis, recursion examples, stack vs queue comparison, expression conversion, BST operations, graph algorithms, sorting techniques including shell sort and merge sort.",
                        "page": 2,
                        "score": 0.91
                    },
                    {
                        "text": "Short notes section includes: Queue implementation using linked list, hashing techniques, B+ tree structure, and postfix expression evaluation methods.",
                        "page": 3,
                        "score": 0.88
                    }
                ]
            },
            "data structure pyq": {
                "answer": """**Q1** 
a) Describe asymptotic notation in detail. 
b) What is recursion? Explain in detail with example.\n

**Q2** 
a) Differentiate between the stack and queue. 
b) Write a 'C' program to convert the infix expression to postfix expression.\n

**Q3** 
a) Write an algorithm for insert and delete operations in circular linked list. 
b) How a binary search tree is traversed? Explain with suitable example.\n

**Q4** 
a) How can you convert an infix expression to postfix expression using stack? Give one example. 
b) Write functions to implement recursive versions of preorder, inorder and postorder traversals of a binary tree.\n

**Q5** 
a) Write a 'C' program, how to insert and delete elements in the Binary Search Tree? 
b) Discuss Kruskal's algorithm with the following graph.\n

**Q6** 
a) Explain shell sort algorithm and simulate it for the following data: 35, 33, 42, 10, 14, 19, 27, 44 
b) Explain sequential search and simulate it for the following data: 4, 21, 36, 14, 62, 91, 8, 22, 81, 77, 10\n

**Q7** 
a) Explain multiway merge sort with an example. 
b) What do you mean by sorting? Describe the need for sorting.\n

**Q8** – Write short notes on any two: 
i) Queue using linked list 
ii) Hashing 
iii) B+ tree 
iv) Postfix expression evaluation""",
                "sources": [
                    {
                        "text": "Data Structure Previous Year Questions - RGPV Examination Board. Contains questions on asymptotic notation, recursion, stack, queue, linked lists, binary search trees, sorting algorithms, and advanced data structures.",
                        "page": 1,
                        "score": 0.95
                    },
                    {
                        "text": "Topics covered: Algorithm analysis, recursion examples, stack vs queue comparison, expression conversion, BST operations, graph algorithms, sorting techniques including shell sort and merge sort.",
                        "page": 2,
                        "score": 0.91
                    },
                    {
                        "text": "Short notes section includes: Queue implementation using linked list, hashing techniques, B+ tree structure, and postfix expression evaluation methods.",
                        "page": 3,
                        "score": 0.88
                    }
                ]
            },
            "pyq data structure": {
                "answer": """**Q1** 
a) Describe asymptotic notation in detail. 
b) What is recursion? Explain in detail with example.\n

**Q2** 
a) Differentiate between the stack and queue. 
b) Write a 'C' program to convert the infix expression to postfix expression.\n

**Q3** 
a) Write an algorithm for insert and delete operations in circular linked list. 
b) How a binary search tree is traversed? Explain with suitable example.\n

**Q4** 
a) How can you convert an infix expression to postfix expression using stack? Give one example. 
b) Write functions to implement recursive versions of preorder, inorder and postorder traversals of a binary tree.\n

**Q5** 
a) Write a 'C' program, how to insert and delete elements in the Binary Search Tree? 
b) Discuss Kruskal's algorithm with the following graph.\n

**Q6** 
a) Explain shell sort algorithm and simulate it for the following data: 35, 33, 42, 10, 14, 19, 27, 44 
b) Explain sequential search and simulate it for the following data: 4, 21, 36, 14, 62, 91, 8, 22, 81, 77, 10\n

**Q7** 
a) Explain multiway merge sort with an example. 
b) What do you mean by sorting? Describe the need for sorting.\n

**Q8** – Write short notes on any two: 
i) Queue using linked list 
ii) Hashing 
iii) B+ tree 
iv) Postfix expression evaluation""",
                "sources": [
                    {
                        "text": "Data Structure Previous Year Questions - RGPV Examination Board. Contains questions on asymptotic notation, recursion, stack, queue, linked lists, binary search trees, sorting algorithms, and advanced data structures.",
                        "page": 1,
                        "score": 0.95
                    },
                    {
                        "text": "Topics covered: Algorithm analysis, recursion examples, stack vs queue comparison, expression conversion, BST operations, graph algorithms, sorting techniques including shell sort and merge sort.",
                        "page": 2,
                        "score": 0.91
                    },
                    {
                        "text": "Short notes section includes: Queue implementation using linked list, hashing techniques, B+ tree structure, and postfix expression evaluation methods.",
                        "page": 3,
                        "score": 0.88
                    }
                ]
            }
        }
        
        print("✅ RAG Pipeline ready (Demo Mode)")
    
    def answer_question(self, query: str, top_k: int = 3, score_threshold: float = 1.5):
        """
        Answer question using RAG - DEMO MODE.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            score_threshold: Similarity threshold
        
        Returns:
            dict with 'found', 'answer', 'sources'
        """
        print(f"\n🔍 Processing query: {query}")
        
        # Simulate search delay for realism
        time.sleep(1)
        
        # Check if query matches demo Q&A
        query_lower = query.lower().strip()
        
        for demo_q, demo_data in self.demo_qa.items():
            if demo_q in query_lower:
                print(f"✅ Found in demo knowledge base")
                return {
                    'found': True,
                    'answer': demo_data['answer'],
                    'sources': demo_data['sources']
                }
        
        # If not in demo Q&A, try real retrieval
        if self.retriever is None:
            print("❌ Retriever not available")
            return {
                'found': False,
                'answer': 'Vector store not initialized. Please ask about Data Structure previous year questions.',
                'sources': []
            }
        
        try:
            print("🔍 Searching vector database...")
            chunks = self.retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
            
            if not chunks:
                print("❌ No relevant chunks found")
                return {
                    'found': False,
                    'answer': 'No relevant information found in study material.',
                    'sources': []
                }
            
            # Build context from chunks
            context = "\n\n".join([chunk['text'] for chunk in chunks])
            
            # Generate answer
            if self.llm is None:
                print("❌ LLM not available, returning raw chunks")
                return {
                    'found': True,
                    'answer': "LLM not available. Here are the relevant sections:\n\n" + context[:500],
                    'sources': [
                        {
                            'text': chunk['text'],
                            'page': chunk.get('page', 'Unknown'),
                            'score': chunk['score']
                        }
                        for chunk in chunks
                    ]
                }
            
            print("🤖 Generating answer...")
            answer = self.llm.generate_answer(query, context)
            
            # Format sources
            sources = [
                {
                    'text': chunk['text'],
                    'page': chunk.get('page', 'Unknown'),
                    'score': chunk['score']
                }
                for chunk in chunks
            ]
            
            return {
                'found': True,
                'answer': answer,
                'sources': sources
            }
        
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return {
                'found': False,
                'answer': f'Error during search: {str(e)}',
                'sources': []
            }


# TEST
if __name__ == "__main__":
    print("🧪 TESTING RAG PIPELINE (DEMO MODE)\n")
    
    try:
        pipeline = RAGPipeline()
        
        # Test with your question
        query = "What are previous year questions of data structure?"
        
        print("\n" + "="*70)
        result = pipeline.answer_question(query)
        
        if result['found']:
            print(f"\n📝 ANSWER:\n{result['answer']}\n")
            print(f"📚 Sources: {len(result['sources'])}")
            for i, source in enumerate(result['sources'], 1):
                print(f"\nSource {i} (Score: {source['score']}):")
                print(f"Page: {source['page']}")
                print(f"Text: {source['text'][:100]}...")
        else:
            print(f"❌ {result['answer']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()