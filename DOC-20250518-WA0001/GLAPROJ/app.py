import io
import os
import asyncio
import numpy as np
import uvicorn
import httpx
from urllib.parse import urlparse
from typing import Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
from pathlib import Path
import logging

# Add debug logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add startup event handler
@app.on_event("startup")
async def startup_event():
    """Clear all states and pickle files when the application starts."""
    logger.info("Starting up application - clearing all states")
    try:
        # Ensure storage directory exists
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        # Clear all pickle files
        for file in STORAGE_DIR.glob("*.pkl"):
            file.unlink()
            logger.debug(f"Deleted pickle file: {file}")
            
        # Reset global states
        global document_chunks, document_metadata
        document_chunks = []
        document_metadata = {
            "title": None,
            "type": None,
            "source": None
        }
        logger.info("Application state reset successfully")
    except Exception as e:
        logger.error(f"Error during startup cleanup: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Update Gemini API configuration to use environment variable
api_key = 'YOUR_API_KEY_HERE'
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")
genai.configure(api_key=api_key)

# Load a pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global in-memory storage for document chunks.
# Each entry is a dict with keys "text" and "embedding"
document_chunks = []

# Global in-memory storage for document metadata
document_metadata = {
    "title": None,
    "type": None,
    "source": None
}

# Add session tracking
last_activity = None
session_timeout = 3600  # 1 hour in seconds

# Add storage configuration
STORAGE_DIR = Path(__file__).parent / "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)  # Create directory if it doesn't exist

def save_document_state():
    """Save the current document state to disk."""
    try:
        if not document_chunks:
            logger.warning("No document chunks to save")
            return False

        # Create a simplified state object for storage
        simplified_state = {
            "chunks": [
                {
                    "text": chunk["text"],
                    "embedding": list(chunk["embedding"])  # Ensure it's a regular list
                }
                for chunk in document_chunks
            ],
            "metadata": document_metadata,
            "timestamp": time.time()
        }
        
        # Ensure storage directory exists
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        state_file = STORAGE_DIR / "document_state.pkl"
        logger.debug(f"Saving state to {state_file}")
        
        # Use the highest protocol version that's stable
        with open(state_file, "wb") as f:
            pickle.dump(simplified_state, f, protocol=4)
        
        logger.debug(f"State saved successfully: {len(document_chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"Error saving document state: {e}")
        return False

def load_document_state() -> bool:
    """Load document state from disk."""
    global document_chunks, document_metadata
    try:
        state_file = STORAGE_DIR / "document_state.pkl"
        if not state_file.exists():
            return False
            
        # Check if state is too old (e.g., more than 24 hours)
        if time.time() - state_file.stat().st_mtime > 86400:
            return False
            
        with open(state_file, "rb") as f:
            state = pickle.load(f)
            
        # Convert loaded chunks back to numpy arrays if needed
        document_chunks = state["chunks"]  # Now stored as regular lists
        document_metadata = state["metadata"]
        return True
    except Exception as e:
        logger.error(f"Error loading document state: {e}")
        return False

def verify_document_state() -> bool:
    """Verify if the document is loaded and valid."""
    global document_chunks, document_metadata
    
    logger.debug("Verifying document state...")
    logger.debug(f"Number of chunks: {len(document_chunks)}")
    logger.debug(f"Metadata: {document_metadata}")
    
    try:
        # Check if we have chunks in memory
        if not document_chunks:
            logger.debug("No chunks in memory, trying to load from disk")
            if not load_document_state():
                logger.warning("Failed to load document state from disk")
                return False
        
        # Verify chunks exist and have correct format
        if not document_chunks or len(document_chunks) == 0:
            logger.warning("No document chunks found")
            return False
            
        # Verify each chunk has required fields
        for i, chunk in enumerate(document_chunks):
            if not isinstance(chunk, dict) or 'text' not in chunk or 'embedding' not in chunk:
                logger.error(f"Invalid chunk format at index {i}")
                return False
                
        # Verify metadata exists
        if not document_metadata or not document_metadata.get('title'):
            logger.warning("Invalid or missing document metadata")
            return False
            
        logger.debug("Document state verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error during document state verification: {e}")
        return False

def clear_document_state():
    """Clear all document state including stored files."""
    global document_chunks, document_metadata
    
    # Clear memory
    document_chunks = []
    document_metadata = {
        "title": None,
        "type": None,
        "source": None
    }
    
    # Clear stored files
    try:
        for file in STORAGE_DIR.glob("*.pkl"):
            file.unlink()
    except Exception as e:
        print(f"Error clearing stored files: {e}")

### Helper functions for document processing and indexing ###

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_html(file_bytes: bytes) -> dict:
    """Extract text and metadata from an HTML file."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']
        html_content = None
        
        for encoding in encodings:
            try:
                html_content = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if html_content is None:
            # If all encodings fail, use beautifulsoup's built-in parser
            soup = BeautifulSoup(file_bytes, "html.parser", from_encoding='utf-8')
        else:
            soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract title
        title = soup.title.string if soup.title else "Untitled HTML Document"
        title = title.strip() if title else "Untitled HTML Document"
        
        # Collect text from all important sections
        text_elements = []
        
        # Get all text elements while excluding scripts, styles, and code
        for element in soup.find_all(['p', 'div', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if element.parent.name not in ['script', 'style', 'code']:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 20:  # Only keep substantial content
                    text_elements.append(text)
        
        # Extract meta description
        meta_desc = ""
        meta_tags = soup.find_all('meta', attrs={'name': ['description', 'keywords'], 
                                               'property': ['og:description', 'og:title']})
        for tag in meta_tags:
            content = tag.get('content', '').strip()
            if content:
                meta_desc += content + " "
        
        # Combine all content with clear section breaks
        content = "\n\n".join(text_elements)
        
        if not content.strip():
            # Fallback: get all text if no structured content found
            content = soup.get_text(separator='\n\n', strip=True)
        
        return {
            "title": title,
            "content": content,
            "description": meta_desc.strip()
        }
        
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        # Return minimal valid structure even if parsing fails
        return {
            "title": "Untitled Document",
            "content": "",
            "description": ""
        }

def chunk_text(text: str, max_words: int = 150, overlap: int = 30) -> list:
    """Break text into overlapping chunks to preserve context."""
    # First split by double newlines to preserve natural breaks
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
            
        # If a single paragraph is too long, split it
        if len(words) > max_words:
            for i in range(0, len(words), max_words - overlap):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
        else:
            # Try to combine shorter paragraphs
            if current_length + len(words) <= max_words:
                current_chunk.extend(words)
                current_length += len(words)
            else:
                # Store current chunk and start a new one
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = words
                current_length = len(words)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

### Dummy Embedding & Similarity Functions ###

def dummy_embedding(text: str) -> np.ndarray:
    """
    Create a dummy embedding by counting normalized letter frequencies.
    """
    vector = np.zeros(26)
    for char in text.lower():
        if 'a' <= char <= 'z':  # Ensure the character is a lowercase letter
            index = ord(char) - ord('a')
            vector[index] += 1
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def get_embedding(text: str):
    """Generate an embedding for the given text using a transformer model."""
    try:
        # Get the embedding and immediately convert to a simple Python list
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # Convert numpy array to regular Python list
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * 384  # default dimension for 'all-MiniLM-L6-v2'

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 > 0 and norm2 > 0:
        return dot / (norm1 * norm2)
    return 0.0

### Search Functions ###

def keyword_search(query: str) -> list:
    """A simple keyword search that returns chunks containing the query text."""
    results = []
    query_lower = query.lower()
    for chunk in document_chunks:
        if query_lower in chunk["text"].lower():
            results.append(chunk["text"])
    return results

def semantic_search(query: str, top_k: int = 7) -> list:
    """Perform semantic search with improved relevance scoring."""
    query_emb = np.array(get_embedding(query))
    scored_chunks = []
    
    for chunk in document_chunks:
        # Convert chunk embedding to numpy array for comparison
        chunk_emb = np.array(chunk["embedding"])
        sim = float(np.dot(query_emb, chunk_emb) / 
                   (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)))
        
        # Calculate term overlap score
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk["text"].lower().split())
        term_overlap = len(query_terms.intersection(chunk_terms)) / len(query_terms) if query_terms else 0
        
        combined_score = (sim * 0.7) + (term_overlap * 0.3)
        scored_chunks.append({
            "text": chunk["text"],
            "score": combined_score
        })

    # Sort by combined score and filter
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top_k results above threshold
    if scored_chunks:
        max_score = scored_chunks[0]["score"]
        threshold = max(0.05, max_score * 0.3)
        return [chunk["text"] for chunk in scored_chunks[:top_k] 
                if chunk["score"] > threshold]
    return []

def combine_results(keyword_results: list, semantic_results: list) -> str:
    """Combine results from both searches and remove duplicates."""
    combined = list(set(keyword_results + semantic_results))
    # Join chunks with a clear delimiter
    return "\n---\n".join(combined)

def analyze_query_type(query: str) -> dict:
    """Analyze the query to determine its type and requirements."""
    query_lower = query.strip().lower()
    
    # Query type analysis
    analysis = {
        "is_greeting": is_greeting(query_lower),
        "is_comparison": any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better', 'pros and cons']),
        "is_listing": any(word in query_lower for word in ['list', 'what are', 'tell me all', 'enumerate', 'show me', 'give me']),
        "is_definition": any(word in query_lower for word in ['what is', 'define', 'explain', 'describe', 'meaning of']),
        "is_summary": any(word in query_lower for word in ['summarize', 'summary', 'brief', 'overview', 'tldr']),
        "is_example": any(word in query_lower for word in ['example', 'instance', 'show me an example', 'sample']),
        "is_how_to": any(word in query_lower for word in ['how to', 'how do i', 'steps to', 'guide', 'procedure']),
        "is_why": query_lower.startswith('why') or 'reason for' in query_lower or 'explain why' in query_lower,
        "requires_context": not (is_greeting(query_lower) or query_lower.startswith('hi') or query_lower.startswith('hello'))
    }
    
    # Extract key terms for focused search
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    key_terms = [word for word in query_lower.split() if word not in stop_words]
    analysis["key_terms"] = key_terms
    
    return analysis

def get_response_format(analysis: dict) -> str:
    """Get appropriate response format based on query analysis."""
    if analysis["is_comparison"]:
        return """
        Format your response as a clear comparison:
        1. First, identify the items being compared
        2. Use a structured format with clear headings
        3. Present key differences and similarities
        4. Include a brief summary of the comparison
        """
    elif analysis["is_listing"]:
        return """
        Format your response as a clear list:
        • Use bullet points for better readability
        • Group related items together
        • Provide brief explanations where needed
        • Keep the formatting consistent
        """
    elif analysis["is_how_to"]:
        return """
        Format your response as a step-by-step guide:
        1. Start with any prerequisites
        2. Present steps in a logical order
        3. Include any important cautions or notes
        4. End with expected outcomes
        """
    elif analysis["is_summary"]:
        return """
        Format your response as a concise summary:
        • Start with the main point
        • Include only key information
        • Use clear, direct language
        • End with any important conclusions
        """
    else:
        return """
        Format your response clearly:
        • Provide a direct answer first
        • Include supporting details
        • Use examples where helpful
        • Maintain clarity and precision
        """

def build_comparison_prompt(query: str) -> str:
    """Build a specialized prompt for comparison questions."""
    return f"""You are analyzing a document. Answer this comparison question: {query}

Instructions for clear comparison:

1. First explain what specific items you are comparing
2. For each major difference found in the document:
   • Clearly state what differs
   • Support with specific quotes or evidence from the text
   • Explain why this difference matters

3. Present findings as clear bullet points:
   • Start each point with the key difference
   • Follow with evidence from the document
   • Explain practical implications

4. Focus on:
   • Direct comparisons only
   • Facts stated in the document
   • Clear distinctions
   • Practical differences

Do NOT:
- Do not create tables
- Do not add decorative formatting
- Do not make comparisons not supported by the document
- Do not include general knowledge outside the document

Format the response as:

Brief introduction: What is being compared

Key Differences:
• [First major difference]
  - Evidence: [Quote or reference from document]
  - Meaning: [Why this matters]

• [Second major difference]
  - Evidence: [Quote or reference from document]
  - Meaning: [Why this matters]

[Continue with additional differences]

Practical Implications:
• When to use/prefer one over the other
• Specific scenarios from the document

Only include differences that are explicitly stated or directly implied by the document content.
If certain common aspects aren't covered in the document, state that clearly.

Response:"""

def build_prompt(query: str) -> str:
    """Construct an improved prompt for more precise answers."""
    # Analyze the query
    analysis = analyze_query_type(query)
    
    # Handle greetings without accessing the document
    if analysis["is_greeting"]:
        return query

    if not document_chunks:
        return "No document has been uploaded yet. Please upload a document or provide a URL."

    # Get relevant document chunks with expanded context for comparisons
    kw_results = keyword_search(query)
    if analysis["is_comparison"]:
        sem_results = semantic_search(query, top_k=12)  # Get more context for comparisons
    else:
        sem_results = semantic_search(query)
    retrieved_context = combine_results(kw_results, sem_results)

    if not retrieved_context:
        return f"Question: {query}\nAnswer: I couldn't find specific information about '{query}' in the document. Could you rephrase your question or ask about a different aspect?"

    # For comparison queries, use specialized prompt
    if analysis["is_comparison"]:
        return f"""Based on this context:
{retrieved_context}

{build_comparison_prompt(query)}"""
    
    # For other queries, use standard prompt
    # Get document metadata
    doc_type = document_metadata.get('type', 'document')
    doc_title = document_metadata.get('title', 'Untitled')
    doc_desc = document_metadata.get('description', '')

    # Get appropriate response format
    response_format = get_response_format(analysis)

    # Build the enhanced prompt
    prompt = f"""You are a precise and thorough analysis assistant working with a {doc_type} titled "{doc_title}".
    {f'Document description: {doc_desc}' if doc_desc else ''}

    User Question: {query}
    
    Query Analysis:
    - Type: {', '.join(k for k, v in analysis.items() if v and k.startswith('is_') and k != 'is_greeting')}
    - Key Terms: {', '.join(analysis['key_terms'])}
    
    Context Excerpts:
    {retrieved_context}
    
    Instructions:
    1. Focus on answering the specific question asked
    2. Use only information from the provided excerpts
    3. If information is incomplete or unclear, state this explicitly
    4. Cite specific details and examples from the text
    5. Maintain technical accuracy and precision
    6. If multiple interpretations are possible, explain the alternatives
    
    {response_format}
    
    Answer:"""  # Fixed: Added missing quotation mark
    
    return prompt

async def real_gemini_response(prompt: str):
    """Calls Google Gemini API to generate a structured response."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        if hasattr(response, "text"):  # Ensure response has text
            return response.text
        return "Error: No response text received from Gemini."
    
    except Exception as e:
        return f"Error: {str(e)}"

def format_response(text: str) -> str:
    """Format the response for better readability without borders."""
    if not text.strip():
        return text
    
    # Convert any table format to bullet points
    if '|' in text and '-|-' in text:
        lines = text.split('\n')
        formatted_lines = []
        header_processed = False
        
        for line in lines:
            line = line.strip()
            if not line or '-|-' in line:
                continue
                
            # Convert table row to bullet point
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) >= 2:
                    formatted_lines.append(f"• {cells[0]}: {cells[1]}")
        
        text = '\n'.join(formatted_lines)
    
    # Handle bullet points and lists
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_lines.append("")  # Add space between list items
            continue
            
        # Handle bullet points
        if line.startswith('•') or line.startswith('-'):
            in_list = True
            formatted_lines.append(f"• {line[1:].strip()}")  # Standardize bullet points
        # Handle numbered lists
        elif line[0].isdigit() and line[1:3] in ['. ', ') ']:
            in_list = True
            formatted_lines.append(f"• {line[line.find(' ')+1:].trip()}")  # Convert numbers to bullets
        else:
            in_list = False
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting."""
    greetings = {
        'hi', 'hello', 'hey', 'howdy', 'hola', 'greetings', 
        'good morning', 'good afternoon', 'good evening',
        'hi there', 'hello there'
    }
    return text.lower().strip() in greetings

def get_greeting_response() -> str:
    """Return a friendly greeting response."""
    return """Hello! I'm ready to help you understand the document. You can ask me specific questions about its content, and I'll do my best to provide accurate answers. What would you like to know?"""

async def stream_gemini_response(prompt: str):
    """Streams response from Gemini API with improved formatting."""
    try:
        # Verify document state first
        if not verify_document_state() and not is_greeting(prompt):
            yield "I apologize, but I've lost connection to the document. Please try uploading it again."
            return
            
        # Handle greetings
        if is_greeting(prompt):
            yield get_greeting_response()
            return
            
        # Handle metadata responses
        if prompt.startswith("METADATA_RESPONSE:"):
            yield prompt.replace("METADATA_RESPONSE:", "").strip() + "\n"
            return
            
        model = genai.GenerativeModel("gemini-2.0-flash")
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "block_none",
            "HARM_CATEGORY_HATE_SPEECH": "block_none",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
        }
        
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Get the response without streaming first
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False
        )
        
        if not hasattr(response, "text"):
            yield "I apologize, but I couldn't generate a response at the moment. Please try asking your question again."
            return
            
        # Format the complete response
        formatted_response = format_response(response.text)
        
        # Stream the formatted response line by line
        for line in formatted_response.split('\n'):
            if line.strip():  # Only yield non-empty lines
                yield line + '\n'
                await asyncio.sleep(0.01)
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            yield "I apologize, but I'm experiencing high traffic at the moment. Please try again in a few seconds."
        else:
            yield "I encountered an error processing your request. Please try uploading the document again or rephrase your question."

### API Endpoints ###

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF or HTML), extract and chunk text, and index it."""
    try:
        global document_chunks, document_metadata
        
        logger.debug(f"Starting upload of file: {file.filename}")
        
        # Clear existing document state first
        clear_document_state()
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
            
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        # Process the document
        try:
            if file.filename.lower().endswith('.pdf'):
                text = parse_pdf(content)
                doc_type = "PDF"
                title = file.filename
            else:  # Treat as HTML
                parsed_html = parse_html(content)
                text = parsed_html["content"]
                doc_type = "HTML"
                title = parsed_html["title"] or file.filename
                
            if not text.strip():
                raise ValueError("No content extracted from document")
            
            document_metadata = {
                "title": title,
                "type": doc_type,
                "source": f"uploaded_file: {file.filename}"
            }
            
            logger.debug(f"Document parsed successfully: {document_metadata}")
            
            # Create chunks and embeddings
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError("No text chunks created from document")
            
            document_chunks = []
            for chunk in chunks:
                embedding = get_embedding(chunk)
                document_chunks.append({
                    "text": chunk,
                    "embedding": embedding
                })
            
            logger.debug(f"Created {len(document_chunks)} chunks with embeddings")
            
            if not save_document_state():
                raise ValueError("Failed to save document state")
            
            return {
                "message": f"Document processed successfully",
                "num_chunks": len(document_chunks),
                "document_info": document_metadata
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            clear_document_state()
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        clear_document_state()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload_url")
async def upload_url(url: str):
    """Upload and process a webpage from URL."""
    # Clear existing document state first
    clear_document_state()
    
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    try:
        content = await fetch_url_content(url)
        global document_metadata
        global document_chunks
        
        document_chunks = []
        
        # Parse the HTML content
        parsed_html = parse_html(content)
        text = parsed_html["content"]
        document_metadata = {
            "title": parsed_html["title"],
            "type": "URL",
            "source": url,
            "description": parsed_html["description"]
        }
        
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            document_chunks.append({"text": chunk, "embedding": embedding})
            
        # Save state after successful processing
        save_document_state()
        
        return {
            "message": "URL processed and indexed successfully",
            "num_chunks": len(document_chunks),
            "document_info": document_metadata
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

def is_valid_url(url: str) -> bool:
    """Check if the URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

async def fetch_url_content(url: str) -> bytes:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

@app.get("/status")
async def get_status():
    """Check if a document is loaded and get basic stats."""
    is_loaded = len(document_chunks) > 0 and document_metadata.get("title") is not None
    return {
        "document_loaded": is_loaded,
        "num_chunks": len(document_chunks) if is_loaded else 0,
        "document_info": document_metadata if is_loaded else None,
        "sample_chunk": document_chunks[0]["text"][:200] if is_loaded else None
    }

@app.get("/stream_query")
async def stream_query(q: str):
    """Process a query with document state verification."""
    logger.debug("Processing query request...")
    logger.debug(f"Query: {q}")
    
    if not verify_document_state():
        logger.warning("Document state verification failed")
        return JSONResponse({
            "error": "Document state invalid",
            "message": "Please upload your document again to continue the conversation."
        })

    logger.debug("Document state verified, building prompt...")
    prompt = build_prompt(q)
    return StreamingResponse(stream_gemini_response(prompt), media_type="text/plain")

@app.get("/query")
async def query(q: str):
    """Process a query with document state verification."""
    if not verify_document_state():
        return JSONResponse({
            "error": "Document state invalid",
            "message": "Please upload your document again to continue the conversation."
        })
        
    prompt = build_prompt(q)
    response_text = await real_gemini_response(prompt)
    return JSONResponse({"answer": response_text})

@app.post("/reset")
async def reset_state():
    """Reset the application state completely."""
    try:
        # Clear document state
        clear_document_state()
        
        # Return success with more detailed message
        return {
            "message": "Application state reset successfully",
            "status": "success",
            "details": {
                "documents_cleared": True,
                "memory_cleared": True
            }
        }
    except Exception as e:
        logger.error(f"Error during reset: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset application state")

@app.post("/cleanup")
async def cleanup_old_files():
    """Remove old document states."""
    try:
        for file in STORAGE_DIR.glob("*.pkl"):
            if time.time() - file.stat().st_mtime > 86400:  # 24 hours
                file.unlink()
        return {"message": "Old files cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

# Mount the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add a root endpoint to serve the chat interface
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "chat.html"))

def analyze_query_intent(query: str) -> dict:
    """Enhanced query intent analysis with multiple dimensions."""
    query_lower = query.strip().lower()
    words = set(query_lower.split())
    
    # Define intent patterns
    patterns = {
        "factual": {"what", "who", "where", "when", "which"},
        "explanatory": {"why", "how", "explain", "describe", "elaborate"},
        "comparative": {"compare", "difference", "versus", "vs", "better", "advantages", "disadvantages"},
        "procedural": {"steps", "process", "procedure", "guide", "instructions", "method"},
        "analytical": {"analyze", "evaluate", "assess", "examine", "review"},
        "examples": {"example", "instance", "sample", "show me", "illustrate"},
        "summary": {"summarize", "summary", "brief", "overview", "key points"},
        "validation": {"is it true", "verify", "confirm", "fact check", "accurate"}
    }
    
    # Detect query complexity
    complexity = {
        "multi_part": "and" in words or ";" in query,
        "requires_context": not bool(words.intersection({"hi", "hello", "hey"})),
        "needs_examples": bool(words.intersection(patterns["examples"])),
        "needs_verification": bool(words.intersection(patterns["validation"]))
    }
    
    # Identify primary and secondary intents
    intents = []
    for intent, keywords in patterns.items():
        if words.intersection(keywords):
            intents.append(intent)
    
    # Extract key entities and concepts
    entities = extract_key_entities(query)
    
    return {
        "primary_intent": intents[0] if intents else "general",
        "secondary_intents": intents[1:],
        "complexity": complexity,
        "entities": entities,
        "requires_sources": complexity["needs_verification"]
    }

def extract_key_entities(text: str) -> list:
    """Extract important entities and concepts from the query."""
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    words = [word.strip('.,?!') for word in text.lower().split()]
    key_terms = [word for word in words if word not in stop_words]
    
    # Look for quoted phrases
    quoted_phrases = re.findall(r'"([^"]*)"', text)
    
    # Look for technical terms or proper nouns (simplified)
    technical_terms = [word for word in key_terms if word[0].isupper()]
    
    return {
        "key_terms": key_terms,
        "quoted_phrases": quoted_phrases,
        "technical_terms": technical_terms
    }

def enhanced_semantic_search(query: str, intent_analysis: dict) -> list:
    """Improved semantic search with intent-aware retrieval."""
    query_emb = np.array(get_embedding(query))
    scored_chunks = []
    
    # Adjust search based on intent
    top_k = 7  # Default
    if intent_analysis["primary_intent"] == "summary":
        top_k = 10  # Get more context for summaries
    elif intent_analysis["primary_intent"] == "comparative":
        top_k = 12  # Need more context for comparisons
    
    # Get initial semantic matches
    for chunk in document_chunks:
        chunk_emb = np.array(chunk["embedding"])
        semantic_score = float(np.dot(query_emb, chunk_emb) / 
                            (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)))
        
        # Calculate term overlap score
        term_overlap = calculate_term_overlap(query, chunk["text"], intent_analysis["entities"])
        
        # Calculate context relevance
        context_score = calculate_context_relevance(chunk["text"], intent_analysis)
        
        # Combine scores with weights adjusted by intent
        if intent_analysis["primary_intent"] == "factual":
            combined_score = semantic_score * 0.6 + term_overlap * 0.3 + context_score * 0.1
        elif intent_analysis["primary_intent"] == "explanatory":
            combined_score = semantic_score * 0.4 + term_overlap * 0.2 + context_score * 0.4
        else:
            combined_score = semantic_score * 0.5 + term_overlap * 0.25 + context_score * 0.25
        
        scored_chunks.append({
            "text": chunk["text"],
            "score": combined_score,
            "semantic_score": semantic_score,
            "term_overlap": term_overlap,
            "context_score": context_score
        })
    
    # Sort and filter results
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Adaptive threshold based on score distribution
    if scored_chunks:
        max_score = scored_chunks[0]["score"]
        mean_score = np.mean([c["score"] for c in scored_chunks])
        threshold = max(0.1, mean_score * 0.7)  # Adaptive threshold
        
        results = [chunk["text"] for chunk in scored_chunks[:top_k] 
                  if chunk["score"] > threshold]
        
        # If results are too few, lower threshold
        if len(results) < 3 and scored_chunks:
            results = [chunk["text"] for chunk in scored_chunks[:3]]
            
        return results
    return []

def calculate_term_overlap(query: str, chunk_text: str, entities: dict) -> float:
    """Calculate weighted term overlap score."""
    query_terms = set(query.lower().split())
    chunk_terms = set(chunk_text.lower().split())
    
    # Give extra weight to technical terms and quoted phrases
    weighted_matches = 0
    for term in query_terms.intersection(chunk_terms):
        weight = 1.0
        if term in entities["technical_terms"]:
            weight = 2.0
        if any(term in phrase.lower() for phrase in entities["quoted_phrases"]):
            weight = 2.5
        weighted_matches += weight
    
    return weighted_matches / len(query_terms) if query_terms else 0

def calculate_context_relevance(chunk_text: str, intent_analysis: dict) -> float:
    """Calculate contextual relevance based on query intent."""
    relevance_score = 0.0
    
    # Look for intent-specific indicators
    if intent_analysis["primary_intent"] == "explanatory":
        indicators = ["because", "therefore", "thus", "as a result", "due to"]
        relevance_score += sum(indicator in chunk_text.lower() for indicator in indicators) * 0.2
    
    elif intent_analysis["primary_intent"] == "comparative":
        indicators = ["whereas", "while", "unlike", "similar to", "different from"]
        relevance_score += sum(indicator in chunk_text.lower() for indicator in indicators) * 0.2
    
    elif intent_analysis["primary_intent"] == "procedural":
        indicators = ["first", "second", "then", "next", "finally"]
        relevance_score += sum(indicator in chunk_text.lower() for indicator in indicators) * 0.2
    
    # Add base relevance
    relevance_score += 0.5
    
    return min(1.0, relevance_score)  # Normalize to [0, 1]

def build_enhanced_prompt(query: str) -> str:
    """Build an enhanced prompt with better context and guidance."""
    # Get query intent analysis
    intent_analysis = analyze_query_intent(query)
    
    # Get relevant document chunks using enhanced search
    relevant_chunks = enhanced_semantic_search(query, intent_analysis)
    
    if not relevant_chunks:
        return f"""Question: {query}\nAnswer: I couldn't find specific information about this in the document. Could you rephrase your question or ask about something else?"""
    
    # Build context-aware prompt
    prompt_parts = [
        f"""You are a precise and knowledgeable assistant analyzing a {document_metadata.get('type', 'document')} titled "{document_metadata.get('title', 'Untitled')}".

Query: {query}

Intent Analysis:
- Primary Intent: {intent_analysis['primary_intent']}
- Secondary Intents: {', '.join(intent_analysis['secondary_intents']) if intent_analysis['secondary_intents'] else 'None'}
- Complexity: {', '.join(k for k, v in intent_analysis['complexity'].items() if v)}

Relevant Context:"""]
    
    # Add numbered context sections
    for i, chunk in enumerate(relevant_chunks, 1):
        prompt_parts.append(f"[{i}] {chunk}")
    
    # Add intent-specific instructions
    prompt_parts.append("\nResponse Guidelines:")
    
    if intent_analysis["primary_intent"] == "factual":
        prompt_parts.append("1. Provide precise, fact-based information from the context")
        prompt_parts.append("2. Include specific details and numbers when available")
        prompt_parts.append("3. If multiple facts are relevant, organize them clearly")
        
    elif intent_analysis["primary_intent"] == "explanatory":
        prompt_parts.append("1. Explain concepts clearly and logically")
        prompt_parts.append("2. Use analogies or examples if helpful")
        prompt_parts.append("3. Break down complex ideas into simpler parts")
        
    elif intent_analysis["primary_intent"] == "comparative":
        prompt_parts.append("1. Present clear comparisons with distinct points")
        prompt_parts.append("2. Highlight key differences and similarities")
        prompt_parts.append("3. Use structured format for better clarity")
    
    # Add general guidelines
    prompt_parts.extend([
        "- Use only information from the provided context",
        "- Acknowledge any uncertainties or missing information",
        "- Keep the response focused and relevant",
        "- Format the response for easy reading",
        "\nResponse:"])
    
    return "\n".join(prompt_parts)

# Update the stream_query endpoint to use enhanced prompt
@app.get("/stream_query")
async def stream_query(q: str):
    """Process a query using enhanced understanding and retrieval."""
    logger.debug("Processing enhanced query request...")
    logger.debug(f"Query: {q}")
    
    if not verify_document_state():
        logger.warning("Document state verification failed")
        return JSONResponse({
            "error": "Document state invalid",
            "message": "Please upload your document again to continue the conversation."
        })

    logger.debug("Document state verified, building enhanced prompt...")
    prompt = build_enhanced_prompt(q)
    return StreamingResponse(stream_gemini_response(prompt), media_type="text/plain")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
