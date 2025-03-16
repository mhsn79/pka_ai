import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import re
import json
import time
from upstash_redis import Redis
import urllib
from datetime import datetime
from flask import Flask, request, jsonify, Response, stream_with_context
import uuid
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Initialize rate limiter with memory storage for testing
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",  # Use in-memory storage for testing
    default_limits=["200 per day", "50 per hour"]  # Default limits
)

# Database configuration - load from environment variables
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Redis configuration
REDIS_DB = os.environ.get("REDIS_DB")
REDIS_TOKEN = os.environ.get("REDIS_TOKEN")

redis = Redis(url=REDIS_DB, token=REDIS_TOKEN)

# OpenAI API configuration
api_key = os.environ.get("CAPI_KEY")
client = OpenAI(api_key=api_key)

# Default OpenAI model
DEFAULT_MODEL = "gpt-4o"

# Similarity threshold for vector searches
SIMILARITY_THRESHOLD = 0.7

# Max content length for reference display
ref_content_len = 500

# Initialize PostgreSQL connection
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=5432
    )
    register_vector(conn)
    return conn

# Functions
def refine_question(user_question, model=DEFAULT_MODEL):
    """
    Refine the user's question to extract questions in English and Urdu.
    """
    prompt = f"""
    User's input is a question for chatbot. User may have asked question in unclear way and may have added unnecessary text and formatting intructions within the question. 
    Your job is to separate the formatting intrections and provide refined question with clear keywords. Provide Question in English and it's translation in Urdu Language. 
    Ensure the output strictly adheres to the following JSON format without quotes:
    
    {{
        "Question": {{
            "en": "User's question in English",
            "ur": "User's question in Urdu"
        }},
        "Formatting": "Formatting instructions"
    }}
    
    User's input:
    {user_question}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for refining user questions."},
            {"role": "user", "content": prompt}
        ]
    )
    refined_questions = response.choices[0].message.content
    return refined_questions

def query_vector_db(query_text):
    """
    Query the PostgreSQL vector database to retrieve similar texts.
    """
    # Generate embedding for the query
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding

    # Perform similarity search
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT id, title, content, book_title, author, page_number,
               1 - (embedding <=> %s::vector) AS similarity
        FROM public.documents
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY similarity DESC
        LIMIT 5;
    """, (query_embedding, query_embedding, SIMILARITY_THRESHOLD))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results

def get_conversation_title(q, model=DEFAULT_MODEL):
    """
    Generate a title for the conversation based on the question.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Summarize the following conversation in 3 words:"
                + q,
            },
        ],
        stop=None,
    )
    conversation_title = response.choices[0].message.content
    return conversation_title

def generate_response(query, context, formatting, model=DEFAULT_MODEL):
    """
    Generate a response using OpenAI GPT based on the provided context.
    """
    if not context:
        return "I'm sorry, I couldn't find any relevant information in the database.", False, []

    # Format references for GPT prompt
    context_text = "\n\n---\n\n".join([
        f'Excerpt {i+1} from the book "{item["book_title"]}", Chapter: "{item["title"]}", Page: {item["page_number"]}:\n\n"{item["content"]}"'
        for i, item in enumerate(context)
    ])

    # Create GPT prompt
    prompt = f"""
    You are an AI assistant. Answer the question based on the following excerpts from books and articles of Prof. Khurshid Ahmad. Also provide references.

    Question: {query}

    Excerpts:
    {context_text}

    Formatting Instructions:
    - {formatting}

    Your response should:
    - Answer the question elaboratively and accurately.
    - Include relevant information from the excerpts.
    - Include numbered references for the excerpts that support your answer, at the end in the following Suggested format:
        - Excerpt [[[1]]]: [Book Title], Page: 123
    - If no relevant information is available, say "I couldn't find relevant information."

    Response:
    """

    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can summarize information from given context."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    return response, True

def format_excerpts_and_extract_numbers(response_text):
    """
    Extract excerpt numbers and replace the format 'Excerpt [[[n]]]' with 'Excerpt [n]'
    in the given response text. Also, return the list of extracted numbers.
    """
    # Regular expression to find 'Excerpt [[[n]]]'
    pattern = r'\[\[\[(\d+)\]\]\]'
    
    # Extract all numbers into a list
    numbers = re.findall(pattern, response_text)
    
    # Replace with 'Excerpt [n]'
    formatted_text = re.sub(pattern, r'[\1]', response_text)
    
    # Convert extracted numbers to integers
    numbers = list(map(int, numbers))
    
    return formatted_text, numbers

# API Endpoints with rate limiting
@app.route('/api/session/create', methods=['GET'])
@limiter.limit("30 per minute")  # Limit session creation
def create_session():
    """
    Create a new unique session token and return it
    """
    session_key = f"pka_ai.session.{datetime.now().strftime('%Y%m%d%H%M%S')}.{str(uuid.uuid4())[:8]}"
    return jsonify({
        "session_token": session_key,
        "created_at": datetime.now().isoformat()
    })

@app.route('/api/history', methods=['GET'])
@limiter.limit("60 per minute")  # Limit history requests
def get_chat_history():
    """
    Return chat history for a given session token
    """
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({"error": "No session token provided"}), 400
    
    # Get all history keys for this session
    history_keys = [key for key in reversed(redis.keys('*')) if key.startswith(session_token)]
    
    history = []
    for key in history_keys:
        content = json.loads(urllib.parse.unquote(redis.get(key)))
        title = list(content.keys())[0]
        chat = content[title]
        
        # Format references for API response
        formatted_refs = []
        if chat["found"] and chat["references"]:
            for i, ref in enumerate(chat["references"], start=1):
                if i in chat["excerpts"]:
                    content_preview = ref['content'][:ref_content_len] + "..." if len(ref['content']) > ref_content_len else ref['content']
                    formatted_refs.append({
                        "excerpt_number": i,
                        "content_preview": content_preview,
                        "book_title": ref['book_title'],
                        "page_number": ref['page_number'],
                        "similarity": ref['similarity']
                    })
        
        history.append({
            "id": chat["id"],
            "title": title,
            "query": chat["user"],
            "response": chat["assistant"],
            "references": formatted_refs,
            "timestamp": key.split(".")[-1]  # Extract timestamp from key
        })
    
    return jsonify({"history": history})

@app.route('/api/delete_conversation', methods=['DELETE'])
@limiter.limit("30 per minute")  # Limit deletion requests
def delete_conversation():
    """
    Delete a specific conversation
    """
    conversation_id = request.args.get('conversation_id')
    if not conversation_id:
        return jsonify({"error": "No conversation ID provided"}), 400
    
    redis.delete(conversation_id)
    return jsonify({"status": "success", "message": "Conversation deleted"})

@app.route('/api/chat', methods=['POST'])
@limiter.limit("20 per minute, 100 per hour")  # Stricter limits for chat endpoint
def chat():
    """
    Process a chat request and return a streamed response
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_input = data.get('query')
    session_token = data.get('session_token')
    model = data.get('model', DEFAULT_MODEL)
    
    if not user_input or not session_token:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Create conversation ID
    current_conversation = f"{session_token}.history.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Process the query
    def generate():
        # Step 1: Yield processing status
        yield json.dumps({"status": "processing", "step": "refining_question"}) + "\n"
        
        # Step 2: Refine the question
        refined_questions_json = refine_question(user_input, model)
        try:
            parsed_data = json.loads(refined_questions_json)
            question_en = parsed_data["Question"]["en"]
            question_ur = parsed_data["Question"]["ur"]
            formatting = parsed_data["Formatting"] or "No specific formatting instructions provided."
            refined_questions = [question_en, question_ur]
            
            yield json.dumps({
                "status": "processing", 
                "step": "question_refined",
                "refined_questions": {
                    "en": question_en,
                    "ur": question_ur
                },
                "formatting": formatting
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": f"Error parsing refined questions: {str(e)}"}) + "\n"
            return
        
        # Step 3: Query the vector database
        yield json.dumps({"status": "processing", "step": "gathering_references"}) + "\n"
        
        all_references = []
        for question in refined_questions:
            references = query_vector_db(question)
            if references:
                for ref in references:
                    exists = False
                    for all_ref in all_references:
                        if ref['id'] == all_ref['id']:
                            exists = True
                            break
                    if not exists:
                        all_references.append(ref)
        
        # Step 4: Generate response
        yield json.dumps({"status": "processing", "step": "generating_response"}) + "\n"
        
        response_stream, found = generate_response(user_input, all_references, formatting, model)
        
        # Step 5: Stream the response
        full_response = ""
        yield json.dumps({"status": "streaming", "step": "response_text"}) + "\n"
        
        for chunk in response_stream:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield json.dumps({"type": "content", "content": content}) + "\n"
        
        # Step 6: Process and format the response
        formatted_response, extracted_numbers = format_excerpts_and_extract_numbers(full_response)
        if not formatted_response:
            formatted_response = full_response
            extracted_numbers = []
            
        # Get references used in the response
        refs_used = []
        if found:
            for i, ref in enumerate(all_references, start=1):
                if i in extracted_numbers:
                    refs_used.append({
                        "excerpt_number": i,
                        "content": ref['content'][:ref_content_len] + "..." if len(ref['content']) > ref_content_len else ref['content'],
                        "book_title": ref['book_title'],
                        "page_number": ref['page_number'],
                        "similarity": ref['similarity']
                    })
        
        # Step 7: Generate conversation title
        conversation_title = get_conversation_title(user_input, model)
        
        # Step 8: Save to Redis
        chat_data = {
            "id": current_conversation,
            "user": user_input,
            "assistant": formatted_response,
            "title": conversation_title,
            "references": all_references,
            "found": len(refs_used) > 0,
            "excerpts": extracted_numbers
        }
        
        redis.set(current_conversation, urllib.parse.quote(json.dumps({conversation_title: chat_data})))
        
        # Step 9: Send final summary
        yield json.dumps({
            "status": "complete",
            "conversation_id": current_conversation,
            "title": conversation_title,
            "query": user_input,
            "response": formatted_response,
            "references": refs_used
        }) + "\n"
    
    # Return a streaming response
    return Response(stream_with_context(generate()), content_type='application/json')

# Error handlers for rate limiting
@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded error"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "retry_after": int(e.retry_after)
    }), 429

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Default to port 5001
    app.run(host='0.0.0.0', port=port, debug=True)  # Enable debug mode for testing
    