import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI 
import re

# # Database configuration
# DB_HOST = "localhost"
# DB_NAME = "pka_books"
# DB_USER = "postgres"
# DB_PASSWORD = "pass4123"
DB_HOST = st.secrets["DB_HOST"] 
DB_NAME =st.secrets["DB_NAME"] 
DB_USER = st.secrets["DB_USER"] 
DB_PASSWORD = st.secrets["DB_PASSWORD"] 

# OpenAI API configuration
# # OPENAI_API_KEY = "your_openai_api_key"
# openai.api_key = OPENAI_API_KEY
api_key = st.secrets["CAPI_KEY"] 
client = OpenAI(
  api_key=api_key # os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.7

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
# pka-ai:us-central1:pka-ai-vdb

register_vector(conn)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Ask Prof. Khurshid Ahmad", page_icon="💬")

# Sidebar for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Functions
def query_vector_db(query_text):
    """
    Query the PostgreSQL vector database to retrieve similar texts.
    """
    # Generate embedding for the query
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    # print("QR: ", response)
    query_embedding = response.data[0].embedding

    # Perform similarity search
    cursor.execute("""
        SELECT id, title, content, book_title, author, page_number,
               1 - (embedding <=> %s::vector) AS similarity
        FROM public.documents
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY similarity DESC
        LIMIT 7;
    """, (query_embedding, query_embedding, SIMILARITY_THRESHOLD))
    results = cursor.fetchall()
    # print("Refs:", results)
    return results

def generate_response(query, context):
    """
    Generate a response using OpenAI GPT based on the provided context.
    """
    if not context:
        return "I'm sorry, I couldn't find any relevant information in the database.", False

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

    Your response should:
    - Answer the question concisely and accurately.
    - Include numbered references for the excerpts that support your answer, at the end in the following Suggested format:
        - Excerpt [[[1]]]: [Book Title], Page: 123
    - If no relevant information is available, say "I couldn't find relevant information."

    Response:
    """
    
    # Call OpenAI API
    response = client.chat.completions.create( # openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can summarize information from given context."},
            {"role": "user", "content": prompt}
        ]
    )
    # print("Response: ", response)
    return response.choices[0].message.content, True

def format_excerpts_and_extract_numbers(response_text):
    """
    Extract excerpt numbers and replace the format 'Excerpt [[[n]]]' with 'Excerpt [n]'
    in the given response text. Also, return the list of extracted numbers.

    Args:
    response_text (str): The text containing excerpt references.

    Returns:
    tuple: A tuple containing the formatted text and a list of extracted numbers.
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

# Main app
st.title("From the Library of Prof. Khurshid Ahmad 💬")
st.write("Ask questions from Prof. Khurshid Ahmad's Library. This is a Proof of Concept (PoC) for the PKA AI project.")
st.write("You can use any language (English, Urdu, Arabic, Roman Urdu etc.) for asking question.")

# Chat input
user_input = st.text_input("Type question here...", key="user_input")
# found = False
if st.button("Submit") and user_input:
    # Query the vector database
    references = query_vector_db(user_input)
    
    # Generate response
    response, found = generate_response(user_input, references)

    # if found:
    # Format the excerpts and extract numbers
    formatted_response, extracted_numbers = format_excerpts_and_extract_numbers(response)
    if not formatted_response:
        formatted_response = response
        extracted_numbers = []
        
    # Update chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "assistant": formatted_response,
        "references": references,
        "found": found,
        "excerpts": extracted_numbers
    })

# Display chat history
st.sidebar.title("Chat History")
for chat in st.session_state.chat_history:
    st.sidebar.markdown(f"**You asked:** {chat['user']}")
    # st.sidebar.markdown(f"**From PKA Library:** {chat['assistant']}")

ref_content_len = 500
# Display chat
for chat in st.session_state.chat_history:
    st.markdown(f"**You asked:** {chat['user']}")
    st.markdown(f"**PKA AI Assistant:** {chat['assistant']}")
    print(chat["excerpts"])
    if chat["references"] and chat["found"]:
        st.markdown("**References:**")
        for i, ref in enumerate(chat["references"], start=1):
            if i in chat["excerpts"]:
                content_some = ref['content'][:ref_content_len] + "..." if len(ref['content']) > ref_content_len else ref['content']
                st.markdown(f""":blue
                            - **Excerpt {i}:** 
                            {content_some} \n
                            [REFERENCE from '{ref['book_title']}', Page: {ref['page_number']}]
                            """)
    st.markdown("---")

