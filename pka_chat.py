import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI 
import re
import json

# from st_supabase_connection import SupabaseConnection

# # Database configuration
# DB_HOST = "localhost"
# DB_NAME = "pka_books"
# DB_USER = "postgres"
# DB_PASSWORD = "pass4123"
DB_HOST = st.secrets["DB_HOST"] 
DB_NAME =st.secrets["DB_NAME"] 
DB_USER = st.secrets["DB_USER"] 
DB_PASSWORD = st.secrets["DB_PASSWORD"] 

# Streamlit configuration
st.set_page_config(layout="wide", page_title="From the Library of Prof. Khurshid Ahmad", page_icon="💬")

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
    password=DB_PASSWORD,
    port=5432
)
# pka-ai:us-central1:pka-ai-vdb
# conn2 = st.connection("supabase",type=SupabaseConnection)

register_vector(conn)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Sidebar for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Functions
# Functions
def refine_question(user_question):
    """
    Refine the user's question to extract up to three distinct questions.
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for refining user questions."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Refine: ", response)
    refined_questions = response.choices[0].message.content # response["choices"][0]["message"]["content"]
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
    # print("QR: ", response)
    query_embedding = response.data[0].embedding

    # Perform similarity search
    cursor.execute("""
        SELECT id, title, content, book_title, author, page_number,
               1 - (embedding <=> %s::vector) AS similarity
        FROM public.documents
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY similarity DESC
        LIMIT 5;
    """, (query_embedding, query_embedding, SIMILARITY_THRESHOLD))
    results = cursor.fetchall()

    # # Perform similarity search
    # query = """
    #     SELECT id, title, content, book_title, author, page_number,
    #         1 - (embedding <=> %s::vector) AS similarity
    #     FROM public.documents
    #     WHERE 1 - (embedding <=> %s::vector) >= %s
    #     ORDER BY similarity DESC
    #     LIMIT 5;
    # """
    # # Parameters for the query
    # params = (query_embedding, query_embedding, SIMILARITY_THRESHOLD)

    # # Execute the query
    # with conn.cursor() as cursor:
    #     cursor.execute(query, params)
    #     results = cursor.fetchall()

    # # Process results
    # for result in results:
    #     print(result)
    # print("Refs:", results)
    return results

def generate_response(query, context, formatting):
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
       # Refine the user's question
    refined_questions_json = refine_question(user_input)
    print("Refined2: ", refined_questions_json)
    try:
        parsed_data = json.loads(refined_questions_json)
        question_en = parsed_data["Question"]["en"]  # English version of the question
        question_ur = parsed_data["Question"]["ur"]  # Urdu version of the question
        formatting = parsed_data["Formatting"]  # Formatting instructions

        if not formatting:
            formatting = "No specific formatting instructions provided."

        # Output extracted information
        print(f"English Question: {question_en}")
        print(f"Urdu Question: {question_ur}")
        print(f"Formatting Instructions: {formatting}")

        # st.json(refined_questions_json)

        # Parse refined questions
        refined_questions = [ question_en, question_ur] # eval(refined_questions_json)["questions"]
    except Exception as e:
        st.error("Error parsing refined questions. Please try again." + str(e))
        refined_questions = []

    final_response = ""
    all_references = []
    # Process each refined question
    for question in refined_questions:
        # st.markdown(f"**Processing question:** {question}")

        # Query the vector database
        references = query_vector_db(question)

        print("Refs: ", len(references))
        # Append references if not already present in all_references
        if references:
            for ref in references:
                # if ref not in all_references:
                exists = False
                for all_ref in all_references:
                    if ref['id'] == all_ref['id']:
                        exists = True
                        break
                if not exists:
                    all_references.append(ref)

    # Generate response
    response, found = generate_response(user_input, all_references, formatting)
    # # Query the vector database
    # references = query_vector_db(user_input)
    
    # # Generate response
    # response, found = generate_response(user_input, references)

    # if found:
    # Format the excerpts and extract numbers
    formatted_response, extracted_numbers = format_excerpts_and_extract_numbers(response)
    if not formatted_response:
        formatted_response = response
        extracted_numbers = []
    else:
        final_response += formatted_response + "\n"
    
    refsUsed = []
    if found:
        for i, ref in enumerate(all_references, start=1):
            if i in extracted_numbers:
                # print(f"Excerpt {i}: {ref['content']}")
                # if ref not in all_references:
                # exists = False
                # for all_ref in all_references:
                #     if ref['id'] == all_ref['id']:
                #         exists = True
                #         break
                # if not exists:
                #     all_references.append(ref)
                refsUsed.append(ref)

    # Update chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "assistant": final_response,
        "references": all_references,
        "found": len(refsUsed) > 0
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
    # print(chat["excerpts"])
    if chat["references"] and chat["found"]:
        st.markdown("**References:**")
        for i, ref in enumerate(chat["references"], start=1):
            if i in chat["excerpts"]:
                content_some = ref['content'][:ref_content_len] + "..." if len(ref['content']) > ref_content_len else ref['content']
                st.markdown(f""":blue
                            - **Excerpt {i}:** 
                            {content_some} \n
                            [REFERENCE from '{ref['book_title']}', Page: {ref['page_number']}, Similarity: {ref['similarity']}]
                            """)
    st.markdown("---")

