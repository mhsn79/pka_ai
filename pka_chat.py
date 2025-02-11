import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI 
import re
import json
import time
from upstash_redis import Redis
# import extra_streamlit_components as stx
from streamlit_local_storage import LocalStorage
import urllib
from datetime import datetime

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

REDIS_DB = st.secrets["REDIS_DB"]
REDIS_TOKEN = st.secrets["REDIS_TOKEN"]

redis = Redis(url=REDIS_DB, token=REDIS_TOKEN)

redis.set("foo", "bar")
value = redis.get("foo")

print("Value: ", value)

session_key = None
# counter = 0
current_conversation = None
ref_content_len = 500
button_keys = []
history_data = {}

# Streamlit configuration
st.set_page_config(layout="wide", page_title="From the Library of Prof. Khurshid Ahmad", page_icon="💬", )

# cookie_manager = stx.CookieManager()
# session_key = cookie_manager.get("pka_ai_session_id")
# print("Session Key from cookie: ", session_key)
# if not session_key:
#     session_key = "pka_ai.session." + datetime.now().strftime("%Y%m%d%H%M%S")
#     print("Creating new session key...", session_key)
#     cookie_manager.set("pka_ai_session_id", session_key)

localS = LocalStorage()
if localS:
    session_key = localS.getItem("pka_ai_session_id")
    if session_key:
        print("Session Key from Local Storage: ", session_key)
    else:
        session_key = "pka_ai.session." + datetime.now().strftime("%Y%m%d%H%M%S")
        print("Creating new session key...", session_key)
        localS.setItem("pka_ai_session_id", session_key)
else:
    print("Error: Local Storage not available. History won't be saved.")

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
    st.session_state["chat_history"] = []
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# Model Choice - Name to be adapted to your deployment
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"
    
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
        model=st.session_state["openai_model"],
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

def get_conversation_title(q):

    # full_text = "".join([item["user"] for item in st.session_state["chat_history"]])
    
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
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
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can summarize information from given context."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    response_text = st.write_stream(response)
    print("Response: ", response_text)
    #return response.choices[0].message.content, True

    # response_text = "" # response.choices[0].message.content
    # for chunk in response:
    #     print("Chunk: ", chunk)
    #     response_text += chunk + " "
    return response_text, True

def stream_data(response_text):
    for word in response_text.split(" "):
        yield word + " "
        time.sleep(0.02)


# if st.button("Stream data"):
#     st.write_stream(stream_data)

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

def display_chat(id):
    current_conversation = id
    disable(True)
    print("Displaying Chat...", id)
    content = history_data.get(id)
    if current_conversation and content:
        # for chat in st.session_state["chat_history"]:
        title = list(content.keys())[0]
        chat = content[title]
        # print("==================== Chat: ", chat)
        if chat:
            # if chat.get("id") and chat["id"] != current_conversation:
            #     continue
            # st.markdown(f"**You asked:** {chat['user']}")
            with placeholder.container():
                st.markdown(f"**You asked:** {chat['user']}")

            st.markdown(f"**PKA AI Assistant:**") # {chat['assistant']}")
            st.write_stream(stream_data(chat['assistant']))
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

    else:
        print("No conversation selected.")
        #st.markdown("No conversation selected.")

def display_history():
    print("Displaying History...")
    st.sidebar.write()
    sc1, sc2 = st.sidebar.columns((6, 1))

    # history_keys = [key for key in reversed(list(st.context.cookies)) if key.startswith('history')]
    history_keys = [key for key in reversed(redis.keys('*')) if key.startswith(session_key)]  #"pka_ai")]  #'history')]

    print("History Keys: ", history_keys)

    for key, conversation_id in enumerate(history_keys):

        print("Key: ", conversation_id)
        # print("data: ", st.context.cookies.get(conversation_id))
        # content = json.loads(urllib.parse.unquote(st.context.cookies.get(conversation_id)))
        content = json.loads(urllib.parse.unquote(redis.get(conversation_id)))
        # print("Content: ", content)

        title = list(content.keys())[0]
        history_data.update({conversation_id: content})

        if conversation_id not in button_keys:
            button_keys.append(conversation_id)

            if sc1.button(title, key=f"c{conversation_id}"):
                #st.sidebar.info(f'Reload "{title}"', icon="💬")
                # current_conversation = conversation_id
                display_chat(conversation_id)
                display_history()

            if sc2.button("❌", key=f"x{conversation_id}"):
                st.sidebar.info("Conversation removed", icon="❌")
                # cookie_manager.delete(conversation_id)
                redis.delete(conversation_id)
                display_history()
            
# Main app
st.title("From the Library of Prof. Khurshid Ahmad 💬")
st.write("Ask questions from Prof. Khurshid Ahmad's Library. This is a Proof of Concept (PoC) for the PKA AI project.")
st.write("You can use any language (English, Urdu, Arabic, Roman Urdu etc.) for asking question.")
st.markdown("---")

# Initialize session state for text input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Function to clear input
def clear_user_input():
    st.session_state.user_input = ""  # Reset session state variable

def disable(b):
    st.session_state["disabled"] = b
disable(False)
# Chat input
user_input = ""
placeholder = st.empty()
with placeholder.container():
    user_input = st.text_input("Type question here...", key="user_input",)

# found = False
if st.button("Submit", key='btn_submit', disabled=st.session_state.get("disabled", False)) and user_input:
    # counter += 1
    disable(True)
    current_conversation = session_key + ".history." + datetime.now().strftime("%Y%m%d%H%M%S")
    # Refine the user's question
    with placeholder.container():
        st.markdown(f"**You asked:** {user_input}")
        st.write("Understanding your question...")
    refined_questions_json = refine_question(user_input)
    print("Refined2: ", refined_questions_json)
    formatting = "No specific formatting instructions provided."
    try:
        parsed_data = json.loads(refined_questions_json)
        question_en = parsed_data["Question"]["en"]  # English version of the question
        question_ur = parsed_data["Question"]["ur"]  # Urdu version of the question
        formatting = parsed_data["Formatting"]  # Formatting instructions

        # if not formatting:
        #     formatting = "No specific formatting instructions provided."

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

    with placeholder.container():
        st.markdown(f"**You asked:** {user_input}")
        st.write("Gathering References from Prof. Khurdhid Ahmad's Library...")

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
    with placeholder.container():
        st.markdown(f"**You asked:** {user_input}")
        st.write("Generating responce...")

    st.markdown(f"**PKA AI Assistant:**")
    response, found = generate_response(user_input, all_references, formatting)
    with placeholder.container():
        st.markdown(f"**You asked:** {user_input}")
        st.write("Generating responce... ... Done.")

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
                refsUsed.append(ref)

    conversation_title = get_conversation_title(user_input)

    cur_chat = {
        "id": current_conversation,
        "user": user_input,
        "assistant": final_response,
        "title": conversation_title,
        "references": all_references,
        "found": len(refsUsed) > 0,
        "excerpts": extracted_numbers
    }
    # Update chat history
    st.session_state["chat_history"].append(cur_chat)
    st.session_state["conversation_id"] = current_conversation
    redis.set(st.session_state["conversation_id"], urllib.parse.quote(json.dumps({conversation_title: cur_chat}))) # st.session_state["chat_history"])))

    # print(chat["excerpts"])
    if cur_chat["references"] and cur_chat["found"]:
        st.markdown("**References:**")
        for i, ref in enumerate(cur_chat["references"], start=1):
            if i in cur_chat["excerpts"]:
                content_some = ref['content'][:ref_content_len] + "..." if len(ref['content']) > ref_content_len else ref['content']
                st.markdown(f""":blue
                            - **Excerpt {i}:** 
                            {content_some} \n
                            [REFERENCE from '{ref['book_title']}', Page: {ref['page_number']}, Similarity: {ref['similarity']}]
                            """)
    st.markdown("---")


# Display chat history
# st.sidebar.title("Chat History")
# for chat in st.session_state.chat_history:
#     st.sidebar.markdown(f"**You asked:** {chat['user']}")
#     # st.sidebar.markdown(f"**From PKA Library:** {chat['assistant']}")

# Display chat
display_chat(current_conversation)

# # If there is at least one message in the chat, we display the options
# if len(st.session_state["chat_history"]) > 0:
#     if "conversation_id" in st.session_state:
#         print("Conversation ID in State: ", st.session_state["conversation_id"])
#     if "conversation_id" not in st.session_state:
#         st.session_state["conversation_id"] = session_key + ".history_" + datetime.now().strftime("%Y%m%d%H%M%S")
#         print("Conversation ID: ", st.session_state["conversation_id"])


# if "conversation_id" in st.session_state:

#     conversation_title = get_conversation_title()
#     print("Conversation Title: ", conversation_title)
#     # cookie_manager.set(st.session_state["conversation_id"], val={conversation_title: st.session_state["chat_history"]})
#     redis.set(st.session_state["conversation_id"], urllib.parse.quote(json.dumps({conversation_title: st.session_state["chat_history"]})))

st.sidebar.header("Past Conversations")

if st.sidebar.button("New Conversation"):
    st.session_state["conversation_id"] = None
    current_conversation = None
    display_chat(None)
    # clear_user_input()

display_history()
