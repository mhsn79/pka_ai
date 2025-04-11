import os
from PyPDF2 import PdfReader
# from langchain_openai import OpenAIEmbeddings
# import openai
from openai import OpenAI
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from docx import Document
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = {
    'DB_HOST': os.environ.get('DB_HOST'),
    'DB_NAME': os.environ.get('DB_NAME'),
    'DB_USER': os.environ.get('DB_USER'),
    'DB_PASSWORD': os.environ.get('DB_PASSWORD'),
    'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY')
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Database connection parameters from environment variables
DB_HOST = required_env_vars['DB_HOST']
DB_NAME = required_env_vars['DB_NAME']
DB_USER = required_env_vars['DB_USER']
DB_PASSWORD = required_env_vars['DB_PASSWORD']

# OpenAI API configuration
client = OpenAI(api_key=required_env_vars['OPENAI_API_KEY'])

# Initialize LangChain OpenAI Embeddings
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
register_vector(conn)
cursor = conn.cursor()

# Ensure the required table exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    book_title TEXT,
    page_number INT,
    author TEXT,
    content TEXT,
    embedding VECTOR(1536), -- Adjust dimension to match OpenAI's embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);
""")
# Create a trigger function to automatically update `updated_at` field on row update
# cursor.execute("""
# DO $$
# BEGIN
#     IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at') THEN
#         CREATE OR REPLACE FUNCTION update_updated_at()
#         RETURNS TRIGGER AS $$
#         BEGIN
#             NEW.updated_at = CURRENT_TIMESTAMP;
#             RETURN NEW;
#         END;
#         $$ LANGUAGE plpgsql;
#     END IF;
# END;
# $$;

# -- Create the trigger to call the function on UPDATE
# CREATE OR REPLACE TRIGGER set_updated_at
# BEFORE UPDATE ON documents
# FOR EACH ROW
# EXECUTE FUNCTION update_updated_at();
# """)
conn.commit()

def extract_metadata(pdf_path):
    """Extract metadata and content from a PDF file."""
    reader = PdfReader(pdf_path)
    metadata = reader.metadata
    if not metadata:
        metadata = {}
    author = metadata.get("/Author", "Unknown")
    title = metadata.get("/Title", os.path.basename(pdf_path))
    book_title = title  # Assuming PDF title matches book title
    return reader, author, title, book_title

def create_embedding(text, model="text-embedding-ada-002"): #"text-embedding-3-small"):
    """
    Generate embeddings for the given text using OpenAI's embedding model.
    :param text: The input text to generate embeddings for.
    :param model: The OpenAI embedding model to use.
    :return: A list of embeddings.
    """
    try:
        response = client.embeddings.create(   #openai.Embedding.create(
            input=text,
            model=model
        )
        return response.data[0].embedding # response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_pdf(pdf_path):
    """Process a single PDF and store its data into the database."""
    reader, author, title, book_title = extract_metadata(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        content = page.extract_text().strip()
        if not content:
            continue  # Skip empty pages

        # Generate embedding for the page content
        # embedding = embedding_model.embed_query(content)

        # Generate embedding for the page content using OpenAI API
        embedding = create_embedding(content)
        if not embedding:
            print(f"Failed to generate embedding for {title}, page {page_number}")
            continue

        # # Insert data into PostgreSQL
        # cursor.execute("""
        #     INSERT INTO documents (title, book_title, page_number, author, content, embedding)
        #     VALUES (%s, %s, %s, %s, %s, %s);
        # """, (title, book_title, page_number, author, content, embedding))
        # Check if an entry with the same title and page_number exists
        cursor.execute("""
            SELECT id FROM documents WHERE title = %s AND page_number = %s;
        """, (title, page_number))
        result = cursor.fetchone()
        
        if result:
            print("Skipping existing record")
            return
            # Update existing record
            cursor.execute("""
                UPDATE documents
                SET book_title = %s, author = %s, content = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (book_title, author, content, embedding, result[0]))
        else:
            # Insert a new record
            cursor.execute("""
                INSERT INTO documents (title, book_title, page_number, author, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (title, book_title, page_number, author, content, embedding))
    conn.commit()

def process_docx(docx_path):
    """Process a single DOCX file and store its data into the database."""
    # Extract metadata from the docx file
    author = "Prof. Khurshid Ahmad"  # Replace with actual logic if metadata is available
    title = docx_path.split("/")[-1]  # Use the filename as the title
    book_title = title  # Assuming the title matches the book title

    print(f"Processing docx: {book_title}...")

    # Read the DOCX file
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Tokenize each paragraph into sentences
    tokenized_paragraphs = [sent_tokenize(paragraph) for paragraph in paragraphs]

    # Merge short paragraphs and split long paragraphs
    processed_chunks = []
    current_chunk = []
    current_paragraph_numbers = []

    for i, sentences in enumerate(tokenized_paragraphs):
        paragraph_number = i + 1

        # Merge short paragraphs
        if len(sentences) <= 5:
            if current_chunk and len(current_chunk) + len(sentences) >= 10:
                processed_chunks.append({
                    "content": " ".join(current_chunk),
                    "paragraph_numbers": current_paragraph_numbers
                })
                current_chunk = []
                current_paragraph_numbers = []

            current_chunk.extend(sentences)
            current_paragraph_numbers.append(paragraph_number)
        else:
            # Add current_chunk if it's not empty
            if current_chunk:
                processed_chunks.append({
                    "content": " ".join(current_chunk),
                    "paragraph_numbers": current_paragraph_numbers
                })
                current_chunk = []
                current_paragraph_numbers = []

            # Split long paragraphs
            while len(sentences) > 10:
                processed_chunks.append({
                    "content": " ".join(sentences[:10]),
                    "paragraph_numbers": [paragraph_number]
                })
                sentences = sentences[10:]

            # Add remaining sentences to current chunk
            current_chunk = sentences
            current_paragraph_numbers = [paragraph_number]

    # Add the last chunk if it exists
    if current_chunk:
        processed_chunks.append({
            "content": " ".join(current_chunk),
            "paragraph_numbers": current_paragraph_numbers
        })

    # Process each chunk
    for chunk_index, chunk_data in enumerate(processed_chunks, start=1):
        content = chunk_data["content"]
        # paragraph_numbers = ", ".join(map(str, chunk_data["paragraph_numbers"]))
        # print(content)
        # print("----------------------------------------")
        # continue
        if not content.strip():
            continue  # Skip empty chunks
        # Generate embedding for the chunk
        embedding = create_embedding(content)
        if not embedding:
            print(f"Failed to generate embedding for {title}, chunk {chunk_index}")
            continue

        # Check if an entry with the same title and chunk index exists
        cursor.execute("""
            SELECT id FROM documents WHERE title = %s AND page_number = %s;
        """, (title, chunk_index))
        result = cursor.fetchone()

        if result:
            print(f"Skipping existing record for chunk {chunk_index}")
            return
            # Update existing record
            cursor.execute("""
                UPDATE documents
                SET book_title = %s, author = %s, content = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (book_title, author, content, embedding, result[0]))
        else:
            # Insert a new record
            cursor.execute("""
                INSERT INTO documents (title, book_title, page_number, author, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (title, book_title, chunk_index, author, content, embedding))

    conn.commit()


def process_directory(directory_path):
    """Process all PDF files in the given directory."""
    i = 0
    for filename in os.listdir(directory_path):
        # if filename.lower().endswith(".pdf"):
            # i += 1
            # pdf_path = os.path.join(directory_path, filename)
            # print(f"{i}. Processing: {pdf_path}")
            # process_pdf(pdf_path)
        if filename.lower().endswith(".docx"):
            i += 1
            pdf_path = os.path.join(directory_path, filename)
            print(f"{i}. Processing: {pdf_path}")
            process_docx(pdf_path)
    return i

# Directory containing PDF files
PDF_DIRECTORY = "/Users/mohsinsiddiqui/src/mhsn79/IPS/data/Word"

# Process all PDFs in the directory
c = process_directory(PDF_DIRECTORY)

# Close database connection
cursor.close()
conn.close()

print(f"Processed {c} files.")
