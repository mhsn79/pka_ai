# PKA Chat API

A Flask-based REST API for interacting with Prof. Khurshid Ahmad's library using AI-powered chat functionality. The API provides endpoints for session management, chat interactions, and conversation history.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [API Documentation](#api-documentation)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Features

- Session-based chat system
- Real-time streaming responses
- Bilingual question processing (English and Urdu)
- Vector database search for relevant content
- Redis-based chat history storage
- Rate limiting for API protection
- OpenAI integration for natural language processing

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- Redis (Upstash)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pka-chat-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
# Database Configuration
DB_HOST=your-db-host.example.com
DB_NAME=pka_books
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Redis Configuration
REDIS_DB=https://your-redis-instance.upstash.io
REDIS_TOKEN=your-redis-token

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key

# Flask Configuration (optional)
FLASK_ENV=development
FLASK_DEBUG=1
```

4. Run the application:
```bash
python pka_chat_api.py
```

The API will be available at `http://localhost:5001`.

## API Documentation

### 1. Create Session
Creates a new unique session for chat interactions.

**Endpoint:** `GET /api/session/create`

**Response:**
```json
{
    "session_token": "pka_ai.session.20240321123456.a1b2c3d4",
    "created_at": "2024-03-21T12:34:56.789Z"
}
```

Keep session token in browser local storage, to retrived chat history and chats with it. 

### 2. Chat
Initiates a chat interaction with streaming response.

**Endpoint:** `POST /api/chat`

**Request Body:**
```json
{
    "session_token": "pka_ai.session.20240321123456.a1b2c3d4",
    "query": "What is role of Islamic economics and finance in eliminating poverty and exploitation of weaker people from society?"
}
```

**Response Stream:**
The response is streamed as Server-Sent Events (SSE) with the following event types:

1. Sample Processing Status:
```json
{
    "status": "processing",
    "step": "refining_question"
}
```

3. Sample Content Chunks:
```json
{
    "type": "content",
    "content": "Computational linguistics is..."
}
```

4. Final Sample Response:
```json
{
    "status": "complete",
    "conversation_id": "pka_ai.session.20240321123456.history.20240321123457",
    "title": "Computational Linguistics Overview",
    "query": "What is computational linguistics?",
    "response": "Computational linguistics is...",
    "references": [
        {
            "excerpt_number": 1,
            "content": "Excerpt from the text...",
            "book_title": "Book Title",
            "page_number": 45,
            "similarity": 0.89
        }
    ]
}
```

### 3. Get Chat History
Retrieves chat history for a specific session.

**Endpoint:** `GET /api/history?session_token=<token>`

**Sample Response:**
```json
{
    "history": [
        {
            "id": "pka_ai.session.20240321123456.history.20240321123457",
            "title": "Computational Linguistics Discussion",
            "query": "What is computational linguistics?",
            "response": "Computational linguistics is...",
            "references": [
                {
                    "excerpt_number": 1,
                    "content_preview": "Excerpt text...",
                    "book_title": "Book Title",
                    "page_number": 45,
                    "similarity": 0.89
                }
            ],
            "timestamp": "20240321123457"
        }
    ]
}
```

### 4. Delete Conversation
Deletes a specific conversation from history.

**Endpoint:** `DELETE /api/delete_conversation?conversation_id=<id>`

**Response:**
```json
{
    "status": "success",
    "message": "Conversation deleted"
}
```

## Rate Limiting

The API implements the following rate limits:

1. Global Limits:
   - 50 requests per day
   - 10 requests per hour
   - 3 requests per minute

2. Endpoint-Specific Limits:
   All endpoints are limited to:
   - 3 requests per minute
   - Subject to global limits

When a rate limit is exceeded, the API returns a 429 status code:
```json
{
    "error": "Rate limit exceeded",
    "message": "<limit description>",
    "retry_after": "<seconds until reset>"
}
```

Best practices for handling rate limits:
1. Implement exponential backoff in your client code
2. Cache responses when possible
3. Monitor the `retry-after` header
4. Batch requests when feasible
5. Store session token securely to avoid unnecessary session creation requests

## Error Handling

Common error responses:

```json
{
    "error": "Error message",
    "message": "Detailed error description"
}
```

HTTP Status Codes:
- 200: Success
- 400: Bad Request
- 429: Rate Limit Exceeded
- 500: Internal Server Error

## Examples

### Using cURL

1. Create a session:
```bash
curl -X GET http://localhost:5001/api/session/create
```

2. Send a chat message:
```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_token": "pka_ai.session.20250316235559.68c49c97",
    "query": "What is computational linguistics?"
  }'
```

3. Get chat history:
```bash
curl -X GET "http://localhost:5001/api/history?session_token=pka_ai.session.20250316235559.68c49c97"
```

4. Delete a conversation:
```bash
curl -X DELETE "http://localhost:5001/api/delete_conversation?conversation_id=pka_ai.session.20250316235559.68c49c97.history.20250316235617"
```

### Using JavaScript

```javascript
// Create a session
async function createSession() {
    const response = await fetch('http://localhost:5001/api/session/create');
    const data = await response.json();
    return data.session_token;
}

// Chat with streaming response
async function chat(sessionToken, query) {
    const response = await fetch('http://localhost:5001/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_token: sessionToken,
            query: query
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        
        const lines = decoder.decode(value).split('\n');
        for (const line of lines) {
            if (line.trim()) {
                const data = JSON.parse(line);
                switch (data.status) {
                    case 'processing':
                        console.log('Processing step:', data.step);
                        break;
                    case 'streaming':
                        console.log('Content chunk:', data.content);
                        break;
                    case 'complete':
                        console.log('Final response:', data);
                        break;
                }
            }
        }
    }
}

// Get chat history
async function getChatHistory(sessionToken) {
    const response = await fetch(`http://localhost:5001/api/history?session_token=${sessionToken}`);
    return await response.json();
}

// Delete conversation
async function deleteConversation(conversationId) {
    const response = await fetch(`http://localhost:5001/api/delete_conversation?conversation_id=${conversationId}`, {
        method: 'DELETE'
    });
    return await response.json();
}
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 