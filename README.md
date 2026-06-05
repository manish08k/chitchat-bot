# 🤖 Advanced RAG Chatbot Backend

A production-grade AI chatbot backend with **voice I/O**, **RAG memory**, **PDF analysis**, and **streaming** — powered by Gemini + ChromaDB + sentence-transformers.

---

## 📁 Project Structure

```
chatbot/
├── main.py                    # FastAPI app entry point
├── requirements.txt
├── .env.example               # Copy to .env and fill in your API key
│
├── config/
│   └── settings.py            # Pydantic settings from .env
│
├── core/
│   ├── embeddings.py          # sentence-transformers embedding engine
│   └── llm.py                 # Gemini LLM client (chat + PDF Q&A + streaming)
│
├── vector_db/
│   └── store.py               # ChromaDB: chat memory + PDF knowledge collections
│
├── pdf/
│   └── processor.py           # PDF text extraction + chunking
│
├── voice/
│   └── processor.py           # STT (Google Speech) + TTS (gTTS)
│
├── api/
│   ├── schemas.py             # Pydantic request/response models
│   ├── chat.py                # Chat endpoints (text + voice + stream)
│   ├── pdf_routes.py          # PDF upload/query/delete endpoints
│   └── voice_routes.py        # Standalone TTS + transcription endpoints
│
└── utils/
    └── session.py             # In-memory session manager
```

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note**: On Linux you may need `portaudio19-dev` for audio:
> ```bash
> sudo apt-get install portaudio19-dev ffmpeg
> ```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Run the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🔌 API Reference

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/session` | Create new session |
| `POST` | `/chat/message` | Send text message |
| `POST` | `/chat/voice` | Send voice (WAV) → get response |
| `GET` | `/chat/stream?message=...` | SSE streaming response |
| `GET` | `/chat/history/{session_id}` | Get conversation history |
| `DELETE` | `/chat/session/{session_id}` | Delete session |

#### Example: Text Chat
```bash
# Create session
curl -X POST http://localhost:8000/chat/session

# Send message
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "YOUR_SESSION_ID", "return_audio": true}'
```

#### Example: Voice Chat
```bash
curl -X POST http://localhost:8000/chat/voice \
  -F "audio=@recording.wav" \
  -F "session_id=YOUR_SESSION_ID" \
  -F "return_audio=true"
```

#### Example: Streaming
```javascript
const evtSource = new EventSource(
  "http://localhost:8000/chat/stream?message=Tell+me+about+AI&session_id=xxx"
);
evtSource.onmessage = (e) => {
  if (e.data === "[DONE]") evtSource.close();
  else process.stdout.write(e.data);
};
```

---

### PDF Analyser

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/pdf/upload` | Upload + index a PDF |
| `POST` | `/pdf/query` | Ask question (text) |
| `POST` | `/pdf/voice-query` | Ask question (voice) |
| `GET` | `/pdf/list` | List indexed PDFs |
| `DELETE` | `/pdf/{name}` | Remove PDF from index |

#### Example: Upload & Query PDF
```bash
# Upload
curl -X POST http://localhost:8000/pdf/upload \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/pdf/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "pdf_name": "document.pdf", "return_audio": true}'
```

---

### Voice

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/voice/tts` | Text → MP3 audio |
| `POST` | `/voice/transcribe` | Audio file → transcript |

---

## ⚙️ Configuration (.env)

| Key | Default | Description |
|-----|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `TOP_K_RESULTS` | `5` | RAG retrieval count |
| `CHUNK_SIZE` | `500` | Words per PDF chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping words between chunks |
| `TTS_LANGUAGE` | `en` | gTTS language (`en`, `hi`, etc.) |
| `VOICE_LANGUAGE` | `en` | STT recognition language |

---

## 🏗️ Architecture

```
User Input (text/voice)
        │
        ▼
  [Voice STT]  ←── if audio
        │
        ▼
  [Embedding]  ──→  query vector
        │
        ▼
  [ChromaDB]   ──→  retrieve top-K similar past interactions / PDF chunks
        │
        ▼
  [Gemini LLM] ←── user message + RAG context + session history
        │
        ▼
  [Response]
        │
        ├──→  [ChromaDB] save interaction for future memory
        │
        └──→  [gTTS]  ──→  MP3 audio  (if return_audio=true)
```

---

## 📝 Notes

- **Audio format**: Upload WAV files for best STT accuracy. WebM/OGG from browser MediaRecorder may need conversion (use `ffmpeg` on server or client side).
- **Gemini API key**: Get yours at https://aistudio.google.com/app/apikey (free tier available).
- **RAG memory**: Each chat Q&A pair is embedded and stored. Future queries retrieve semantically similar past interactions automatically.
- **PDF chunking**: Uses word-level sliding window with configurable overlap for better context preservation.
