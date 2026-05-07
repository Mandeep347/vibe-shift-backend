# Vive-Shift Pipeline Server

AI-powered multimodal music mood analysis pipeline built with FastAPI. The system combines audio signal processing, lyrics analysis, LLM-assisted sentiment estimation, and a Hugging Face prediction service to estimate the emotional valence of songs and recommend mood-improving tracks.

## Overview

Vibe-Shift is a backend inference pipeline that:

* Downloads and processes song audio from YouTube
* Extracts audio features using Librosa
* Fetches lyrics from Genius
* Combines audio + lyrics for multimodal mood prediction
* Uses caching with Supabase for low-latency repeated requests
* Recommends higher-valence songs for mood uplift
* Exposes a production-ready REST API with FastAPI

The project is designed for deployment on cloud environments such as Hugging Face Spaces, Azure VPS, or Docker-based infrastructure.

---

# Features

## Core Features

* Multimodal sentiment prediction using:

  * Audio features
  * Lyrics
  * LLM-assisted sentiment estimation
* Automatic YouTube audio fetching using yt-dlp
* Intelligent 30-second highlight extraction
* Librosa-based feature engineering
* Lyrics fetching from Genius API
* Hugging Face model inference integration
* FastAPI REST endpoints
* Supabase caching layer
* Mood uplift recommendations
* Docker support
* Production-ready CORS support
* SSL/network hardening for cloud deployments

## Audio Processing Features

* MFCC extraction
* Chroma feature extraction
* Spectral centroid analysis
* Spectral bandwidth analysis
* RMS energy analysis
* Zero Crossing Rate calculation
* Energetic segment detection

## Reliability Features

* Retry logic for remote inference
* Fallback lyrics support
* Fallback audio support
* Graceful API degradation
* Temporary file cleanup
* Cached repeated predictions

---

# Tech Stack

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

## Audio & ML

* Librosa
* NumPy
* SoundFile
* Hugging Face Inference Endpoint

## External Services

* Supabase
* Genius Lyrics API
* Groq API
* Google Gemini API
* YouTube (via yt-dlp)

## Deployment

* Docker
* Hugging Face Spaces
* Azure VPS compatible

---

# Project Structure

```txt
mood-transformation/
│
├── pipeline_server.py      # Main FastAPI pipeline server
├── utils.py                # LLM-based sentiment utilities
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment config
├── fallback.txt            # Default fallback lyrics
├── cookies.txt             # YouTube cookies for yt-dlp
├── .env                    # Environment variables
├── .gitignore
└── README.md
```

---

# System Architecture

```txt
                ┌─────────────────────┐
                │   Client Request    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ FastAPI /run API    │
                └──────────┬──────────┘
                           │
               Cache Hit?  │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌──────────────────┐             ┌────────────────────┐
│ Supabase Cache   │             │ Audio Download     │
│ Instant Response │             │ yt-dlp + YouTube   │
└──────────────────┘             └─────────┬──────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ Audio Feature Pipeline │
                              │ Librosa + NumPy        │
                              └─────────┬──────────────┘
                                        │
                                        ▼
                             ┌─────────────────────────┐
                             │ Lyrics Fetching         │
                             │ Genius API              │
                             └─────────┬───────────────┘
                                       │
                                       ▼
                            ┌──────────────────────────┐
                            │ LLM Sentiment Estimation │
                            │ Groq / Gemini            │
                            └─────────┬────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────────┐
                           │ Hugging Face Prediction  │
                           └─────────┬────────────────┘
                                     │
                                     ▼
                         ┌────────────────────────────┐
                         │ Final Valence + Recommend  │
                         └────────────────────────────┘
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/mood-transformation.git
cd mood-transformation
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Linux/macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
# Supabase
SUPABASE_URL=
SUPABASE_ANON_KEY=

# APIs
GENIUS_TOKEN=
mnd_groq_key=
mnd_gemini_key=

# Fallback audio URL
default_audio=

# yt-dlp cookies
COOKIES_PATH=/app/cookies.txt
```

---

# Running Locally

```bash
python pipeline_server.py
```

Server:

```txt
http://localhost:7860
```

Swagger Docs:

```txt
http://localhost:7860/docs
```

---

# Docker Deployment

## Build Image

```bash
docker build -t moodshift .
```

## Run Container

```bash
docker run -p 7860:7860 --env-file .env moodshift
```

---

# API Documentation

## Base URL

```txt
http://localhost:7860
```

---

## POST /run

Runs the complete mood analysis pipeline.

### Request

```json
{
  "song": "Blinding Lights",
  "artist": "The Weeknd",
  "domain": 0
}
```

### Domain Values

| Value | Language |
| ----- | -------- |
| 0     | English  |
| 1     | Hindi    |

### Response

```json
{
  "song": "Blinding Lights",
  "artist": "The Weeknd",
  "domain": 0,
  "valence": 0.84,
  "confidence": "high",
  "valence_zscore": 1.42,
  "latency_ms": 5321,
  "pipeline_ms": 4910,
  "lyrics_chars": 2145,
  "from_hf": true,
  "from_cache": false,
  "recommendation": []
}
```

---

## GET /playlist

Returns cached songs sorted by valence.

### Example

```http
GET /playlist?domain=0
```

### Response

```json
{
  "songs": [],
  "count": 0
}
```

---

## GET /health

Health check endpoint.

### Response

```json
{
  "status": "ok",
  "hf_endpoint": "https://your-hf-space/predict",
  "supabase": "connected",
  "cookies": "found"
}
```

---

## GET /

Basic server status endpoint.

---

# Audio Feature Pipeline

The model extracts 29 audio features:

## MFCC Features

* mfcc_1 → mfcc_13

## Chroma Features

* chroma_1 → chroma_12

## Spectral Features

* spectral_centroid
* spectral_bandwidth
* zcr
* rms

---

# Prediction Workflow

```txt
Song + Artist
      │
      ▼
Download Audio
      │
      ▼
Extract Highlight Segment
      │
      ▼
Generate Audio Features
      │
      ▼
Fetch Lyrics
      │
      ▼
LLM Sentiment Estimation
      │
      ▼
Hugging Face Prediction
      │
      ▼
Final Valence Score
      │
      ▼
Recommendation Generation
```

---

# Screenshots

Add your UI/backend screenshots here.

## Swagger API Docs

```md
![Swagger Docs](./screenshots/swagger.png)
```

## API Response Example

```md
![API Response](./screenshots/api-response.png)
```

## Architecture Diagram

```md
![Architecture](./screenshots/architecture.png)
```

---

# Deployment Notes

## Recommended Deployment Setup

### Backend VPS

Deploy the FastAPI server on:

* Azure VPS
* AWS EC2
* DigitalOcean
* Railway
* Render

### Model Inference

Use a separate Hugging Face Space or GPU server for inference.

### Database

Supabase Postgres is used as:

* Cache layer
* Recommendation source
* Persistent song storage

---

# Performance Optimizations

* Cached predictions avoid recomputation
* Temporary file cleanup reduces storage overhead
* Highlight extraction avoids full-song processing
* Remote inference separated from API layer
* SSL hardening improves HF Spaces reliability

---

# Known Limitations

* yt-dlp reliability depends on YouTube changes
* Lyrics API may occasionally fail
* Long inference times for uncached songs
* Requires external services for full functionality
* Current recommendation system uses valence-only ranking

---

# Future Improvements

* Real-time streaming inference
* Playlist generation endpoint
* User authentication
* Advanced recommendation engine
* Emotion classification beyond valence
* Frontend dashboard
* Spotify integration
* Batch prediction support
* Background task queue
* Redis caching

---

# Security Notes

Do not commit:

* `.env`
* API keys
* cookies.txt

Use environment variables in production.

---

# License

This project is intended for educational and research purposes.

Add your preferred open-source license if distributing publicly.

---

# Author

Developed by:
Mandeep Chauhan
Deepanshi Kashyap
Aakanksha
Shobha Kumari
 - at NITRA Technical Campus, Ghaziabad
