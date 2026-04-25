"""
pipeline_server.py — Moodshift Pipeline Server (Production)
============================================================
Deploys on HuggingFace Spaces (Docker SDK).
Database: Supabase (Postgres) via REST API — no psycopg2 needed.

Endpoints:
  POST /run        — DB cache check → pipeline if miss → save → return
  GET  /playlist   — all songs sorted by valence asc, optional ?domain=0|1
  GET  /health     — status

Environment variables (set as HF Space secrets):
  SUPABASE_URL        e.g. https://abcdef.supabase.co
  SUPABASE_ANON_KEY   anon/public key from Supabase dashboard
  GENIUS_TOKEN        Genius API token
  HF_PREDICT_URL      your model space /predict URL (has default below)
"""

import os
import shutil
import tempfile
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Config — must match training EXACTLY ─────────────────────────────────────
SAMPLE_RATE  = 22050
N_MFCC       = 13
HOP_LENGTH   = 512
N_FFT        = 2048
N_CHROMA     = 12
CLIP_SECONDS = 30

AUDIO_COLS = (
    [f'mfcc_{i}'   for i in range(1, 14)] +
    [f'chroma_{i}' for i in range(1, 13)] +
    ['spectral_centroid', 'spectral_bandwidth', 'zcr', 'rms']
)  # 29 features, exact order

# ── Env vars ──────────────────────────────────────────────────────────────────
HF_PREDICT_URL = os.environ.get(
    "HF_PREDICT_URL",
    "https://mandeep347-song-valence-predictor.hf.space/predict"
)
GENIUS_TOKEN = os.environ.get("GENIUS_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
TABLE        = "songs"

# Path to Netscape-format cookies.txt — upload to HF Space repo as /app/cookies.txt
# Override via env var COOKIES_PATH if stored elsewhere
COOKIES_PATH = os.environ.get("COOKIES_PATH", "/app/cookies.txt")


# ── Supabase REST helpers ─────────────────────────────────────────────────────
def _sb_headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

def sb_get_song(song: str, artist: str, domain: int):
    """Return cached DB row dict or None."""
    if not SUPABASE_URL:
        return None
    params = {
        "song":   f"eq.{song.strip().lower()}",
        "artist": f"eq.{artist.strip().lower()}",
        "domain": f"eq.{domain}",
        "limit":  "1",
        "select": "*",
    }
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_sb_headers(), params=params, timeout=10
        )
        if r.status_code == 200:
            rows = r.json()
            return rows[0] if rows else None
    except Exception:
        pass
    return None

def sb_insert_song(entry: dict):
    """Upsert song row — ignore if (song, artist, domain) already exists."""
    if not SUPABASE_URL:
        return
    headers = _sb_headers()
    headers["Prefer"] = "resolution=ignore-duplicates,return=representation"
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=headers, json=entry, timeout=10
        )
    except Exception:
        pass

def sb_get_playlist(domain: Optional[int] = None):
    """Return all songs sorted by valence ascending."""
    if not SUPABASE_URL:
        return []
    params = {"order": "valence.asc", "select": "*"}
    if domain is not None:
        params["domain"] = f"eq.{domain}"
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_sb_headers(), params=params, timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Moodshift Pipeline Server",
    description="Song name → audio + lyrics → valence (with Supabase cache)",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    song:   str
    artist: str
    domain: int = 0   # 0=English/Spotify, 1=Hindi/Bollywood


class PredictResponse(BaseModel):
    song:           str
    artist:         str
    domain:         int
    valence:        float
    confidence:     str
    valence_zscore: float
    latency_ms:     int
    pipeline_ms:    int
    lyrics_chars:   int
    from_cache:     bool


# ── Pipeline steps ────────────────────────────────────────────────────────────
def download_audio(song: str, artist: str, out_dir: str) -> str:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed.")

    ydl_opts = {
        "format":         "bestaudio/best",
        "outtmpl":        os.path.join(out_dir, "%(title)s.%(ext)s"),
        "quiet":          True,
        "no_warnings":    True,
        "noplaylist":     True,
        "default_search": "ytsearch1",
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "mp3",
            "preferredquality": "192",
        }],
        # HF Spaces datacenter SSL & network hardening
        "nocheckcertificate":  True,
        "legacyserverconnect": True,
        "retries":             10,
        "fragment_retries":    10,
        "socket_timeout":      30,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }

    # Inject cookies if file exists
    if os.path.isfile(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH

    query = f"{artist} {song} official audio"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(f"ytsearch1:{query}", download=True)

    for fname in os.listdir(out_dir):
        if fname.endswith(".mp3"):
            return os.path.join(out_dir, fname)
    raise RuntimeError("Download succeeded but mp3 not found.")


def find_highlight(mp3_path: str):
    import librosa
    y, sr         = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
    rms           = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    onset_str     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else x

    score         = norm(rms) + norm(onset_str)
    fps           = SAMPLE_RATE / HOP_LENGTH
    window_frames = int(CLIP_SECONDS * fps)
    duration      = librosa.get_duration(y=y, sr=sr)

    if window_frames >= len(score):
        return 0.0, duration

    window_sum = np.convolve(score, np.ones(window_frames), mode="valid")
    best_frame = int(np.argmax(window_sum))
    start_sec  = best_frame / fps
    end_sec    = min(start_sec + CLIP_SECONDS, duration)
    start_sec  = max(0.0, end_sec - CLIP_SECONDS)
    return start_sec, end_sec


def clip_audio(mp3_path: str, out_dir: str) -> str:
    import librosa, soundfile as sf
    start_sec, _ = find_highlight(mp3_path)
    y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True,
                         offset=start_sec, duration=CLIP_SECONDS)
    clip_path = os.path.join(out_dir, "clip.wav")
    sf.write(clip_path, y, sr)
    return clip_path


def extract_features(audio_path: str) -> list:
    import librosa
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(y) < sr:
        raise RuntimeError("Clip too short (< 1 second).")

    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    for i in range(N_MFCC):
        features[f'mfcc_{i+1}'] = float(np.mean(mfcc[i]))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA,
                                          hop_length=HOP_LENGTH, n_fft=N_FFT)
    for i in range(N_CHROMA):
        features[f'chroma_{i+1}'] = float(np.mean(chroma[i]))

    features['spectral_centroid']  = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)))
    features['spectral_bandwidth'] = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)))
    features['zcr']  = float(np.mean(
        librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)))
    features['rms']  = float(np.mean(
        librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=N_FFT)))

    return [features[col] for col in AUDIO_COLS]


def fetch_lyrics(song: str, artist: str) -> str:
    try:
        import lyricsgenius
    except ImportError:
        raise RuntimeError("lyricsgenius not installed.")
    if not GENIUS_TOKEN:
        raise RuntimeError("GENIUS_TOKEN env var not set.")

    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        remove_section_headers=True,
    )
    genius.verbose = False

    result = genius.search_song(song, artist)
    if result is None:
        raise RuntimeError(f"Lyrics not found on Genius for '{song}' by '{artist}'.")

    lyrics = result.lyrics.strip()
    lines  = lyrics.split("\n")
    if lines and lines[0].lower().endswith("lyrics"):
        lyrics = "\n".join(lines[1:]).strip()
    return lyrics


def call_hf_predict(lyrics: str, audio_features: list, domain: int) -> dict:
    payload = {"lyrics": lyrics, "audio_features": audio_features, "domain": domain}
    r = requests.post(HF_PREDICT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


# ── API routes ────────────────────────────────────────────────────────────────
@app.post("/run", response_model=PredictResponse)
async def run_pipeline(req: PredictRequest):
    total_start = time.time()

    # 1. DB cache check — return instantly if found
    cached = sb_get_song(req.song, req.artist, req.domain)
    if cached:
        return PredictResponse(
            song=cached.get("song_display", cached["song"]),
            artist=cached.get("artist_display", cached["artist"]),
            domain=cached["domain"],
            valence=cached["valence"],
            confidence=cached["confidence"],
            valence_zscore=cached["valence_zscore"],
            latency_ms=int((time.time() - total_start) * 1000),
            pipeline_ms=0,
            lyrics_chars=cached.get("lyrics_chars", 0),
            from_cache=True,
        )

    # 2. Full pipeline
    tmp_dir = tempfile.mkdtemp(prefix="moodshift_")
    try:
        pipe_start = time.time()
        mp3_path       = download_audio(req.song, req.artist, tmp_dir)
        clip_path      = clip_audio(mp3_path, tmp_dir)
        audio_features = extract_features(clip_path)
        lyrics         = fetch_lyrics(req.song, req.artist)
        pipeline_ms    = int((time.time() - pipe_start) * 1000)

        hf_result = call_hf_predict(lyrics, audio_features, req.domain)
        total_ms  = int((time.time() - total_start) * 1000)

        # 3. Save to Supabase
        sb_insert_song({
            "song":           req.song.strip().lower(),
            "song_display":   req.song.strip(),
            "artist":         req.artist.strip().lower(),
            "artist_display": req.artist.strip(),
            "domain":         req.domain,
            "valence":        hf_result["valence"],
            "confidence":     hf_result["confidence"],
            "valence_zscore": hf_result.get("valence_zscore", 0.0),
            "pipeline_ms":    pipeline_ms,
            "total_ms":       total_ms,
            "lyrics_chars":   len(lyrics),
        })

        return PredictResponse(
            song=req.song,
            artist=req.artist,
            domain=req.domain,
            valence=hf_result["valence"],
            confidence=hf_result["confidence"],
            valence_zscore=hf_result.get("valence_zscore", 0.0),
            latency_ms=total_ms,
            pipeline_ms=pipeline_ms,
            lyrics_chars=len(lyrics),
            from_cache=False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/playlist")
def get_playlist(
    domain: Optional[int] = Query(default=None, description="0=English, 1=Hindi. Omit for all.")
):
    """All songs from DB sorted by valence ascending."""
    songs = sb_get_playlist(domain)
    for s in songs:
        s["song"]   = s.get("song_display")   or s["song"]
        s["artist"] = s.get("artist_display") or s["artist"]
    return {"songs": songs, "count": len(songs)}


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "supabase":  "connected" if (SUPABASE_URL and SUPABASE_KEY) else "not configured",
        "hf_model":  HF_PREDICT_URL,
        "cookies":   "found" if os.path.isfile(COOKIES_PATH) else "missing — YouTube will likely block downloads",
    }


@app.get("/")
def root():
    return {"message": "Moodshift v2. POST /run | GET /playlist | GET /health"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)