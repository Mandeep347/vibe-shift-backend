"""
pipeline_server.py — Moodshift Pipeline Server (Production)
============================================================
Based on user's local version. Added:
  - Supabase (Postgres) cache via REST — DB hit returns instantly
  - SSL hardening for HF Spaces datacenter (UNEXPECTED_EOF fix)
  - /playlist endpoint — all songs sorted by valence asc
  - /health shows supabase + cookies status

All original logic preserved:
  - format_text / utils import
  - default_audio fallback for download
  - fallback.txt for lyrics
  - cl if cl<1 valence override

Environment variables (HF Space secrets):
  SUPABASE_URL        e.g. https://abcdef.supabase.co
  SUPABASE_ANON_KEY   anon/public key from Supabase dashboard
  GENIUS_TOKEN        Genius API token
  default_audio       fallback audio URL (same as before)
  COOKIES_PATH        path to cookies.txt (default /app/cookies.txt)
"""

import os
import re
import ssl
import shutil
import tempfile
import time
import warnings
import json

warnings.filterwarnings("ignore")

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Global SSL patch — fixes UNEXPECTED_EOF_WHILE_READING on HF Spaces ───────
# Forces TLS 1.2 + disables cert checks globally (affects yt-dlp internals too)
_orig_ssl_context = ssl.create_default_context

def _patched_ssl_context(*args, **kwargs):
    ctx = _orig_ssl_context(*args, **kwargs)
    ctx.check_hostname  = False
    ctx.verify_mode     = ssl.CERT_NONE
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
    return ctx

ssl.create_default_context = _patched_ssl_context

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from utils import format_text

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

HF_API_URL   = "https://mandeep347-song-valence-predictor.hf.space/predict"
GENIUS_TOKEN = os.environ.get(
    "GENIUS_TOKEN",
    "oKoy_hycWMA4d6VBkupZ3d09O1TM_KJg0dgPxmQWAA2vUFm4o2Srt0Wajgl4PMxl"
)

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
TABLE        = "songs"

# Cookies for yt-dlp
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
    """Return cached DB row or None."""
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
            headers=_sb_headers(), params=params, timeout=10, verify=False,
        )
        if r.status_code == 200:
            rows = r.json()
            return rows[0] if rows else None
    except Exception:
        pass
    return None

def sb_insert_song(entry: dict):
    """Insert song — ignore if (song, artist, domain) already exists."""
    if not SUPABASE_URL:
        return
    headers = _sb_headers()
    headers["Prefer"] = "resolution=ignore-duplicates,return=representation"
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=headers, json=entry, timeout=10, verify=False,
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
            headers=_sb_headers(), params=params, timeout=10, verify=False,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def sb_get_better_songs(valence: float, domain: Optional[int] = None, limit: int = 10):
    if not SUPABASE_URL:
        return []

    params = {
        "valence": f"gt.{valence}",   # STRICTLY greater than
        "order": "valence.asc",       # closest higher first
        "limit": str(limit),
        "select": "*",
    }

    if domain is not None:
        params["domain"] = f"eq.{domain}"

    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_sb_headers(),
            params=params,
            timeout=10,
            verify=False,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    return []


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Moodshift Pipeline Server",
    description="Middleware: song name → audio features + lyrics → HF /predict",
    version="2.0.0"
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
    domain: int = 0   # 0 = English, 1 = Hindi


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
    from_hf:        bool
    from_cache:     bool
    recommendation: list = []


# ── Step 1: Download audio ────────────────────────────────────────────────────
def download_audio(song: str, artist: str, out_dir: str) -> str:
    default_fallback = os.getenv("default_audio")

    os.makedirs(out_dir, exist_ok=True)

    def download_fallback():
        fallback_path = os.path.join(out_dir, "fallback.mp3")
        try:
            r = requests.get(default_fallback, timeout=10, verify=False)
            r.raise_for_status()
            with open(fallback_path, "wb") as f:
                f.write(r.content)
            return fallback_path
        except Exception as e:
            print(f"[fallback download ERROR]: {e}")
            return None

    try:
        import yt_dlp
    except ImportError:
        return download_fallback()

    query    = f"{artist} {song} official audio"
    out_tmpl = os.path.join(out_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format":         "bestaudio/best",
        "outtmpl":        out_tmpl,
        "quiet":          True,
        "no_warnings":    True,
        "noplaylist":     True,
        "default_search": "ytsearch1",
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "mp3",
            "preferredquality": "192",
        }],
        # ── HF Spaces SSL & network hardening ─────────────────────────
        "nocheckcertificate":  True,
        "legacyserverconnect": True,
        "retries":             1,
        "fragment_retries":    1,
        "socket_timeout":      30,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }

    if os.path.isfile(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH

    # Retry loop — SSL EOF errors are often transient
    last_err = None
    for attempt in range(1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info     = ydl.extract_info(f"ytsearch1:{query}", download=True)
                entry    = info["entries"][0] if "entries" in info else info
                filename = ydl.prepare_filename(entry)
                mp3_file = os.path.splitext(filename)[0] + ".mp3"

            if os.path.exists(mp3_file):
                return mp3_file

            for fname in os.listdir(out_dir):
                if fname.endswith(".mp3"):
                    return os.path.join(out_dir, fname)

            return download_fallback()

        except Exception as e:
            last_err = e
            print(f"[download_audio attempt {attempt+1} ERROR]: {e}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    print(f"[download_audio] all retries failed: {last_err}")
    return download_fallback()

def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


# ── Step 2: Find most energetic 30s window ───────────────────────────────────
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
    import librosa
    import soundfile as sf

    start_sec, _ = find_highlight(mp3_path)
    y, sr = librosa.load(
        mp3_path, sr=SAMPLE_RATE, mono=True,
        offset=start_sec, duration=CLIP_SECONDS
    )
    clip_path = os.path.join(out_dir, "clip.wav")
    sf.write(clip_path, y, sr)
    return clip_path


# ── Step 3: Extract 29 Librosa features ──────────────────────────────────────
def extract_features(audio_path: str) -> list:
    import librosa

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(y) < sr:
        raise RuntimeError("Clip too short (< 1 second) — cannot extract features.")

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
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    ))
    features['spectral_bandwidth'] = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    ))
    features['zcr'] = float(np.mean(
        librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    ))
    features['rms'] = float(np.mean(
        librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=N_FFT)
    ))

    return [features[col] for col in AUDIO_COLS]


# ── Step 4: Fetch lyrics from Genius ─────────────────────────────────────────
def fetch_lyrics(song: str, artist: str) -> str:

    with open("fallback.txt", "r", encoding="utf-8") as f:
        DEFAULT_LYRICS = f.read()

    try:
        import lyricsgenius
    except ImportError:
        return DEFAULT_LYRICS

    if not GENIUS_TOKEN:
        return DEFAULT_LYRICS

    try:
        genius = lyricsgenius.Genius(
            GENIUS_TOKEN,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"],
            remove_section_headers=True,
        )
        genius.verbose = False

        result = genius.search_song(song, artist)
        if result is None or not result.lyrics:
            return DEFAULT_LYRICS

        lyrics = result.lyrics.strip()
        lines  = lyrics.split("\n")
        if lines and lines[0].lower().endswith("lyrics"):
            lyrics = "\n".join(lines[1:]).strip()

        return lyrics if lyrics else DEFAULT_LYRICS

    except Exception as e:
        print(f"[fetch_lyrics ERROR]: {e}")
        return DEFAULT_LYRICS


# ── Step 5: Call HF /predict ──────────────────────────────────────────────────
def call_hf_predict(lyrics: str, audio_features: list, domain: int) -> dict:
    payload = {
        "lyrics":         lyrics,
        "audio_features": audio_features,
        "domain":         domain,
    }
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                HF_API_URL, json=payload, timeout=180, verify=False,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.SSLError as e:
            last_err = e
            time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"HF predict failed after 3 attempts: {last_err}")


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/run", response_model=PredictResponse)
async def run_pipeline(req: PredictRequest):
    total_start = time.time()

    # ── 1. Supabase cache check — return instantly if found ───────────────────
    cached = sb_get_song(req.song, req.artist, req.domain)
    if cached:
        cached_valence = cached["valence"]
        better_songs   = sb_get_better_songs(cached_valence, req.domain)
        return PredictResponse(
            song=cached.get("song_display", cached["song"]),
            artist=cached.get("artist_display", cached["artist"]),
            domain=cached["domain"],
            valence=cached_valence,
            confidence=cached["confidence"],
            valence_zscore=cached["valence_zscore"],
            latency_ms=int((time.time() - total_start) * 1000),
            pipeline_ms=0,
            lyrics_chars=cached.get("lyrics_chars", 0),
            from_hf=True,
            from_cache=True,
            recommendation=better_songs
        )

    # ── 2. Full pipeline ──────────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="moodshift_")

    try:
        pipe_start = time.time()

        # Original logic preserved exactly
        cl       = format_text(req.song, req.artist)
        mp3_path = download_audio(req.song, req.artist, tmp_dir)

        clip_path      = clip_audio(mp3_path, tmp_dir)
        audio_features = extract_features(clip_path)
        lyrics         = fetch_lyrics(req.song, req.artist)

        pipeline_ms = int((time.time() - pipe_start) * 1000)

        hf_result = call_hf_predict(lyrics, audio_features, req.domain)
        total_ms  = int((time.time() - total_start) * 1000)

        # Original valence override logic preserved exactly
        final_valence = cl if cl < 1 else hf_result["valence"]

        # Get only better valence songs to uplift mood
        better_songs = sb_get_better_songs(final_valence, req.domain)

        # ── 3. Save to Supabase ───────────────────────────────────────────────
        sb_insert_song({
            "song":           req.song.strip().lower(),
            "song_display":   req.song.strip(),
            "artist":         req.artist.strip().lower(),
            "artist_display": req.artist.strip(),
            "domain":         req.domain,
            "valence":        final_valence,
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
            valence=final_valence,
            confidence=hf_result["confidence"],
            valence_zscore=hf_result.get("valence_zscore", 0.0),
            latency_ms=total_ms,
            pipeline_ms=pipeline_ms,
            lyrics_chars=len(lyrics),
            from_hf=True,
            from_cache=False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Playlist endpoint ─────────────────────────────────────────────────────────
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
        "status":      "ok",
        "hf_endpoint": HF_API_URL,
        "supabase":    "connected" if (SUPABASE_URL and SUPABASE_KEY) else "not configured",
        "cookies":     "found" if os.path.isfile(COOKIES_PATH) else "missing",
    }


@app.get("/")
def root():
    return {"message": "Moodshift pipeline server v2. POST /run to predict."}


if __name__ == "__main__":
    print("=" * 56)
    print("  Moodshift Pipeline Server v2")
    print("  http://localhost:7860")
    print("  Docs: http://localhost:7860/docs")
    print("=" * 56)
    uvicorn.run(app, host="0.0.0.0", port=7860)