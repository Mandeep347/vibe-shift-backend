FROM python:3.11-slim

# ffmpeg required by yt-dlp for mp3 conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_server.py .
COPY utils.py .
COPY fallback.txt .

# cookies.txt — Netscape format, exported from your browser.
# Export instructions: see README.md → Step 0.
# If file doesn't exist the COPY will fail — create an empty placeholder:
#   touch cookies.txt
# then replace with real cookies before pushing to HF.
COPY cookies.txt /app/cookies.txt

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "pipeline_server.py"]