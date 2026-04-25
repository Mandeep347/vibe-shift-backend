FROM python:3.11-slim

# ffmpeg required by yt-dlp for mp3 conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_server.py .

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "pipeline_server.py"]
