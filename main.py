from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pytube, subprocess, os
from faster_whisper import WhisperModel
import openai

app = FastAPI()
whisper = WhisperModel("small")

class Request(BaseModel):
    youtube_url: str
    length: int = 30
    count: int = 3

@app.post("/generate")
async def generate(req: Request):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "Missing API Key")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    yt = pytube.YouTube(req.youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    stream.download("video.mp4")

    segments, _ = whisper.transcribe("video.mp4")
    transcript = " ".join(seg.text for seg in segments)

    prompt = f"Give me {req.count} highlight moments of {req.length} seconds each. Return start and end times, comma separated, one per line."
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt + "\n" + transcript}]
    )

    lines = res.choices[0].message.content.strip().splitlines()
    for i, line in enumerate(lines):
        start, end = map(int, line.split(","))
        subprocess.run([
            "ffmpeg", "-y", "-i", "video.mp4", "-ss", str(start), "-t", str(end - start),
            "-vf", "scale=1080:1920,setsar=1", f"short{i+1}.mp4"
        ], check=True)

    return {
        "message": f"{len(lines)} shorts generated",
        "downloads": [f"/download/{i+1}" for i in range(len(lines))]
    }

@app.get("/download/{num}")
async def download(num: int):
    filename = f"short{num}.mp4"
    if not os.path.exists(filename):
        raise HTTPException(404, "Short not found")
    from fastapi.responses import FileResponse
    return FileResponse(filename, media_type="video/mp4", filename=filename)
