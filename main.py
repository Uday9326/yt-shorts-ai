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

@app.post("/generate")
async def gen(req: Request):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "API_KEY missing")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    yt = pytube.YouTube(req.youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    stream.download("video.mp4")

    segments, _ = whisper.transcribe("video.mp4")
    transcript = " ".join(seg.text for seg in segments)
    prompt = f"Give me start,end seconds for a {req.length}s highlight"
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt+"\n"+transcript}])
    start, end = map(int, resp.choices[0].message.content.strip().split(","))
    subprocess.run([
        "ffmpeg","-y","-i","video.mp4","-ss",str(start),"-t",str(end-start),
        "-vf","scale=1080:1920,setsar=1","short.mp4"
    ], check=True)
    return {"download": "/download"}

@app.get("/download")
async def download():
    if not os.path.exists("short.mp4"):
        raise HTTPException(404,"Not ready")
    from fastapi.responses import FileResponse
    return FileResponse("short.mp4", media_type="video/mp4", filename="short.mp4")