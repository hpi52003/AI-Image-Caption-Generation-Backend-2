from fastapi import FastAPI, File, UploadFile, Query  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse, JSONResponse  # type: ignore
from googletrans import Translator  # type: ignore
from gtts import gTTS  # type: ignore
from typing import Optional
import requests  # type: ignore
import os

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint to prevent 404 on base URL
@app.get("/")
def read_root():
    return {"message": "Image Captioning API is live"}

# Hugging Face Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Store token securely in environment variables

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# Translator instance
translator = Translator()

# Globals to hold latest caption
translated_caption_global = ""
lang_global = "en"

@app.post("/caption/")
async def generate_caption(file: UploadFile = File(...), lang: Optional[str] = Query("en")):
    global translated_caption_global, lang_global

    try:
        image_bytes = await file.read()
        response = requests.post(
            HF_API_URL,
            headers=headers,
            files={"file": (file.filename, image_bytes)}
        )

        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Model API error or quota exceeded"})

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            caption = result[0]["generated_text"]
        else:
            return JSONResponse(status_code=500, content={"error": "Invalid model response"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Caption generation error: {str(e)}"})

    # Translate if necessary
    translated_caption = caption
    if lang != "en":
        try:
            translated_caption = translator.translate(caption, dest=lang).text
        except Exception as e:
            translated_caption = f"[Translation error: {str(e)}]"

    translated_caption_global = translated_caption
    lang_global = lang

    return {
        "original_caption": caption,
        "translated_caption": translated_caption
    }

@app.get("/audio")
async def get_audio():
    global translated_caption_global, lang_global

    if not translated_caption_global:
        return JSONResponse(status_code=400, content={"error": "No caption available for audio generation"})

    try:
        path = "caption.mp3"
        gTTS(text=translated_caption_global, lang=lang_global).save(path)
        return FileResponse(path, media_type="audio/mpeg", filename="caption.mp3")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Text-to-speech error: {str(e)}"})


       
        
