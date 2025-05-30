from fastapi import FastAPI, File, UploadFile, Query # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import FileResponse, JSONResponse # type: ignore
from PIL import Image, UnidentifiedImageError # type: ignore
import io
import torch # type: ignore
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator # type: ignore
from gtts import gTTS # type: ignore
from typing import Optional
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

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Translator
translator = Translator()

# Global caption state
translated_caption_global = ""
lang_global = "en"

@app.post("/caption/")
async def generate_caption(file: UploadFile = File(...), lang: Optional[str] = Query("en")):
    global translated_caption_global, lang_global

    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "Unsupported image format"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error loading image: {str(e)}"})

    try:
        # Generate caption
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        print("Generated caption:", caption)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Caption generation error: {str(e)}"})

    # Translate caption if needed
    translated_caption = caption
    if lang != "en":
        try:
            translation = translator.translate(caption, dest=lang)
            translated_caption = translation.text
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
        return JSONResponse(status_code=400, content={"error": "No caption available to generate audio."})

    try:
        audio_path = "caption.mp3"
        tts = gTTS(text=translated_caption_global, lang=lang_global)
        tts.save(audio_path)
        return FileResponse(audio_path, media_type="audio/mpeg", filename="caption.mp3")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Text-to-speech error: {str(e)}"})
