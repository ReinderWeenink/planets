import json
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from slanggen import models
from tokenizers import Tokenizer
try:
    # Try importing as a package (works for pytest locally)
    from backend.utils import sample_n
except ModuleNotFoundError:
    # Fallback to sibling import (works for Docker flattened structure)
    from utils import sample_n

#logger.add("logs/app.log", rotation="5 MB")

FRONTEND_FOLDER = Path("static").resolve()
ARTEFACTS = Path("artefacts").resolve()
model_state = {}

def load_model():
    """
    Loads the model and tokenizer from the artefacts directory.
    Includes error handling for missing files and invalid configurations.
    """
    logger.info(f"Loading model and tokenizer from {ARTEFACTS}")
    
    tokenizer_path = ARTEFACTS / "tokenizer.json"
    config_path = ARTEFACTS / "config.json"
    model_path = ARTEFACTS / "model.pth"

    # 1. Load Tokenizer
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # 2. Load Configuration
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    try:
        with config_path.open("r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

    # 3. Load Model
    if "model" not in config:
        raise ValueError("Config file missing 'model' key")

    try:
        model = models.SlangRNN(config["model"])
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model architecture: {e}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    try:
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'), weights_only=False))
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    logger.success("Model and tokenizer loaded successfully")
    return model, tokenizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ARTEFACTS

    if not FRONTEND_FOLDER.exists():
        raise FileNotFoundError(f"Cant find the frontend folder at {FRONTEND_FOLDER}")
    
    if not ARTEFACTS.exists():
        logger.warning(f"Couldnt find artefacts at {ARTEFACTS}, trying parent")
        ARTEFACTS = Path("../artefacts").resolve()
        if not ARTEFACTS.exists():
            raise FileNotFoundError(f"Cant find the artefacts folder at {ARTEFACTS}")

    try:
        model, tokenizer = load_model()
        model_state["model"] = model
        model_state["tokenizer"] = tokenizer
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # In production, we might want to fail hard, but for tests we let it pass
        # so we can test the 'Model not loaded' endpoints.
        # re-raising here would crash the test collection if artefacts are missing.
        pass 

    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown...")
    model_state.clear()
    logger.success("Application shutdown complete")


app = FastAPI(lifespan=lifespan)

# Serve static files (check existence to be safe during tests)
if FRONTEND_FOLDER.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_FOLDER)), name="static")

starred_words = []


def new_words(n: int, temperature: float):
    # Fetch from state
    model = model_state.get("model")
    tokenizer = model_state.get("tokenizer")
    if not model or not tokenizer:
        raise RuntimeError("Model or Tokenizer not initialized")
        
    output_words = sample_n(
        n=n,
        model=model,
        tokenizer=tokenizer,
        max_length=20,
        temperature=temperature,
    )
    return output_words


class Word(BaseModel):
    word: str


@app.get("/generate")
async def generate_words(
    num_words: int = Query(default=10, ge=1, description="Number of words to generate (minimum 1)"),
    temperature: float = Query(default=1.0, ge=0, le=10, description="Temperature for sampling (0-10)")
):
    # Check if model is loaded
    if "model" not in model_state or "tokenizer" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        words = new_words(num_words, temperature)
        return words
    except Exception as e:
        logger.exception(f"Error generating words: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/starred")
async def get_starred_words():
    return starred_words


@app.post("/starred")
async def add_starred_word(word: Word):
    if word.word not in starred_words:
        starred_words.append(word.word)
    return starred_words


@app.post("/unstarred")
async def remove_starred_word(word: Word):
    if word.word in starred_words:
        starred_words.remove(word.word)
    return starred_words


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def read_index():
    logger.info("serving index.html")
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # It is crucial to use 0.0.0.0 for Docker
    uvicorn.run(app, host="0.0.0.0", port=80)