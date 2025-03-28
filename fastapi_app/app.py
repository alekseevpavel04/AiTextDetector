import os
import uvicorn
import logging
import re
import json
import docx
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Disable Python warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Global variables for model and tokenizer
model = None
tokenizer = None

logger.info(f"CUDA is available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# Create a context manager to handle application lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    cache_path = os.path.expanduser("~/.cache/huggingface")
    model_name = "SuperAnnotate/ai-detector"
    logger.info("==== Starting model loading ====")
    try:
        logger.info(f"Loading tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_path
        )
        logger.info(f"Tokenizer successfully loaded")
        logger.info(f"Loading model {model_name}...")
        model = RobertaClassifier.from_pretrained(
            model_name,
            cache_dir=cache_path
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        logger.info(f"✓ Model {model_name} successfully loaded")
        logger.info(f"✓ Device used: {next(model.parameters()).device}")
    except Exception as e:
        logger.error(f"✗ Critical error during model loading: {str(e)}")
        raise
    logger.info("==== Model loading completed ====")
    yield
    logger.info("Terminating the application, freeing resources...")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialize FastAPI application with lifecycle manager
app = FastAPI(
    title="AI Text Detector",
    description="API for detecting AI-generated text in documents",
    lifespan=lifespan
)


# Define the request model
class TextRequest(BaseModel):
    text: str


class AnalysisResult(BaseModel):
    sentence: str
    probability: float


class DocumentAnalysisResponse(BaseModel):
    results: List[AnalysisResult]
    filename: str


def split_into_sentences(text):
    """Split text into sentences using regex."""
    # This pattern handles most common sentence endings
    # It considers periods, exclamation marks, question marks followed by space or end of string as sentence boundaries
    # It also tries to avoid splitting on common abbreviations, decimals, etc.
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    return [sentence.strip() for sentence in sentences if sentence.strip()]


async def analyze_text(text: str) -> List[AnalysisResult]:
    """Analyze text and return AI generation probability for each sentence."""
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not yet loaded. Please try again later."
        )

    # Split the text into sentences
    sentences = split_into_sentences(text)

    results = []
    # Process each sentence
    for sentence in sentences:
        if len(sentence) < 10:  # Skip very short sentences
            continue

        try:
            # Preprocess text
            preprocessed_text = preprocessing_text(sentence)

            # Tokenize
            tokens = tokenizer.encode_plus(
                preprocessed_text,
                add_special_tokens=True,
                max_length=512,
                padding='longest',
                truncation=True,
                return_token_type_ids=True,
                return_tensors="pt"
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                tokens = {k: v.to("cuda") for k, v in tokens.items()}

            # Get prediction
            with torch.no_grad():
                _, logits = model(**tokens)
                proba = F.sigmoid(logits).squeeze(1).item()

            # Add result
            results.append(AnalysisResult(
                sentence=sentence,
                probability=round(proba, 4)  # Round to 4 decimal places
            ))

        except Exception as e:
            logger.error(f"Error analyzing sentence: {str(e)}")
            # Include the error sentence with a negative probability to indicate error
            results.append(AnalysisResult(
                sentence=f"ERROR: Could not analyze: {sentence[:50]}...",
                probability=-1.0
            ))

    return results


async def process_docx(file_content: bytes, filename: str) -> List[AnalysisResult]:
    """Process a DOCX file and analyze its content."""
    try:
        # Load the document from bytes
        doc = docx.Document(io.BytesIO(file_content))

        # Extract text from paragraphs
        full_text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])

        # Analyze the text
        return await analyze_text(full_text)
    except Exception as e:
        logger.error(f"Error processing DOCX file: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing DOCX file: {str(e)}"
        )


@app.post("/analyze/text", response_model=List[AnalysisResult])
async def analyze_text_endpoint(request: TextRequest):
    """Analyze text for AI-generated content probability."""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    results = await analyze_text(request.text)
    return results


@app.post("/analyze/document", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """Analyze a document (DOCX) for AI-generated content probability."""
    # Check file extension
    if not file.filename.lower().endswith('.docx'):
        raise HTTPException(
            status_code=400,
            detail="Only DOCX files are supported"
        )

    # Read file content
    file_content = await file.read()

    # Process the document
    results = await process_docx(file_content, file.filename)

    return DocumentAnalysisResponse(
        results=results,
        filename=file.filename
    )


# Service health check
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        logger.warning("Request to /health, but the model is not yet loaded!")
        return {"status": "loading", "message": "Model is still loading"}
    return {"status": "ok", "message": "Service is fully operational"}


# Start the server
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)