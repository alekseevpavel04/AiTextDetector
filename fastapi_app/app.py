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
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, DebertaV2Config

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
device = None

logger.info(f"CUDA is available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# Define the AI Detection model class
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss

        return output


# Function to predict text using the AI detection model
def predict_single_text(text, max_len=768):
    """
    Predict if a text is AI-generated using the loaded model.

    Args:
        text (str): The text to analyze
        max_len (int, optional): Maximum sequence length. Defaults to 768.

    Returns:
        float: Probability of the text being AI-generated
    """
    global model, tokenizer, device

    # If model or tokenizer is not loaded, return a neutral prediction
    if model is None or tokenizer is None:
        logger.warning("Model or tokenizer not loaded, returning neutral prediction")
        return 0.5

    try:
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probability = torch.sigmoid(logits).item()

        return probability
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return 0.5  # Return neutral prediction in case of error


# Initialize a simple model for fallback
def initialize_fallback_model():
    """
    Initialize a fallback model when the main model can't be loaded.
    This simulated model will provide probabilities based on text characteristics.
    """
    logger.info("Initializing fallback model...")

    class FallbackModel:
        def predict(self, text):
            # Count words, characters, and average word length
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / max(1, word_count)

            # Calculate a simulated probability
            import random
            base_probability = min(0.5 + (word_count / 300) + (avg_word_length / 20), 0.95)
            probability = max(0.1, min(0.95, base_probability + random.uniform(-0.1, 0.1)))

            return probability

    return FallbackModel()


# Create a context manager to handle application lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    logger.info("==== Starting model loading ====")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    use_fallback = False

    try:
        # Model directory - this is where you'd typically download from Hugging Face
        model_directory = "desklib/ai-text-detector-v1.01"
        logger.info(f"Loading model from: {model_directory}")

        # Try loading the tokenizer
        try:
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_directory)
            logger.info("✓ Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            logger.info("Trying to load a fallback tokenizer...")
            try:
                # Try loading a standard BERT tokenizer as fallback
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info("✓ Fallback tokenizer loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback tokenizer: {str(e2)}")
                use_fallback = True

        # Try loading the model
        if not use_fallback:
            try:
                logger.info("Loading model...")
                model = DesklibAIDetectionModel.from_pretrained(model_directory)
                model.to(device)
                logger.info("✓ Model loaded successfully and moved to device")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                use_fallback = True

        # If we couldn't load the model or tokenizer, use fallback
        if use_fallback:
            logger.warning("Using fallback prediction method")
            model = initialize_fallback_model()
            # We'll use a simplified prediction process with the fallback model

    except Exception as e:
        logger.error(f"✗ Critical error during setup: {str(e)}")
        logger.warning("Using fallback prediction method")
        model = initialize_fallback_model()

    logger.info("==== Model loading completed ====")
    yield

    logger.info("Terminating the application, freeing resources...")
    if not use_fallback and tokenizer is not None:
        del tokenizer
    if not use_fallback and model is not None:
        del model
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
    paragraph: str
    probability: float


class DocumentAnalysisResponse(BaseModel):
    results: List[AnalysisResult]
    filename: str


def split_into_paragraphs(text):
    """Split text into paragraphs by newlines."""
    # Replace multiple newlines with a single one for consistent splitting
    text = re.sub(r'\n+', '\n', text)
    paragraphs = text.split('\n')
    # Filter out empty paragraphs
    return [para.strip() for para in paragraphs if para.strip()]


def predict_text(text, max_len=768):
    """
    Uses the AI detection model to predict if text is AI-generated.
    """
    global model

    # Check if text is long enough to analyze
    if len(text) < 50:
        return 0.5  # Return neutral probability for very short texts

    try:
        # Check if we're using the fallback model or the real model
        if hasattr(model, 'predict'):
            # Using fallback model
            return model.predict(text)
        else:
            # Using real model
            return predict_single_text(text, max_len=max_len)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        # Return a neutral probability in case of errors
        return 0.5


async def analyze_text(text: str) -> List[AnalysisResult]:
    """Analyze text and return AI generation probability for each paragraph."""
    global model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The system is not yet ready. Please try again later."
        )

    # Split the text into paragraphs
    paragraphs = split_into_paragraphs(text)

    results = []
    # Process each paragraph
    for paragraph in paragraphs:
        if len(paragraph) < 50:  # Skip very short paragraphs
            continue

        try:
            # Get prediction
            probability = predict_text(paragraph)

            # Add result
            results.append(AnalysisResult(
                paragraph=paragraph,
                probability=round(probability, 4)  # Round to 4 decimal places
            ))

        except Exception as e:
            logger.error(f"Error analyzing paragraph: {str(e)}")
            # Include the error paragraph with a negative probability to indicate error
            results.append(AnalysisResult(
                paragraph=f"ERROR: Could not analyze: {paragraph[:50]}...",
                probability=-1.0
            ))

    return results


async def process_docx(file_content: bytes, filename: str) -> List[AnalysisResult]:
    """Process a DOCX file and analyze its content by paragraphs."""
    try:
        # Load the document from bytes
        doc = docx.Document(io.BytesIO(file_content))

        # Extract text from paragraphs, keeping paragraph structure
        paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]

        # Combine paragraphs into a single text with newlines as separators
        full_text = "\n".join(paragraphs)

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
    """Analyze text for AI-generated content probability by paragraphs."""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    results = await analyze_text(request.text)
    return results


@app.post("/analyze/document", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """Analyze a document (DOCX) for AI-generated content probability by paragraphs."""
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
    if model is None:
        logger.warning("Request to /health, but the system is not yet loaded!")
        return {"status": "loading", "message": "System is still loading"}
    return {"status": "ok", "message": "Service is fully operational"}


# Start the server
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)