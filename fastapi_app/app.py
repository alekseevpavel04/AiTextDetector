import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
import re
import json
from fastapi.responses import StreamingResponse
import asyncio

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
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info("==== Starting model loading ====")
    try:
        logger.info(f"Loading tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_path
        )
        logger.info(f"Tokenizer successfully loaded")
        logger.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_path
        )
        logger.info(f"✓ Model {model_name} successfully loaded")
        logger.info(f"✓ Device used: {model.device}")
    except Exception as e:
        logger.error(f"✗ Critical error during model loading: {str(e)}")
        raise
    logger.info("==== Model loading completed ====")
    yield
    logger.info("Terminating the application, freeing resources...")
    del model
    del tokenizer
    torch.cuda.empty_cache()

# Initialize FastAPI application with lifecycle manager
app = FastAPI(
    title="Culinary Assistant",
    description="API for recipe generation and comparison",
    lifespan=lifespan
)

# Define the request model
class RecipeRequest(BaseModel):
    query: str
    max_length: Optional[int] = 1024
    temperature: Optional[float] = 0.7

# Headers for web requests to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Function to use LLM to optimize search query
async def optimize_search_query(query: str) -> str:
    global model, tokenizer
    logger.info(f"Using LLM to optimize query: {query}")
    system_message = {
        "role": "system",
        "content": (
            "You are a culinary search assistant. Your task is to extract the main dish name from user cooking queries and convert it into an optimal search term. "
            "Follow these rules STRICTLY:\n"
            "1. Remove unnecessary words like 'how to cook', 'recipe for', 'best way to prepare', 'homemade', etc.\n"
            "2. Fix obvious spelling mistakes ONLY if they relate to dishes, ingredients, or culinary terms (e.g., 'spagheti' → 'spaghetti', 'burgir' → 'burger').\n"
            "   - Do NOT correct words that are not related to food (e.g., 'how to' → remains 'how to').\n"
            "   - If the misspelled word is ambiguous and could refer to a non-culinary term, assume it is a dish or ingredient (e.g., 'burgir' → 'burger').\n"
            "3. Extract ONLY the dish name (e.g., 'Recipe for delicious chicken parmesan' → 'chicken parmesan').\n"
            "4. If the dish name includes a specific cuisine, region, or style (e.g., 'japanese ramen', 'texas bbq ribs'), preserve those words as part of the dish name.\n"
            "5. Do not add any extra words that were not in the original query.\n"
            "6. No punctuation, capitalization, or extra formatting in your response.\n"
            "7. Your response must be extremely short: just 1-3 words.\n\n"
            "Examples:\n"
            "- 'How to cook lasagna at home?' → 'lasagna'\n"
            "- 'Best way to prepare beef stroganoff' → 'beef stroganoff'\n"
            "- 'Recipe for easy chicken tikka masala' → 'chicken tikka masala'\n"
            "- 'Home made spagheti carbonara' → 'spaghetti carbonara'\n"
            "- 'How do I make a classic apple pie?' → 'apple pie'\n"
            "- 'I want chiken burgir' → 'chicken burger'\n"
            "- 'How to make japanese ramen?' → 'japanese ramen'\n"
            "- 'Recipe for texas bbq ribs' → 'texas bbq ribs'\n"
            "- 'How to cook vegan pad thai' → 'vegan pad thai'\n"
            "- 'Best way to prepare new york cheesecake' → 'new york cheesecake'\n"
            "- 'I need a recipe for burgir' → 'burger'\n"
            "- 'How to make spageti bolognese' → 'spaghetti bolognese'\n"
            "- 'Easy sushi rolls for beginners' → 'sushi roll'\n"
        )
    }
    user_message = {
        "role": "user",
        "content": f"Convert this cooking query to optimal search terms: {query}"
    }
    messages = [system_message, user_message]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=15,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    optimized_query = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().strip('"\'').strip()
    common_additions = ["preparation", "method", "recipe", "cooking", "instructions", "how to", "how to make"]
    for addition in common_additions:
        optimized_query = re.sub(r'\s+' + addition + r'\s*$', '', optimized_query, flags=re.IGNORECASE)
        optimized_query = re.sub(r'^' + addition + r'\s+', '', optimized_query, flags=re.IGNORECASE)
    optimized_query = re.sub(r'\s+', ' ', optimized_query).strip()
    logger.info(f"LLM optimized query: '{query}' -> '{optimized_query}'")
    return optimized_query


# Function to search AllRecipes and get multiple recipes
async def search_multiple_recipes(query: str, max_recipes: int = 3) -> List[Dict]:
    try:
        cleaned_query = await optimize_search_query(query)
        logger.info(f"Searching AllRecipes for: {cleaned_query}, max recipes: {max_recipes}")
        search_url = f"https://www.allrecipes.com/search?q={cleaned_query.replace(' ', '+')}"
        search_response = requests.get(search_url, headers=HEADERS, timeout=15)
        search_response.raise_for_status()
        search_soup = BeautifulSoup(search_response.text, 'html.parser')

        recipe_links = []
        recipe_titles = []

        for card in search_soup.find_all(['a'], href=True):
            href = card['href']
            if '/recipe/' in href and not href.endswith('/reviews/') and not href.endswith('/photos/'):
                if href not in recipe_links:
                    title_element = card.find('h3', class_='card__title')
                    if title_element:
                        title = title_element.text.strip()
                    else:
                        title_element = card.find(['h1', 'h2', 'h3', 'h4'])
                        if title_element:
                            title = title_element.text.strip()
                        else:
                            title = None  # Set title to None if not found
                    recipe_links.append(href)
                    recipe_titles.append(title)
                    if len(recipe_links) >= max_recipes:
                        break

        if not recipe_links:
            logger.info(f"No recipes found for query: {cleaned_query}")
            return []

        logger.info(f"Found {len(recipe_links)} recipe links")

        recipes = []
        for i, (link, title) in enumerate(zip(recipe_links, recipe_titles)):
            if title is None:
                recipe_details = await extract_recipe_details(link)
                title = recipe_details["title"]
            recipes.append({
                "title": title,
                "link": link,
                "index": i + 1
            })

        return recipes

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return []

# Optimized function to extract detailed recipe information
async def extract_recipe_details(recipe_link: str) -> Dict:
    try:
        logger.info(f"Fetching recipe from: {recipe_link}")
        recipe_response = requests.get(recipe_link, headers=HEADERS, timeout=10)
        recipe_response.raise_for_status()
        recipe_soup = BeautifulSoup(recipe_response.text, 'html.parser')
        title_element = recipe_soup.find('h1', class_='article-heading')
        title = title_element.text.strip() if title_element else "Unknown Recipe"
        return {
            "title": title,
            "source_url": recipe_link
        }
    except Exception as e:
        logger.error(f"Error fetching recipe details from {recipe_link}: {str(e)}")
        return {
            "title": "Error fetching recipe",
            "source_url": recipe_link
        }

# Function to generate recipe analysis using the model
async def generate_recipe_analysis(recipe: Dict, query: str, max_length: int = 256,
                                   temperature: float = 0.7) -> str:
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not yet loaded. Please try again later."
        )
    logger.info(f"Generating recipe analysis for: {recipe['title']}")
    system_message = {
        "role": "system",
        "content": (
            "You are a professional chef providing very concise recipe insights. Examine the entire recipe "
            "carefully and provide a brief comment (just 1-2 sentences) that highlights one or two notable aspects "
            "of this recipe. Focus ONLY on unique or distinctive features, such as:\n"
            "- Unusgit ual or rare ingredients (e.g., smoked paprika, bonito flakes, saffron) and their role in the dish.\n"
            "- Specific cooking techniques (e.g., sous-vide, fermentation, pressure cooking) and their impact (e.g., '48-hour fermentation enhances umami and complexity').\n"
            "- Cultural or regional significance (e.g., 'Traditional Mexican mole with over 20 ingredients, including chocolate and chili').\n"
            "- Notable deviations from traditional versions (e.g., 'uses quinoa instead of rice for a nutty texture and higher protein content').\n"
            "Avoid generic praise (e.g., 'delicious', 'tasty') or vague statements. Be specific, factual, and highlight what makes this recipe stand out.\n"
            "If applicable, mention:\n"
            "- Time-saving techniques (e.g., 'pressure cooking reduces broth preparation time to 30 minutes').\n"
            "- Unique flavor combinations (e.g., 'coconut milk and lemongrass create a creamy, aromatic broth').\n"
            "- Ingredient substitutions and their impact (e.g., 'pork belly replaces beef cutlet for a richer, fattier flavor').\n"
            "Your response must not exceed 2 sentences.\n\n"
            "Examples:\n"
            "- 'Uses smoked paprika and saffron for a unique, smoky-sweet flavor profile.'\n"
            "- 'Features a 48-hour fermentation process, resulting in a tangy and complex sourdough with enhanced umami.'\n"
            "- 'A modern twist on classic ramen, incorporating coconut milk and lemongrass for a creamy, aromatic broth.'\n"
            "- 'Traditional Mexican mole with over 20 ingredients, including chocolate and chili, for a rich, layered flavor.'\n"
            "- 'Deviates from the classic recipe by using quinoa instead of rice, adding a nutty texture and higher protein content.'\n"
            "- 'Pressure cooking reduces broth preparation time to 30 minutes while intensifying umami flavors from kombu and bonito.'\n"
            "- 'Substitutes pork belly for beef cutlet, creating a richer, fattier tonkatsu with a crispy texture.'\n"
        )
    }
    try:
        logger.info(f"Fetching full page content from: {recipe['source_url']}")
        full_response = requests.get(recipe['source_url'], headers=HEADERS, timeout=15)
        full_response.raise_for_status()
        full_soup = BeautifulSoup(full_response.text, 'html.parser')
        for script in full_soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
            script.extract()
        full_text = full_soup.get_text(separator=' ', strip=True)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        if len(full_text) > 15000:
            full_text = full_text[:15000] + "..."
        logger.info(f"Successfully extracted full page context ({len(full_text)} characters)")
    except Exception as e:
        logger.warning(f"Failed to get full page content: {str(e)}. Using structured data only.")
        full_text = f"Title: {recipe['title']}"
    user_message = {
        "role": "user",
        "content": (
            f"Here's a recipe for {query} that I'd like you to briefly comment on. I'm providing the full webpage context below:\n\n"
            f"{full_text}\n\n"
            f"Please provide 1-2 sentences about what makes this recipe interesting or unique."
        )
    }
    messages = [system_message, user_message]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    logger.info("Chat template for analysis successfully applied")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    logger.info(f"Starting recipe analysis text generation...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    analysis_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    sentences = re.split(r'(?<=[.!?])\s+', analysis_text.strip())
    if len(sentences) > 2:
        analysis_text = ' '.join(sentences[:2])
    logger.info(f"Recipe analysis successfully generated (length: {len(analysis_text)} characters)")
    return analysis_text

# Streaming response generator for multiple recipes
async def recipe_stream_generator(recipe_request: RecipeRequest):
    try:
        query = recipe_request.query
        logger.info(f"Streaming recipes for query: {query}")
        recipes = await search_multiple_recipes(query, max_recipes=3)
        if not recipes:
            yield json.dumps({"error": f"No recipes found for '{query}'"}) + "\n"
            return
        for recipe in recipes:
            detailed_recipe = await extract_recipe_details(recipe['link'])
            recipe_analysis = await generate_recipe_analysis(
                detailed_recipe,
                query,
                max_length=256,
                temperature=recipe_request.temperature
            )
            yield json.dumps({
                "type": "recipe",
                "content": f"# Recipe {recipe['index']}: {detailed_recipe['title']}\n\nLink: {detailed_recipe['source_url']}\n\n**Chef's Note:** {recipe_analysis}"
            }) + "\n"
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error in streaming recipes: {str(e)}")
        yield json.dumps({"error": f"Error processing request: {str(e)}"}) + "\n"

# Endpoint for streaming recipes and comparison
@app.post("/stream_recipes")
async def stream_recipes(request: RecipeRequest):
    return StreamingResponse(
        recipe_stream_generator(request),
        media_type="text/event-stream"
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