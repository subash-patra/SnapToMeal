# =============================================================
# Snap-to-Meal Backend — The AI Brain 🧠
# =============================================================
# This file is the SERVER that runs behind the scenes.
# It does two main things:
#   1. Looks at a food photo and tells you what ingredients are in it
#   2. Takes those ingredients and suggests recipes you can cook
# =============================================================

import os
import json
from typing import List, Optional

# FastAPI — the framework that creates our web server
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Pydantic — helps us define the shape/structure of our data
from pydantic import BaseModel, Field

# Google Gemini AI — the AI that "sees" photos and generates recipes
from google import genai
from google.genai import types

# dotenv — loads our secret API key from the .env file
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()

# -------------------------------------------------------------
# Create the FastAPI app (this IS our server)
# -------------------------------------------------------------
app = FastAPI(
    title="Snap-to-Meal AI API",
    description="Upload a photo of your fridge/pantry and get recipe suggestions!",
    version="1.0.0",
)

# -------------------------------------------------------------
# CORS Middleware
# This allows the frontend (React, running on a different port)
# to talk to this backend. Without this, the browser would
# block requests from the frontend to the backend.
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Connect to Gemini AI using our API key
# -------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("[ERROR] GEMINI_API_KEY not found! Make sure it's in your .env file.")

client = genai.Client(api_key=api_key)


# =============================================================
# DATA MODELS (Pydantic)
# These define the "shape" of the data our API sends and receives.
# Think of them as templates that describe what the data looks like.
# =============================================================

class DetectedIngredient(BaseModel):
    """One single ingredient detected in the photo."""
    name: str = Field(description="Ingredient name, e.g., 'egg', 'tomato', 'chicken breast'")
    category: str = Field(description="Category: 'produce', 'protein', 'dairy', 'condiment', 'grain', 'beverage', or 'other'")
    estimated_quantity: Optional[str] = Field(
        default=None,
        description="Rough quantity, e.g., '3 units', 'half jar', 'about 200g'"
    )
    confidence: float = Field(description="How sure the AI is (0.0 = not sure, 1.0 = very sure)")
    state: Optional[str] = Field(
        default=None,
        description="Condition: 'fresh', 'ripe', 'canned', 'frozen', 'leftover', 'cooked'"
    )


class IngredientExtractionResponse(BaseModel):
    """The full response after analyzing a photo."""
    ingredients: List[DetectedIngredient]
    pantry_staples: List[str] = Field(
        default_factory=lambda: ["salt", "black pepper", "cooking oil", "water", "sugar"],
        description="Common kitchen staples assumed to be available"
    )


class RecipeStep(BaseModel):
    """One step in a recipe's cooking instructions."""
    step_number: int
    instruction: str


class Recipe(BaseModel):
    """A complete recipe suggestion."""
    title: str
    description: str
    category: str = Field(description="'Quick & Easy', 'Best Match', or 'Gourmet Upgrade'")
    difficulty: str = Field(description="'Beginner', 'Intermediate', or 'Advanced'")
    prep_time_minutes: int
    cook_time_minutes: int
    servings: int
    used_ingredients: List[str]
    missing_ingredients: List[str]
    instructions: List[RecipeStep]
    chef_tip: str
    dietary_tags: List[str] = Field(
        default_factory=list,
        description="e.g., 'Vegetarian', 'Gluten-Free', 'High-Protein'"
    )


class RecipeResponse(BaseModel):
    """The full response containing multiple recipe suggestions."""
    recipes: List[Recipe]


# =============================================================
# API ENDPOINTS (the URLs our frontend will call)
# =============================================================

# -------------------------------------------------------------
# Health check — a simple endpoint to test if the server is running
# Visit http://localhost:8000/ in your browser to test
# -------------------------------------------------------------
@app.get("/")
async def health_check():
    return {
        "status": "🟢 Snap-to-Meal API is running!",
        "endpoints": {
            "POST /extract-ingredients": "Upload a photo to detect ingredients",
            "POST /generate-recipes": "Get recipe suggestions from ingredients",
        }
    }


# -------------------------------------------------------------
# ENDPOINT 1: Extract Ingredients from a Photo
# 
# How it works:
#   1. User uploads a photo of their fridge/pantry/countertop
#   2. We send the photo to Gemini AI
#   3. Gemini AI "looks" at the photo and identifies all the food
#   4. We return a structured list of ingredients
# -------------------------------------------------------------
@app.post("/extract-ingredients", response_model=IngredientExtractionResponse)
async def extract_ingredients(image: UploadFile = File(...)):
    """
    Upload a photo of your fridge, pantry, or countertop.
    The AI will identify all visible food items and ingredients.
    """
    # Check that the uploaded file is actually an image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file (JPEG, PNG, etc.)"
        )

    # Read the image data
    image_bytes = await image.read()

    # The prompt tells Gemini AI exactly what we want it to do
    prompt = """
    You are a food ingredient detection expert. Analyze this photo of a 
    refrigerator, pantry, or kitchen countertop carefully.

    Your task:
    1. Identify ALL visible edible ingredients, produce, proteins, dairy, 
       condiments, sauces, beverages, and packaged foods.
    2. Ignore non-food items (containers, shelves, appliances) unless 
       food inside is clearly identifiable.
    3. For each ingredient, estimate the quantity and assess its condition 
       (fresh, ripe, canned, frozen, etc.).
    4. Use common canonical names (e.g., "bell pepper" not "capsicum", 
       "cilantro" not "coriander leaves").
    5. Set confidence between 0.0 and 1.0 — use lower values for 
       partially hidden or ambiguous items.
    6. If you see leftovers in containers that you cannot identify, 
       skip them rather than guessing.

    Return the response strictly in the required JSON format.
    """

    try:
        # Send the photo + prompt to Gemini AI
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=image.content_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IngredientExtractionResponse,
                temperature=0.2,  # Low temperature = more precise, less creative
            ),
        )
        result = json.loads(response.text)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )


# -------------------------------------------------------------
# ENDPOINT 2: Generate Recipes from Ingredients
# 
# How it works:
#   1. User sends their confirmed list of ingredients
#   2. We ask Gemini AI to act as a chef
#   3. Gemini AI suggests 3 different recipes:
#      - Quick & Easy (under 20 minutes)
#      - Best Match (uses most ingredients, nothing missing)
#      - Gourmet Upgrade (fancy dish, may need 1-2 extra items)
#   4. We return the recipes with step-by-step instructions
# -------------------------------------------------------------
@app.post("/generate-recipes", response_model=RecipeResponse)
async def generate_recipes(
    ingredients: str = Form(..., description="JSON string of validated ingredient names"),
    dietary_preferences: Optional[str] = Form(
        None, description="e.g., 'Vegetarian', 'Quick meals under 20 min', 'Indian cuisine'"
    ),
):
    """
    Send your confirmed ingredient list and get 3 recipe suggestions.
    Each recipe comes with step-by-step cooking instructions.
    """
    prompt = f"""
    You are a world-class chef creating practical, delicious meals.
    
    Based STRICTLY on the available ingredients below, suggest exactly 
    3 distinct meal recipes:

    1. **Quick & Easy**: The fastest meal to prepare (under 20 minutes).
    2. **Best Match**: Uses the most available ingredients with 0 missing items.
    3. **Gourmet Upgrade**: A more impressive dish that may require 1-2 
       common items from a store.

    AVAILABLE INGREDIENTS:
    {ingredients}

    DIETARY PREFERENCES / RESTRICTIONS:
    {dietary_preferences or 'None specified'}

    IMPORTANT RULES:
    - Assume standard pantry staples are available: water, salt, black 
      pepper, cooking oil, and sugar.
    - Every recipe MUST include clear step-by-step instructions.
    - Include realistic prep and cook times.
    - Add a helpful chef tip for each recipe.
    - Tag each recipe with relevant dietary labels 
      (e.g., Vegetarian, Gluten-Free, High-Protein).
    - Do NOT suggest absurd flavor combinations.
    - Specify exact quantities in the instructions.

    Return the response strictly in the required JSON format.
    """

    try:
        # Send the prompt to Gemini AI
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RecipeResponse,
                temperature=0.7,  # Higher temperature = more creative recipes
            ),
        )
        result = json.loads(response.text)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recipe generation failed: {str(e)}"
        )


# =============================================================
# This runs the server when you execute: python main.py
# =============================================================
if __name__ == "__main__":
    import uvicorn
    print("[*] Starting Snap-to-Meal server...")
    print("[*] API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
