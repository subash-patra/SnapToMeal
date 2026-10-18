# =============================================================
# Snap-to-Meal — Vercel Serverless API
# =============================================================

import os
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="Snap-to-Meal AI API",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not configured."
        )
    return genai.Client(api_key=api_key)

class DetectedIngredient(BaseModel):
    name: str = Field(description="Ingredient name, e.g., 'egg', 'tomato', 'chicken breast'")
    category: str = Field(description="'produce', 'protein', 'dairy', 'condiment', 'grain', 'beverage', or 'other'")
    estimated_quantity: Optional[str] = Field(default=None)
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    state: Optional[str] = Field(default=None)

class IngredientExtractionResponse(BaseModel):
    ingredients: List[DetectedIngredient]
    pantry_staples: List[str] = Field(
        default_factory=lambda: ["salt", "black pepper", "cooking oil", "water", "sugar"]
    )

class RecipeStep(BaseModel):
    step_number: int
    instruction: str

class Recipe(BaseModel):
    title: str
    description: str
    category: str
    difficulty: str
    prep_time_minutes: int
    cook_time_minutes: int
    servings: int
    used_ingredients: List[str]
    missing_ingredients: List[str]
    instructions: List[RecipeStep]
    chef_tip: str
    dietary_tags: List[str] = Field(default_factory=list)

class RecipeResponse(BaseModel):
    recipes: List[Recipe]

@app.get("/api/health")
@app.get("/api")
@app.get("/")
async def health():
    return {"status": "ok", "app": "Snap-to-Meal AI API"}

async def handle_extract(image: UploadFile):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    image_bytes = await image.read()
    client = get_gemini_client()

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
    5. Set confidence between 0.0 and 1.0.
    6. If you see leftovers in containers that you cannot identify, skip them.

    Return the response strictly in the required JSON format.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=image.content_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IngredientExtractionResponse,
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@app.post("/api/extract-ingredients", response_model=IngredientExtractionResponse)
@app.post("/extract-ingredients", response_model=IngredientExtractionResponse)
async def extract_ingredients(image: UploadFile = File(...)):
    return await handle_extract(image)

async def handle_recipes(ingredients: str, dietary_preferences: Optional[str] = None):
    client = get_gemini_client()

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
    - Tag each recipe with relevant dietary labels.
    - Do NOT suggest absurd flavor combinations.
    - Specify exact quantities in the instructions.

    Return the response strictly in the required JSON format.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RecipeResponse,
                temperature=0.7,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")

@app.post("/api/generate-recipes", response_model=RecipeResponse)
@app.post("/generate-recipes", response_model=RecipeResponse)
async def generate_recipes(
    ingredients: str = Form(...),
    dietary_preferences: Optional[str] = Form(None),
):
    return await handle_recipes(ingredients, dietary_preferences)
