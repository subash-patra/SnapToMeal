// =============================================================
// Snap-to-Meal — Main App Component
// =============================================================
// This is the heart of the frontend. It manages:
//   1. Photo upload / camera capture
//   2. Sending the photo to our backend for ingredient detection
//   3. Letting the user confirm/edit ingredients
//   4. Requesting and displaying recipe suggestions
// =============================================================

import { useState, useRef, useCallback } from 'react'
import './App.css'

// The URL of our backend server (uses relative path in production on Vercel, or localhost in development)
const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

// Category emoji mapping for ingredient chips
const categoryEmoji = {
  produce: '\u{1F966}',    // broccoli
  protein: '\u{1F969}',    // meat
  dairy: '\u{1F9C0}',      // cheese
  condiment: '\u{1F9C2}',  // salt
  grain: '\u{1F33E}',      // wheat
  beverage: '\u{1F964}',   // cup
  other: '\u{1F371}',      // bento
}

// =============================================================
// MAIN APP COMPONENT
// =============================================================
function App() {
  // --- State ---
  // "step" tracks where the user is in the flow:
  //   'upload'      → Waiting for photo
  //   'analyzing'   → AI is scanning the photo
  //   'ingredients' → Show detected ingredients for review
  //   'generating'  → AI is generating recipes
  //   'recipes'     → Show recipe suggestions
  const [step, setStep] = useState('upload')
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [ingredients, setIngredients] = useState([])
  const [recipes, setRecipes] = useState([])
  const [selectedRecipe, setSelectedRecipe] = useState(null)
  const [error, setError] = useState(null)
  const [newIngredient, setNewIngredient] = useState('')
  const [dietaryPref, setDietaryPref] = useState('')
  const [isDragging, setIsDragging] = useState(false)

  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)

  // --- Helper: Handle file selection ---
  const handleFileSelect = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select an image file (JPEG, PNG, etc.)')
      return
    }
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setError(null)
  }, [])

  // --- Drag & Drop handlers ---
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }, [handleFileSelect])

  // --- Step 1: Send photo to backend for ingredient extraction ---
  const analyzePhoto = async () => {
    if (!imageFile) return

    setStep('analyzing')
    setError(null)

    try {
      // Create FormData to send the image file
      const formData = new FormData()
      formData.append('image', imageFile)

      // Send to our backend
      const response = await fetch(`${API_URL}/api/extract-ingredients`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Failed to analyze photo')
      }

      const data = await response.json()
      setIngredients(data.ingredients || [])
      setStep('ingredients')
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
      setStep('upload')
    }
  }

  // --- Step 2: Send confirmed ingredients to get recipes ---
  const generateRecipes = async () => {
    if (ingredients.length === 0) return

    setStep('generating')
    setError(null)

    try {
      // Build the ingredient names list
      const ingredientNames = ingredients.map((i) => i.name)

      // Send as form data (matching our backend endpoint)
      const formData = new FormData()
      formData.append('ingredients', JSON.stringify(ingredientNames))
      if (dietaryPref.trim()) {
        formData.append('dietary_preferences', dietaryPref.trim())
      }

      const response = await fetch(`${API_URL}/api/generate-recipes`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Failed to generate recipes')
      }

      const data = await response.json()
      setRecipes(data.recipes || [])
      setStep('recipes')
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
      setStep('ingredients')
    }
  }

  // --- Ingredient management ---
  const removeIngredient = (index) => {
    setIngredients((prev) => prev.filter((_, i) => i !== index))
  }

  const addIngredient = () => {
    if (!newIngredient.trim()) return
    setIngredients((prev) => [
      ...prev,
      {
        name: newIngredient.trim().toLowerCase(),
        category: 'other',
        estimated_quantity: null,
        confidence: 1.0,
        state: 'fresh',
      },
    ])
    setNewIngredient('')
  }

  // --- Reset to start over ---
  const resetApp = () => {
    setStep('upload')
    setImageFile(null)
    setImagePreview(null)
    setIngredients([])
    setRecipes([])
    setSelectedRecipe(null)
    setError(null)
    setNewIngredient('')
    setDietaryPref('')
  }

  // --- Determine step status for the header indicator ---
  const getStepStatus = (targetStep) => {
    const order = ['upload', 'analyzing', 'ingredients', 'generating', 'recipes']
    const currentIndex = order.indexOf(step)
    const targetIndex = order.indexOf(targetStep)
    if (targetIndex < currentIndex) return 'done'
    if (targetIndex === currentIndex) return 'active'
    return ''
  }

  // --- Get recipe card style class by category ---
  const getRecipeCardClass = (category) => {
    if (category.toLowerCase().includes('quick')) return 'recipe-card--quick'
    if (category.toLowerCase().includes('best')) return 'recipe-card--best'
    return 'recipe-card--gourmet'
  }

  // =============================================================
  // RENDER
  // =============================================================
  return (
    <div className="app">
      {/* ---- HEADER ---- */}
      <header className="header" id="header">
        <div className="header__brand" onClick={resetApp} style={{ cursor: 'pointer' }}>
          <span className="header__logo">{'\u{1F4F8}'}</span>
          <div>
            <div className="header__title">Snap to Meal</div>
            <div className="header__subtitle">AI Recipe Generator</div>
          </div>
        </div>

        <div className="header__step-indicator">
          <div className={`header__step header__step--${getStepStatus('upload')}`}>
            <span className="header__step-dot"></span> Upload
          </div>
          <div className="header__step-divider"></div>
          <div className={`header__step header__step--${getStepStatus('ingredients')}`}>
            <span className="header__step-dot"></span> Review
          </div>
          <div className="header__step-divider"></div>
          <div className={`header__step header__step--${getStepStatus('recipes')}`}>
            <span className="header__step-dot"></span> Recipes
          </div>
        </div>
      </header>

      {/* ---- MAIN CONTENT ---- */}
      <main className="main" id="main-content">

        {/* ======= UPLOAD STEP ======= */}
        {step === 'upload' && (
          <>
            <section className="hero">
              <span className="hero__emoji">{'\u{1F373}'}</span>
              <h1 className="hero__title">
                What's in your <span className="hero__title-gradient">fridge?</span>
              </h1>
              <p className="hero__description">
                Snap a photo of your fridge, pantry, or countertop. Our AI will identify
                your ingredients and suggest delicious meals you can cook right now.
              </p>
            </section>

            {!imagePreview ? (
              /* --- Upload Zone --- */
              <div
                className={`upload-zone ${isDragging ? 'upload-zone--dragging' : ''}`}
                id="upload-zone"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="upload-zone__icon">{'\u{1F4E4}'}</span>
                <div className="upload-zone__text">
                  Drop your photo here, or click to browse
                </div>
                <div className="upload-zone__subtext">
                  Supports JPEG, PNG, WebP — Max 10MB
                </div>

                <input
                  ref={fileInputRef}
                  className="upload-zone__input"
                  type="file"
                  accept="image/*"
                  id="file-input"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                />

                <div className="upload-zone__or">or</div>

                <button
                  className="upload-zone__camera-btn"
                  id="camera-btn"
                  onClick={(e) => {
                    e.stopPropagation()
                    cameraInputRef.current?.click()
                  }}
                >
                  {'\u{1F4F7}'} Take a Photo
                </button>
                <input
                  ref={cameraInputRef}
                  className="upload-zone__input"
                  type="file"
                  accept="image/*"
                  capture="environment"
                  id="camera-input"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                />
              </div>
            ) : (
              /* --- Image Preview --- */
              <div className="preview" id="image-preview">
                <div className="preview__container">
                  <img
                    src={imagePreview}
                    alt="Selected food photo"
                    className="preview__image"
                  />
                  <div className="preview__overlay">
                    <span className="preview__filename">{imageFile?.name}</span>
                    <button
                      className="preview__change-btn"
                      id="change-photo-btn"
                      onClick={() => {
                        setImageFile(null)
                        setImagePreview(null)
                      }}
                    >
                      Change Photo
                    </button>
                  </div>
                </div>
                <div className="btn-group">
                  <button
                    className="btn btn--primary btn--large"
                    id="analyze-btn"
                    onClick={analyzePhoto}
                  >
                    {'\u{2728}'} Analyze Ingredients
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {/* ======= ANALYZING STEP (Loading) ======= */}
        {step === 'analyzing' && (
          <div className="loading" id="analyzing-loader">
            <div className="loading__spinner"></div>
            <div className="loading__text">Scanning your photo...</div>
            <div className="loading__subtext">
              AI is identifying every ingredient it can see
            </div>
          </div>
        )}

        {/* ======= INGREDIENTS STEP ======= */}
        {step === 'ingredients' && (
          <section className="ingredients-section" id="ingredients-section">
            <h2 className="ingredients-section__title">
              {'\u{1F50D}'} We found {ingredients.length} ingredients
            </h2>
            <p className="ingredients-section__subtitle">
              Review the list below. Remove anything wrong and add anything we missed.
            </p>

            {/* Ingredient chips */}
            <div className="ingredients-grid" id="ingredients-grid">
              {ingredients.map((item, index) => (
                <div
                  className="ingredient-chip"
                  key={`${item.name}-${index}`}
                  style={{ animationDelay: `${index * 0.05}s` }}
                >
                  <span className="ingredient-chip__category">
                    {categoryEmoji[item.category] || '\u{1F371}'}
                  </span>
                  <span>{item.name}</span>
                  {item.estimated_quantity && (
                    <span className="ingredient-chip__qty">{item.estimated_quantity}</span>
                  )}
                  <button
                    className="ingredient-chip__remove"
                    onClick={() => removeIngredient(index)}
                    title={`Remove ${item.name}`}
                    aria-label={`Remove ${item.name}`}
                  >
                    {'\u2715'}
                  </button>
                </div>
              ))}
            </div>

            {/* Add ingredient input */}
            <div className="add-ingredient" id="add-ingredient">
              <input
                className="add-ingredient__input"
                type="text"
                placeholder="Add a missing ingredient..."
                value={newIngredient}
                onChange={(e) => setNewIngredient(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addIngredient()}
                id="add-ingredient-input"
              />
              <button
                className="add-ingredient__btn"
                onClick={addIngredient}
                id="add-ingredient-btn"
              >
                + Add
              </button>
            </div>

            {/* Dietary preferences */}
            <div className="dietary-section">
              <label className="dietary-section__label" htmlFor="dietary-input">
                Any dietary preferences or restrictions? (optional)
              </label>
              <input
                className="dietary-section__input"
                type="text"
                id="dietary-input"
                placeholder="e.g., Vegetarian, No nuts, Indian cuisine, Quick meals..."
                value={dietaryPref}
                onChange={(e) => setDietaryPref(e.target.value)}
              />
            </div>

            {/* Action buttons */}
            <div className="btn-group">
              <button
                className="btn btn--secondary"
                onClick={resetApp}
                id="back-btn"
              >
                {'\u2190'} Start Over
              </button>
              <button
                className="btn btn--primary btn--large"
                onClick={generateRecipes}
                disabled={ingredients.length === 0}
                id="generate-recipes-btn"
              >
                {'\u{1F373}'} Get Recipes ({ingredients.length} ingredients)
              </button>
            </div>
          </section>
        )}

        {/* ======= GENERATING STEP (Loading) ======= */}
        {step === 'generating' && (
          <div className="loading" id="generating-loader">
            <div className="loading__spinner"></div>
            <div className="loading__text">Cooking up recipes...</div>
            <div className="loading__subtext">
              Our AI chef is creating personalized meals just for you
            </div>
          </div>
        )}

        {/* ======= RECIPES STEP ======= */}
        {step === 'recipes' && (
          <section className="recipes-section" id="recipes-section">
            <h2 className="recipes-section__title">
              {'\u{1F37D}\uFE0F'} Your Personalized Recipes
            </h2>
            <p className="recipes-section__subtitle">
              Here are 3 meal ideas based on your ingredients. Click any recipe to see full details.
            </p>

            {/* Selected recipe detail view */}
            {selectedRecipe !== null && recipes[selectedRecipe] && (
              <RecipeDetail
                recipe={recipes[selectedRecipe]}
                onClose={() => setSelectedRecipe(null)}
              />
            )}

            {/* Recipe cards grid */}
            <div className="recipes-grid" id="recipes-grid">
              {recipes.map((recipe, index) => (
                <div
                  className={`recipe-card ${getRecipeCardClass(recipe.category)}`}
                  key={index}
                  onClick={() => setSelectedRecipe(index)}
                  style={{ animationDelay: `${index * 0.15}s` }}
                  id={`recipe-card-${index}`}
                >
                  <span className="recipe-card__badge">{recipe.category}</span>
                  <h3 className="recipe-card__title">{recipe.title}</h3>
                  <p className="recipe-card__description">{recipe.description}</p>

                  <div className="recipe-card__meta">
                    <span className="recipe-card__meta-item">
                      {'\u23F1\uFE0F'} {recipe.prep_time_minutes + recipe.cook_time_minutes} min
                    </span>
                    <span className="recipe-card__meta-item">
                      {'\u{1F4AA}'} {recipe.difficulty}
                    </span>
                    <span className="recipe-card__meta-item">
                      {'\u{1F374}'} {recipe.servings} servings
                    </span>
                  </div>

                  {recipe.dietary_tags?.length > 0 && (
                    <div className="recipe-card__tags">
                      {recipe.dietary_tags.map((tag, i) => (
                        <span className="recipe-card__tag" key={i}>{tag}</span>
                      ))}
                    </div>
                  )}

                  <div className="recipe-card__expand">
                    View full recipe {'\u2192'}
                  </div>
                </div>
              ))}
            </div>

            {/* Back buttons */}
            <div className="btn-group">
              <button
                className="btn btn--secondary"
                onClick={() => setStep('ingredients')}
                id="back-to-ingredients-btn"
              >
                {'\u2190'} Edit Ingredients
              </button>
              <button
                className="btn btn--primary"
                onClick={resetApp}
                id="new-scan-btn"
              >
                {'\u{1F4F8}'} New Scan
              </button>
            </div>
          </section>
        )}

        {/* ======= ERROR STATE ======= */}
        {error && (
          <div className="error" id="error-message">
            <div className="error__icon">{'\u26A0\uFE0F'}</div>
            <div className="error__title">Something went wrong</div>
            <div className="error__message">{error}</div>
            <button className="btn btn--secondary" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}
      </main>

      {/* ---- FOOTER ---- */}
      <footer className="footer" id="footer">
        <p className="footer__disclaimer">
          {'\u26A0\uFE0F'} AI may not detect all items accurately. Always check for food
          safety, allergens, and expiry dates before cooking.
        </p>
      </footer>
    </div>
  )
}


// =============================================================
// Recipe Detail Component
// Shows the full recipe with all steps when a user clicks a card
// =============================================================
function RecipeDetail({ recipe, onClose }) {
  return (
    <div className="recipe-detail" id="recipe-detail">
      <div className="recipe-detail__header">
        <h3 className="recipe-detail__title">{recipe.title}</h3>
        <button
          className="recipe-detail__close"
          onClick={onClose}
          id="close-recipe-btn"
        >
          {'\u2715'} Close
        </button>
      </div>

      {/* Meta info cards */}
      <div className="recipe-detail__meta">
        <div className="recipe-detail__meta-card">
          <div className="recipe-detail__meta-label">Prep</div>
          <div className="recipe-detail__meta-value">{recipe.prep_time_minutes}m</div>
        </div>
        <div className="recipe-detail__meta-card">
          <div className="recipe-detail__meta-label">Cook</div>
          <div className="recipe-detail__meta-value">{recipe.cook_time_minutes}m</div>
        </div>
        <div className="recipe-detail__meta-card">
          <div className="recipe-detail__meta-label">Servings</div>
          <div className="recipe-detail__meta-value">{recipe.servings}</div>
        </div>
        <div className="recipe-detail__meta-card">
          <div className="recipe-detail__meta-label">Difficulty</div>
          <div className="recipe-detail__meta-value">{recipe.difficulty}</div>
        </div>
      </div>

      {/* Ingredients used & missing */}
      <div className="recipe-detail__section">
        <h4 className="recipe-detail__section-title">Ingredients</h4>
        <div className="recipe-detail__ingredients-grid">
          {recipe.used_ingredients.map((item, i) => (
            <span key={i} className="recipe-detail__ingredient recipe-detail__ingredient--used">
              {'\u2713'} {item}
            </span>
          ))}
          {recipe.missing_ingredients.map((item, i) => (
            <span key={i} className="recipe-detail__ingredient recipe-detail__ingredient--missing">
              + {item} (need to buy)
            </span>
          ))}
        </div>
      </div>

      {/* Step-by-step instructions */}
      <div className="recipe-detail__section">
        <h4 className="recipe-detail__section-title">Step-by-Step Instructions</h4>
        <ol className="recipe-detail__steps">
          {recipe.instructions.map((step) => (
            <li key={step.step_number} className="recipe-detail__step">
              <span className="recipe-detail__step-number">{step.step_number}</span>
              <span className="recipe-detail__step-text">{step.instruction}</span>
            </li>
          ))}
        </ol>
      </div>

      {/* Chef tip */}
      {recipe.chef_tip && (
        <div className="recipe-detail__tip">
          <span className="recipe-detail__tip-icon">{'\u{1F468}\u200D\u{1F373}'}</span>
          <span className="recipe-detail__tip-text">
            <strong>Chef Tip:</strong> {recipe.chef_tip}
          </span>
        </div>
      )}
    </div>
  )
}

export default App
