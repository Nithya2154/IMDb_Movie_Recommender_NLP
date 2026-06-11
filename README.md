# 🎬 IMDB Movie Recommendation System

A content-based movie recommendation system built using NLP and TF-IDF vectorization, trained on ~5,000 IMDB movies from 2024. Users describe a story or plot, and the system returns the top 5 most similar movies using cosine similarity.

---

## 📁 Project Structure

```
├── imdb_scraper.ipynb                  # Web scraper to collect IMDB movie data
├── imdb_recommendation_model.ipynb     # Data preprocessing and model training
├── movie_recommendation.py                              # Streamlit web application
├── imdb_movies_2024.csv                # Raw scraped dataset (~5,088 movies)
├── cleaned_movies.csv                  # Preprocessed dataset with NLP content column
└── movie_recommendation_model.pkl      # Serialized TF-IDF model and matrix
```

---

## ⚙️ How It Works

### 1. Data Collection (`imdb_scraper.ipynb`)
- Uses **Selenium** with Chrome WebDriver to scrape IMDB's 2024 feature film listings.
- Clicks "Load More" repeatedly to paginate through all results.
- Extracts **Title**, **Rating**, **Votes**, **Duration**, and **Storyline** for each movie.
- Saves the raw data to `imdb_movies_2024.csv`.

### 2. Preprocessing & Model Training (`imdb_recommendation_model.ipynb`)
- Loads the raw dataset and cleans the `Title` column (removes numbering prefixes and suffixes).
- Combines `Title` and `Storyline` into a single `content` field for richer NLP representation.
- Applies a custom `text_processing()` pipeline to the content:
  - Lowercasing
  - Contraction expansion (`don't` → `do not`)
  - Emoji demojization
  - Punctuation and special character removal
  - Stop word removal
  - Lemmatization via `WordNetLemmatizer`
- Trains a **TF-IDF Vectorizer** (`max_features=5000`, `ngram_range=(1,2)`) on the cleaned content.
- Saves the trained vectorizer, TF-IDF matrix, and dataframe to `movie_recommendation_model.pkl`.

### 3. Recommendation Engine
- Takes a user's free-text story/query as input.
- Applies the same `text_processing()` pipeline to the query.
- Transforms it using the saved TF-IDF vectorizer.
- Computes **cosine similarity** against all movie vectors.
- Returns the **top 5 most similar movies** with their metadata and similarity scores.

### 4. Web App (`movie_recommendation.py`)
- Built with **Streamlit**.
- Provides a text area for the user to enter a story or plot description.
- Displays the top 5 recommended movies with Title, Rating, Votes, Duration, Storyline, and Similarity Score.

---

## 🗂️ Dataset

| File | Rows | Columns |
|------|------|---------|
| `imdb_movies_2024.csv` | ~5,088 | Title, Rating, Votes, Duration, Storyline |
| `cleaned_movies.csv` | ~5,088 | Title, Rating, Votes, Duration, Storyline, content |

All movies are 2024 feature films scraped from [IMDB](https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31).

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Web Scraping | `selenium` |
| Data Handling | `pandas`, `numpy` |
| NLP | `nltk`, `contractions`, `emoji` |
| ML / Similarity | `scikit-learn` (TF-IDF, cosine similarity) |
| Model Persistence | `pickle` |
| Web App | `streamlit` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Chrome + ChromeDriver (for scraping only)

### Installation

```bash
pip install pandas numpy nltk scikit-learn streamlit contractions emoji selenium
```

Download required NLTK resources (one-time setup):

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### Running the App

Ensure `movie_recommendation_model.pkl` is in the same directory as `movie_recommendation.py`, then run:

```bash
streamlit run movie_recommendation.py
```

Open your browser at `http://localhost:8501`.

### Re-scraping Data (Optional)

To collect fresh data, open and run `imdb_scraper.ipynb`. This requires Chrome and ChromeDriver installed and accessible in your PATH.

### Re-training the Model (Optional)

To retrain after updating the dataset, open and run `imdb_recommendation_model.ipynb` top to bottom. It will overwrite `cleaned_movies.csv` and `movie_recommendation_model.pkl`.

---

## 💡 Example Usage

**Input:**
> "A group of soldiers embark on a secret mission behind enemy lines during World War II"

**Output:** Top 5 movies ranked by cosine similarity to the query, showing Title, Rating, Votes, Duration, Storyline, and Similarity Score.

---

## 📌 Notes

- The model is **content-based** — it matches your description to movie storylines, not user ratings or collaborative filtering.
- The `content` column combines `Title + Storyline` to improve matching accuracy.
- The Streamlit app loads the model once using `@st.cache_resource` for fast repeated queries.
- The app avoids NLTK's `word_tokenize` (which requires the `punkt` tokenizer download at runtime) and instead uses simple `str.split()` for safe, dependency-free deployment.
