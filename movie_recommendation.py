import streamlit as st
import pickle
import re
import contractions
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

#  Initialize (NO downloads here 🚀)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#  Text Cleaning Function (NO NLTK tokenizer )


def text_processing(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.demojize(text)

    # remove punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # keep only alphabetic words
    text = ' '.join([word for word in text.split() if word.isalpha()])

    #  simple tokenization (safe)
    tokens = text.split()

    # remove stopwords + lemmatization
    output_text = ' '.join([
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ])

    return output_text


#  Load pre-trained model (FAST)
@st.cache_resource
def load_model():
    with open("movie_recommendation_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

tfidf = model["tfidf"]
tfidf_matrix = model["matrix"]
df = model["data"]

# Recommendation Function


def recommendations_movie(query):
    query_clean = text_processing(query)
    query_vector = tfidf.transform([query_clean])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:5]
    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity_Score"] = similarities[top_indices]
    recommendations.index = range(1, len(recommendations) + 1)
    return recommendations.drop('content', axis=1)


# STREAMLIT UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Enter a story → Get **Top 5 similar movies**")
# Input box
user_input = st.text_area("✍️ Enter your story:")
# Button
if st.button("Recommend Movies"):
    if user_input.strip() == "":
        st.warning("Please enter a story!")
    else:
        with st.spinner("Finding best matches..."):
            # results = recommend_top5(user_input)
            results = recommendations_movie(user_input)

        st.success(" Top 5 Recommendations")

        for _, row in results.iterrows():
            st.markdown(f"### 🎥 {row['Title']}")
            st.write(f"**Rating:** {row['Rating']}")
            st.write(f"**Votes:** {row['Votes']}")
            st.write(f"**Duration:** {row['Duration']}")
            st.write(f"**Storyline:** {row['Storyline']}")
            st.write(f"**Similarity_Score:** {row['Similarity_Score']:.2f}")
            st.markdown("---")
