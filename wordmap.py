import sqlite3
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.util import ngrams

# Download stopwords if not already present
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Custom filler words that don't add meaning
custom_stopwords = {
    "like", "think", "also", "really", "actually", "probably", "just",
    "people", "know", "one", "get", "going", "would", "even", "much",
    "could", "many", "still", "way", "said", "use", "used", "well",
    "want", "work", "better", "thing", "make", "dont", "doesnt",
    "thats", "us", "im", "something", "years", "good", "tax", "money"
}
stop_words.update(custom_stopwords)

# SQLite Database Connection
DATABASE_URL = "sqlite:///hackernews.db"

# Fetch All Comments From Today
def get_todays_comments():
    from datetime import datetime
    today = datetime.utcnow().date()
    query = f"SELECT text FROM comments WHERE date(timestamp) = '{today}'"

    with sqlite3.connect("hackernews.db") as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]

# Preprocess Text (Clean & Tokenize)
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]  # Remove stopwords and short words
    return words

# Generate Bigram and Trigram Phrases
def generate_ngrams(comments, n=2):
    ngram_counts = Counter()
    
    for comment in comments:
        words = clean_text(comment)
        ngram_counts.update([" ".join(ngram) for ngram in ngrams(words, n)])

    return ngram_counts.most_common(50)  # Return top 50 n-grams

# Generate Word Frequency Map with TF-IDF
def generate_tfidf_word_map(comments):
    cleaned_comments = [" ".join(clean_text(comment)) for comment in comments]
    
    # Use TF-IDF Vectorizer to find meaningful words
    vectorizer = TfidfVectorizer(max_features=100, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(cleaned_comments)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Create a word-frequency dictionary using TF-IDF scores
    word_tfidf = dict(zip(feature_array, tfidf_scores))

    return word_tfidf

# Generate Word Cloud Image
def create_word_cloud(word_counts):
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        colormap="viridis",
        max_words=75
    ).generate_from_frequencies(word_counts)

    # Show the word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Hacker News Comment Word Cloud (TF-IDF & Phrases)")
    plt.show()

    # Save to file with timestamp
    import datetime
    timestamp = int(datetime.datetime.now().timestamp())
    filename = f"hn_wordcloud_{timestamp}.png"
    wordcloud.to_file(filename)
    print(f"✅ Word cloud saved as '{filename}'")

if __name__ == "__main__":
    print("Fetching today's Hacker News comments...")
    comments = get_todays_comments()

    if not comments:
        print("No comments found for today.")
    else:
        print(f"Processing {len(comments)} comments...")
        
        # Generate meaningful word map with TF-IDF
        word_map = generate_tfidf_word_map(comments)
        
        # Get important bigram phrases
        bigram_phrases = generate_ngrams(comments, n=2)
        trigram_phrases = generate_ngrams(comments, n=3)
        
        print("Top Bigrams:", bigram_phrases[:10])
        print("Top Trigrams:", trigram_phrases[:10])

        create_word_cloud(word_map)
        print("✅ Word cloud generated successfully!")
