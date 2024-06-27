import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download the punkt tokenizer for sentence splitting
nltk.download('punkt')

# Preprocess the text
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    documents = [text1, text2]
    # Create TF-IDF vectors for the documents
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    # Compute the cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

# Example texts
text1 = "This is a sample text to check for plagiarism."
text2 = "This text is a sample used to check for plagiarism."

# Preprocess the texts
text1_processed = preprocess(text1)
text2_processed = preprocess(text2)

# Calculate similarity
similarity = calculate_similarity(text1_processed, text2_processed)
print(f"Cosine Similarity: {similarity:.2f}")

# Define a threshold for plagiarism detection
threshold = 0.8
if similarity > threshold:
    print("Potential plagiarism detected.")
else:
    print("No plagiarism detected.")


