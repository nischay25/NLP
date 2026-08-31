from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Python is a programming language",
    "Machine learning uses Python",
    "Artificial Intelligence and Machine Learning",
    "Python is easy to learn"
]

query = input("Enter search query: ")

vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(documents + [query])

similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

scores = similarity.flatten()

ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("\nDocument Ranking:\n")

for index, score in ranked:
    print("Document", index + 1)
    print(documents[index])
    print("Score:", round(score, 3))
    print()
