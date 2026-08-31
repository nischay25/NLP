from nltk import bigrams, word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

text = input("Enter a sentence: ")

words = word_tokenize(text)

print("Bigrams:")
for bg in bigrams(words):
    print(bg)
