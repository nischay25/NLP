import nltk
from nltk.wsd import lesk

nltk.download('wordnet')
nltk.download('omw-1.4')

sentence = input("Enter sentence: ")
word = input("Enter ambiguous word: ")

sense = lesk(sentence.split(), word)

if sense:
    print("\nBest Sense:")
    print(sense.name())
    print(sense.definition())
else:
    print("No sense found.")
