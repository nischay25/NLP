import spacy

nlp = spacy.load("en_core_web_sm")

text = input("Enter a text: ")

doc = nlp(text)

nouns = []

for token in doc:
    if token.pos_ in ["NOUN", "PROPN"]:
        nouns.append(token.text)

print("\nReference Resolution:")

for token in doc:
    if token.pos_ == "PRON":
        if nouns:
            reference = nouns[-1]
            print(f"{token.text} -> {reference}")
        else:
            print(f"{token.text} -> Reference not found")
