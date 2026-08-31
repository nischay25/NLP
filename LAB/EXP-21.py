import spacy


nlp = spacy.load("en_core_web_sm")

sentence = input("Enter a sentence: ")

doc = nlp(sentence)

print("\nNoun Phrases and Meanings:")

for chunk in doc.noun_chunks:
    print("Noun Phrase:", chunk.text)
    print("Root Word:", chunk.root.text)
    print("Meaning/Role:", chunk.root.dep_)
    print()
