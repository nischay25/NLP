import spacy

nlp = spacy.load("en_core_web_sm")

text = input("Enter a paragraph: ")

doc = nlp(text)

sentences = list(doc.sents)

if len(sentences) < 2:
    print("Please enter at least two sentences.")
else:
    similarities = []

    print("\nSentence Similarities:")

    for i in range(len(sentences) - 1):
        similarity = sentences[i].similarity(sentences[i + 1])
        similarities.append(similarity)

        print(
            f"Sentence {i + 1} and Sentence {i + 2}: "
            f"{similarity:.2f}"
        )

    average_score = sum(similarities) / len(similarities)

    print("\nAverage Coherence Score:", round(average_score, 2))

    if average_score > 0.75:
        print("The text has good coherence.")
    elif average_score > 0.40:
        print("The text has moderate coherence.")
    else:
        print("The text has low coherence.")
