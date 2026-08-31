import re

word = input("Enter a word: ")

if re.search("ing$", word):
    print(word, "- Verb (VBG)")
elif re.search("ly$", word):
    print(word, "- Adverb (RB)")
elif re.search("ion$", word):
    print(word, "- Noun (NN)")
else:
    print(word, "- Unknown")
