word = input("Enter a word: ")

if word.endswith("ing"):
    print(word, "- VBG")
elif word.endswith("ed"):
    print(word, "- VBD")
else:
    print(word, "- NN")
