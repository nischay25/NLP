word = input("Enter a word: ")

tag = "NN"  

if word.endswith("ing"):
    tag = "VBG"

print(word, "-", tag)
