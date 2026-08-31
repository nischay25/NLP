import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'cat' | 'dog'
V -> 'chased' | 'saw'
""")

sentence = input("Enter sentence: ").strip().lower().split()

parser = RecursiveDescentParser(grammar)

trees = list(parser.parse(sentence))

if trees:
    print("Sentence Parsed Successfully!\n")
    for tree in trees:
        print(tree)
else:
    print("Sentence cannot be parsed.")
