from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'cat' | 'dog'
V -> 'chased' | 'saw'
""")

sentence = input("Enter sentence: ").lower().split()

parser = ChartParser(grammar)

trees = list(parser.parse(sentence))

if trees:
    print("\nParse Tree:\n")
    for tree in trees:
        print(tree)
        tree.pretty_print()
else:
    print("Sentence cannot be parsed.")
