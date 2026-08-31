from nltk import PCFG
from nltk.parse import ViterbiParser

grammar = PCFG.fromstring("""
S -> NP VP [1.0]

NP -> Det N [0.6]
NP -> 'john' [0.4]

VP -> V NP [1.0]

Det -> 'the' [0.5]
Det -> 'a' [0.5]

N -> 'dog' [0.5]
N -> 'cat' [0.5]

V -> 'sees' [1.0]
""")

sentence = input("Enter sentence: ").lower().split()

parser = ViterbiParser(grammar)

trees = list(parser.parse(sentence))

if trees:
    print("\nMost Probable Parse Tree:\n")
    for tree in trees:
        print(tree)
else:
    print("\nSentence cannot be parsed.")
