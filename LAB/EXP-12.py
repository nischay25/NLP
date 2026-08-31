from nltk import CFG
from nltk.parse import EarleyChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'boy' | 'girl' | 'apple'
V -> 'likes' | 'eats'
""")

sentence = input("Enter sentence: ").lower().split()

parser = EarleyChartParser(grammar)

trees = list(parser.parse(sentence))

if trees:
    print("\nSentence Parsed Successfully!\n")
    for tree in trees:
        print(tree)
else:
    print("Sentence cannot be parsed.")
