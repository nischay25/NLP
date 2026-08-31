from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP_SG VP_SG | NP_PL VP_PL

NP_SG -> 'he' | 'she'
NP_PL -> 'they'

VP_SG -> 'runs' | 'eats'
VP_PL -> 'run' | 'eat'
""")

sentence = input("Enter sentence: ").lower().split()

parser = ChartParser(grammar)

trees = list(parser.parse(sentence))

if trees:
    print("\nSentence is Grammatically Correct.")
else:
    print("\nSentence is NOT Grammatically Correct.")
