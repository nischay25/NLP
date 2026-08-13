import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')s
word ="car"
synsets = wordnet.synsets(word)
print("word:",word)
print("number of synsets: ",len(synsets))
6
for synset in synsets:
  print("\nSynsets: ",synset.name())
  print("Definition: ",synset.definition())
  print("Examples: ",synset.examples())

  synonyms = synset.lemmas()
  print("Synonyms: ",[lemma.name() for lemma in synonyms])
  