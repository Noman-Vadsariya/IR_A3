from xml.dom.minidom import Document
import nltk
from collections import defaultdict
from nltk.corpus import wordnet


syn = wordnet.synsets("document")

words = [word.name().split(".")[0] for word in syn]

hyper = [word.hypernyms() for word in syn]
lemma = [word.lemmas() for word in syn]

print(words)

print(lemma)
print(hypo)
print(hyper)
