from typing import final
import nltk
import string
from heapq import nlargest
from nltk.tag import pos_tag
from string import punctuation
from inspect import getsourcefile
from collections import defaultdict
from nltk.tokenize import word_tokenize
from os.path import abspath, join, dirname
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer

from pathlib import Path
import json

class LexicalChain:

    def __init__(self):

        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
        self.documents = self.ReadFromDisk("train_data")
        self.chains = {}
        self.nouns = []

        self.POS_tagging()

        # print(self.chains)

        self.WriteToDisk(self.chains,"LexicalChain")

    def POS_tagging(self):
        
        docNo = [*self.documents.keys()]

        for no in docNo:

            self.nouns.clear()

            words = self.documents[no]
            tagged = nltk.pos_tag(words)

            for (tok, tag) in tagged:
                if tag == "NNP" or tag == "NN" or tag == "NNS" or tag == "NNPS":
                    self.nouns.append(tok)
                   
            relation = self.relation_list(self.nouns)
            lexical = self.create_lexical_chain(self.nouns, relation)
            final_chain = self.prune(lexical)

            all_keys = list(set().union(*(d.keys() for d in final_chain)))

            self.chains[no] = all_keys

    def relation_list(self,nouns):

        relation_list = defaultdict(list)
        
        for k in range (len(nouns)):   
            relation = []
            for syn in wordnet.synsets(nouns[k], pos = wordnet.NOUN):
                for l in syn.lemmas():
                    relation.append(l.name())
                    if l.antonyms():
                        relation.append(l.antonyms()[0].name())
                for l in syn.hyponyms():
                    if l.hyponyms():
                        relation.append(l.hyponyms()[0].name().split('.')[0])
                for l in syn.hypernyms():
                    if l.hypernyms():
                        relation.append(l.hypernyms()[0].name().split('.')[0])
            relation_list[nouns[k]].append(relation)
        return relation_list
        

    def create_lexical_chain(self,nouns, relation_list):
        lexical = []
        threshold = 0.5
        for noun in nouns:
            flag = 0
            for j in range(len(lexical)):
                if flag == 0:
                    for key in list(lexical[j]):
                        if key == noun and flag == 0:
                            lexical[j][noun] +=1
                            flag = 1
                        elif key in relation_list[noun][0] and flag == 0:
                            syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                            syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                            if syns1[0].wup_similarity(syns2[0]) >= threshold:
                                lexical[j][noun] = 1
                                flag = 1
                        elif noun in relation_list[key][0] and flag == 0:
                            syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                            syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                            if syns1[0].wup_similarity(syns2[0]) >= threshold:
                                lexical[j][noun] = 1
                                flag = 1
            if flag == 0: 
                dic_nuevo = {}
                dic_nuevo[noun] = 1
                lexical.append(dic_nuevo)
                flag = 1

        return lexical
        

    def prune(self,lexical):
        final_chain = []
        while lexical:
            result = lexical.pop()
            if len(result.keys()) == 1:
                for value in result.values():
                    if value != 1: 
                        final_chain.append(result)
            else:
                final_chain.append(result)

        return final_chain

    def WriteToDisk(self, index, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.dataDir + filename, "w") as filehandle:
            filehandle.write(json.dumps(index))

    # reading specified Index from Disk

    def ReadFromDisk(self, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.dataDir + filename, "r") as filehandle:
            index = json.loads(filehandle.read())

        return index


# l = LexicalChain()