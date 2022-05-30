from pydoc_data.topics import topics
import nltk
from pathlib import Path
import os
import json
import heapq
import re

import gensim
import gensim.corpora as corpora

"""

 2nd Feature Selection Strategy => Get Top 50 Nouns and Topic Base sets
 
 - Extracts Top 50 most occuring nouns after pos_tagging
 - Gets terms from overall collection after seperating terms in 10 topic sets


"""

class TopicTerms:

    def __init__(self):

        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
        self.vocablary = self.ReadFromDisk("vocablary")
        self.nouns = {}
        self.remTerms = []

        self.POS_tagging()
        self.getTopKNouns()
        self.extractTopicSets()

        self.WriteToDisk(self.nouns, "topKNouns")
        self.WriteToDisk(self.topic_words, "kTopicSets")

        print("\nTopics Sets Generated => Saved in kTopicSets.txt")

    def POS_tagging(self):
        
        words = [*self.vocablary.keys()]

        # get nltk tagging for each token
 
        tagged = nltk.pos_tag(words)

        for (tok, tag) in tagged:
            if tag == "NNP" or tag == "NN" or tag == "NNS" or tag == "NNPS":
                self.nouns[tok] = self.vocablary[tok]
            else:
                self.remTerms.append(tok)


    def getTopKNouns(self, k=50):

        # Top 50 Nouns
        self.nouns = heapq.nlargest(k, self.nouns, key=self.nouns.get)

        print("\nTop 50 Nouns Selected => Saved in topKNouns.txt")


    def extractTopicSets(self, num_topic=10):

        # co-occurrences term in each Topic Set
        # Converting text to bag of words

        remTerms = [self.remTerms]
        id2word = gensim.corpora.Dictionary(remTerms)
        corpus = [id2word.doc2bow(text) for text in remTerms]

        # Segregate terms in 10 different topic sets 
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=10, id2word=id2word, passes=10)

        topics = self.lda_model.show_topics()

        self.topic_words={}

        for topic,word in topics:
            
            line = re.sub('[^A-Za-z ]+', '', word)
            self.topic_words[topic] = line.split('  ')


    # writing specified Index to Disk

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
