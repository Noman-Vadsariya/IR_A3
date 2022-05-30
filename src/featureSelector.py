import json
from pathlib import Path
import math
import heapq

class FeatureSpaceBuilder:

    def __init__(self):
        
        self.DataDir = str(Path(__file__).parent.resolve()).replace("src", "data")  # Folder to Store Indexes
        self.feature_set = []
        self.load_features()


    def load_features(self):

        self.tfidf_features = self.ReadFromDisk('tfidf_topKFeatures')

        self.noun_features = self.ReadFromDisk('topKNouns')

        self.topicSets = self.ReadFromDisk('kTopicSets')
        
        self.lexical_chain = self.ReadFromDisk('LexicalChain')

        self.add_vector_to_features(self.tfidf_features)
        self.add_vector_to_features(self.noun_features)
        self.add_dict_to_features(self.topicSets)
        self.add_dict_to_features(self.lexical_chain)

        self.WriteToDisk(self.feature_set,"feature_set")

        print(f"Complete Size of Feature Set: {len(self.feature_set)}")

    def add_vector_to_features(self, vector):

        for tok in vector:

            if tok not in self.feature_set:

                self.feature_set.append(tok)


    def add_dict_to_features(self,dict):

        for key in dict.keys():
            
            for tok in dict[key]:
                
                if tok not in self.feature_set:

                    self.feature_set.append(tok)


    def WriteToDisk(self, index, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.DataDir + filename, "w") as filehandle:
            filehandle.write(json.dumps(index))
    
    # reading specified Index from Disk

    def ReadFromDisk(self, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.DataDir + filename, "r") as filehandle:
            index = json.loads(filehandle.read())

        return index


# fs = FeatureSpaceBuilder()