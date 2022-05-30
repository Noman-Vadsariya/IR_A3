from nltk.stem import WordNetLemmatizer
import json
from pathlib import Path
import math
import heapq

"""
 1st Feature Selection Strategy => Top 100 Words based on TFIDF

 Builds and Stores Term frequency index, Inverse document index and tfidf index from Collection of Documents

"""

class TFIDF:

    def __init__(self, FolderName=None):

        self.tf_index = {}          # term frequency index
        self.idf_index = {}         # inverse document frequency index
        self.tfidf_index = {}       # tfidf index
        self.vocablary = {}
        self.features = {}
        self.topfeatures = {}

        self.noOfDocs = 0

        self.DataDir = str(Path(__file__).parent.resolve()).replace("src", "data")  # Folder to Store Indexes
        
        self.documents = self.ReadFromDisk('train_data')
        
        self.BuildTfIndex()
        self.length_normalization()
        self.BuildIdfIndex()
        self.BuildTfIdfIndex()
        self.topKFeatures()

        print("\nTop 100 Words Selected Based on TFIDF => Saved in tfidf_topKFeatures.txt")

    # calculates term frequency for each unique term in each document.
    # tf_index = {doc1 : { t1 : 3, t2: 4, ... ,tn: 5}, doc1 : { t1 : 2, t2: 1, ... ,tn: 4}, ... , docN : { t1 : 1, t2: 4, ... ,tn: 2} )   

    def BuildTfIndex(self):

        # files are not read from directory in sorted order, so reading files of a directory and sorting their names
        # reason - so that posting lists are sorted, sorted posting lists alllows intersection in linear time

        docNo = 0

        print(len(self.documents.keys()))
        
        for key in self.documents.keys():

            text_words = self.documents[key]

            for word in text_words:
                
                if not word.isdigit():
                    
                    if key not in self.tf_index.keys():
                        
                        self.tf_index[key] = {}               # adding term in a particular document

                    if word not in self.tf_index[key].keys():

                        self.tf_index[key][word] = 1          # initializing frequency count for a term
                    
                    else:

                        self.tf_index[key][word] += 1         # incrementing frequency count for a term


                    # Vocablary => term : term frequency

                    if word not in self.vocablary.keys():
                        self.vocablary[word] = 1
                    else:
                        self.vocablary[word] += 1

            docNo += 1

        self.noOfDocs = docNo

        self.WriteToDisk(self.vocablary,'vocablary')

        self.WriteToDisk(self.tf_index,'tf_index')


    # Euclidean Normalization Vector / Magnitude of Vector => V / || V ||
    
    def length_normalization(self):

        self.magnitude = {}                                         # each index stores magnitude for a particular document

        for i in self.tf_index.keys():
            
            self.magnitude[i] = 0

            for key in self.tf_index[i].keys():

                self.magnitude[i] += self.tf_index[i][key] ** 2

            self.magnitude[i] = math.sqrt(self.magnitude[i])        # sqrt(tf1^2 + tf2^2 + tf3^2 + ... + tfn^2)


    # calculates inverse document frequency  for each unique term
    # idf_index = { t1: idf-Val, t2: idf-Val , t3: idf-Val , ... , t4: idf-val }

    def BuildIdfIndex(self):
        df = {}

        for i in self.tf_index.keys():

            temp = []
            for key in self.tf_index[i].keys():

                if key not in temp:
                    if key not in df.keys():
                        df[key] = 1
                    else:
                        df[key] += 1

                    temp.append(key)

        # idf will calculated for each unique term
        for k in df.keys():
            self.idf_index[k] = math.log10(self.noOfDocs / df[k])  # idf = log(N/df)

        # self.WriteToDisk(self.idf_index,'idf_index')


    # calculates tf*idf  for each unique term
    # tfidf_index = {doc1 : { t1 : 0.21, t2: 2.4, ... ,tn: 0.11}, doc1 : { t1 : 2.4, t2: 0.01, ... ,tn: 0.234}, ... , docN : { t1 : 0.21, t2: 0.344, ... ,tn: 0.2})
    
    def BuildTfIdfIndex(self):

        for i in self.tf_index.keys():

            self.tfidf_index[i] = {}

            for key in self.tf_index[i].keys():

                tf = (self.tf_index[i][key] / self.magnitude[i])    # length normalizing term frequency vector 
                idf = self.idf_index[key]
                self.tfidf_index[i][key] = tf * idf                 # tfidf = tf * log(N/df)
    

    # Extract top 100 words after summing tfidf value for each unique vocablary term

    def topKFeatures(self, k=100):
        
        tfidf_sum = {}

        for key in self.vocablary.keys():
            
            tfidf_sum[key] = 0

            for i in self.tfidf_index.keys():
                
                if key in self.tfidf_index[i].keys():

                    tfidf_sum[key] += self.tfidf_index[i][key]

            
        self.topfeatures = heapq.nlargest(k, tfidf_sum, key=tfidf_sum.get)

        self.WriteToDisk(self.topfeatures,"tfidf_topKFeatures")


    # writing specified Index to Disk

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

