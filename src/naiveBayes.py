from unittest import result
from black import out
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report

import re
import os
import json
from pathlib import Path
import math


class NaiveBayes:
    def __init__(self):

        self.DataDir = str(Path(__file__).parent.resolve()).replace("src", "data")  # Folder to Store Indexes

        self.dictionary = self.ReadFromDisk("dictionary")

        self.vocablary = self.ReadFromDisk("feature_set")

        self.tf_index = self.ReadFromDisk("tf_index")

        self.train_data = self.ReadFromDisk("train_data")

        # print(len(self.vocablary))

        self.freq = {}
        self.classFreq = {}

        self.classes = self.dictionary.keys()


    def load_model(self):

        if os.path.isdir(self.DataDir):  # if directory already not present

            print(os.listdir(self.DataDir))

            if "trained_model.txt" in os.listdir(self.DataDir):

                model = self.ReadFromDisk("trained_model")

                self.prior = model["prior_prob"]
                self.condProb = model["cond_prob"]

                print("Model Loaded\n")

            else:
                
                self.CountFrequency()
                self.TrainMultinomialNB()

        else:

            print("No Data Directory Exists")
            exit(0)


    def loadStopwords(self):
        self.stop_words = set(stopwords.words("english"))

    def tokenize(self, text):

        text = text.lower()  # case folding
        text = re.sub(r"-", " ", text)  # handling hyphen
        text = re.sub(
            r"[^\w\s]", " ", text
        )  # noise removal - replacing all types of [^a-zA-Z0-9] and [^\t\r\n\f] with space for splitting on space
        text = text.split()  # splitting on space
        return text

    # Lemmatization - WordNetLemmatizer used
    def Lemmatization(self, token):
        l = WordNetLemmatizer()
        return l.lemmatize(token)

    def remove_number_stopwords(self, tokens):

        filtered_sentence = []

        for w in tokens:
            if (w not in self.stop_words) and (not w.isnumeric()) and (len(w)!=1):
                filtered_sentence.append(self.Lemmatization(w))

        return filtered_sentence


    def PreprocessText(self, text):
        self.loadStopwords()
        tokens = self.tokenize(text)
        tokens = self.remove_number_stopwords(tokens)
        return tokens


    def CountFrequency(self):

        temp = 0

        for _class in self.classes:

            self.classFreq[_class] = 0

            self.freq[_class] = {}

            for tok in self.vocablary:

                self.freq[_class][tok] = 0

                Nc = len(self.dictionary[_class].keys())

                for i in range(temp, Nc + temp):
                    
                    if str(i) in self.tf_index.keys() and tok in self.tf_index[str(i)].keys():

                        self.classFreq[_class] += self.tf_index[str(i)][tok]
                        self.freq[_class][tok] += self.tf_index[str(i)][tok]

                
            temp = Nc
            print(i)

        print(self.classFreq)

        # self.WriteToDisk(self.freq, "frequency")


    def TrainMultinomialNB(self):

        N = len(self.dictionary["course"].keys()) + len(self.dictionary["non-course"].keys())

        self.prior = {}
        self.condProb = {}

        print(f"Total Docs (N) : {N}")

        for _class in self.classes:

            Nc = len(self.dictionary[_class].keys())
            print(f"{_class} Doc Count: {Nc}")
            self.prior[_class] = Nc / N

            self.condProb[_class] = {}

            for tok in self.vocablary:
                self.condProb[_class][tok] = (self.freq[_class][tok] + 1) / (self.classFreq[_class] + len(self.vocablary))

        print("Model Trained\n")

        self.save_trained_model()

        print("Model Saved\n")


    def ApplyMultinomialNB(self, tokens):

        score = {}

        for _class in self.classes:

            score[_class] = math.log(self.prior[_class])

            for tok in tokens:

                if tok in self.condProb[_class].keys():
                    score[_class] += math.log(self.condProb[_class][tok])

        Keymax = max(score, key=lambda x: score[x])

        return Keymax


    def save_trained_model(self):

        model = {}

        model["prior_prob"] = self.prior
        model["cond_prob"] = self.condProb

        self.WriteToDisk(model, "trained_model")


    def predict_test_data(self):
        
        result_labels = {}

        self.test_data = self.ReadFromDisk("test_data")

        for docNo in self.test_data.keys():

            result_labels[docNo] = self.ApplyMultinomialNB(self.test_data[docNo])

        return result_labels


    def predict_input(self,input):

        tokens = self.PreprocessText(input)

        label = self.ApplyMultinomialNB(tokens)

        return label


    def EvaluationMetrics(self, result_labels):

        estimated_labels = []
        true_labels = []

        for key in self.test_data.keys():

            if key in self.dictionary['course'].keys():

                true_labels.append('course')

            else:

                true_labels.append('non-course')
        

        for key in result_labels.keys():

            estimated_labels.append(result_labels.get(key))  

        report = classification_report(true_labels,estimated_labels,output_dict=True)

        for metric in report.keys():
            
            if metric != 'accuracy':

                for key in report[metric].keys():
                    
                    report[metric][key] = round(report[metric][key] * 100) 
            else:

                report[metric] = round(report[metric] * 100)

        return report


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





            



# nb = NaiveBayes()

# input = "The information retrieval part deals with how to find useful information in large textual databases. This part of the course will cover inverted file systems, the vector space model (the SMART system), vector similarity, indexing, weighting, ranking, relevance feedback, phrase generation, term relationships and thesaurus construction, retrieval evaluation, and (if time permits) automatic text structuring and summarization."

# input = " As part of our work on human-centered systems, we study (jointlywith cognitive scientists) human skills in motion planning and spaceorientation. These results are then used for comparison with theperformance of automatic systems and for developong hybrid physical(teleoperated) and computer graphics interaction systems. The majorproperty of such a hybrid system is that it blends together, in asynergistic manner, human and machine intelligences. Ourhardware/experimental work includes systems with massive real-timesensing and control (e.g. with thousands of sensors operating inparallel)."

# input = "My goal in general is to build software systems that improvecommunication among people.  I believe that communication mediums ofthe future will have an increasing understanding of the structure andcontent of the messages they transmit.  They will manipulate,reformat, and even generate that content.  I am interested inhypertext systems, network information access, and collaboration."

# print(nb.ApplyMultinomialNB(input))

# estimated_labels = nb.predict_test_data()

# nb.EvaluationMetrics(estimated_labels)

# TRAINMULTINOMIALNB(C,D)
# 1 V ← EXTRACTVOCABULARY(D)
# 2 N ← COUNTDOCS(D)
# 3 for each c ∈ C
# 4 do Nc ← COUNTDOCSINCLASS(D, c)
# 5 prior[c] ← Nc/N
# 6 textc ← CONCATENATETEXTOFALLDOCSINCLASS(D, c)
# 7 for each t ∈ V
# 8 do Tct ← COUNTTOKENSOFTERM(textc, t)
# 9 for each t ∈ V
# 10 do condprob[t][c] ← Tct+1
# åt′ (Tct′+1)
# 11 return V, prior, condprob

# APPLYMULTINOMIALNB(C,V, prior, condprob, d)
# 1 W ← EXTRACTTOKENSFROMDOC(V, d)
# 2 for each c ∈ C
# 3 do score[c] ← log prior[c]
# 4 for each t ∈ W
# 5 do score[c] += log condprob[t][c]
# 6 return argmaxc∈C score[c]
