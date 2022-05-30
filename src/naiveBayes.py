from unittest import result
from black import out
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from preprocessor import Preprocessor

import os
import json
from pathlib import Path
import math

"""
 
 Implementation of Multinomial Naives Bayes Classifier

"""

class NaiveBayes:
    def __init__(self):

        self.DataDir = str(Path(__file__).parent.resolve()).replace("src", "data")  # Folder to Store Indexes

        self.dictionary = self.ReadFromDisk("dictionary")  # Load dictionary having docid to docURL mapping

        self.vocablary = self.ReadFromDisk("feature_set")  # Load Feature Space

        self.tf_index = self.ReadFromDisk("tf_index")  # Load Term Frequency Index for computing frequencies

        self.train_data = self.ReadFromDisk("train_data")  # Load Training Data

        self.freq = {}
        self.classFreq = {}

        self.classes = self.dictionary.keys()

    def load_model(self):

        if os.path.isdir(self.DataDir):  # if directory already not present

            if "trained_model.txt" in os.listdir(self.DataDir):

                # Load Already Trained Model

                model = self.ReadFromDisk("trained_model")

                self.prior = model["prior_prob"]
                self.condProb = model["cond_prob"]

                print("\nModel Loaded\n")

            else:

                self.CountFrequency()
                self.TrainMultinomialNB()

        else:

            print("\nNo Data Directory Exists")
            exit(0)

    # Calculated frequency of tokens for computing prior and conditional probabilities
    def CountFrequency(self):

        temp = 0

        for _class in self.classes:

            self.classFreq[_class] = 0

            self.freq[_class] = {}

            for tok in self.vocablary:

                self.freq[_class][tok] = 0

                Nc = len(self.dictionary[_class].keys())

                for i in range(temp, Nc + temp):

                    if (str(i) in self.tf_index.keys() and tok in self.tf_index[str(i)].keys()):

                        self.classFreq[_class] += self.tf_index[str(i)][tok]
                        self.freq[_class][tok] += self.tf_index[str(i)][tok]

            temp = Nc

    def TrainMultinomialNB(self):

        N = len(self.dictionary["course"].keys()) + len(self.dictionary["non-course"].keys())

        self.prior = {}
        self.condProb = {}

        print(f"\nTotal Docs (N) : {N}")

        for _class in self.classes:

            # computing Prior Probability of class => P(c) = Nc / N

            Nc = len(self.dictionary[_class].keys())
            print(f"{_class} Doc Count: {Nc}")
            self.prior[_class] = Nc / N

            # computing Conditional Probabilities of terms in a class => P(t|c) = (count(t,c) + 1) / ( count(c) + |v| )

            self.condProb[_class] = {}

            for tok in self.vocablary:
                self.condProb[_class][tok] = (self.freq[_class][tok] + 1) / (self.classFreq[_class] + len(self.vocablary))

        print("\nModel Trained\n")

        self.save_trained_model()

        print("\nModel Saved\n")

    # P(c|doc) = P(c) * (Multiply (for all test terms) P(t|c))

    def ApplyMultinomialNB(self, tokens):

        score = {}

        for _class in self.classes:

            score[_class] = math.log(self.prior[_class])

            for tok in tokens:

                if tok in self.condProb[_class].keys():
                    score[_class] += math.log(self.condProb[_class][tok])

        max_key = max(score, key=lambda x: score[x])

        return max_key

    def save_trained_model(self):

        model = {}

        model["prior_prob"] = self.prior
        model["cond_prob"] = self.condProb

        self.WriteToDisk(model, "trained_model")

    # Predict on all docs in testing set
    def predict_test_data(self):

        result_labels = {}

        self.test_data = self.ReadFromDisk("test_data")

        for docNo in self.test_data.keys():

            result_labels[docNo] = self.ApplyMultinomialNB(self.test_data[docNo])

        return result_labels

    # Predict on input string
    def predict_input(self, input):

        p = Preprocessor()

        tokens = p.PreprocessText(input)

        label = self.ApplyMultinomialNB(tokens)

        return label


    # Generate Classification report for the test data
    def EvaluationMetrics(self, result_labels):

        estimated_labels = []
        true_labels = []

        for key in self.test_data.keys():

            if key in self.dictionary["course"].keys():

                true_labels.append("course")

            else:

                true_labels.append("non-course")

        for key in result_labels.keys():

            estimated_labels.append(result_labels.get(key))

        report = classification_report(true_labels, estimated_labels, output_dict=True)

        for metric in report.keys():

            if metric != "accuracy":

                for key in report[metric].keys():

                    report[metric][key] = round(report[metric][key] * 100)
            else:

                report[metric] = round(report[metric] * 100)

        return report

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

