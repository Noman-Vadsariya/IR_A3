from preprocessor import Preprocessor
from tfidf import TFIDF
from nouns_topics import TopicTerms
from lexicalChain import LexicalChain
from featureSelector import FeatureSpaceBuilder
from naiveBayes import NaiveBayes
from pathlib import Path
import os

"""

- Wrapper Class for Feature Selection , Model Training , Testing and Input Prediction

"""


class Classifier:

    def __init__(self):
        
        self.DataDir = str(Path(__file__).parent.resolve()).replace("src", "data")


    def load_data(self):

        if not os.path.isdir(self.DataDir):  # if directory already not present
            
            os.mkdir(self.DataDir)

            p = Preprocessor()
            p.Scrape(["fulltext\\course", "fulltext\\non-course"])

            p.split_train_test()
            
            t = TFIDF()

            tt = TopicTerms()

            l = LexicalChain()

            fs = FeatureSpaceBuilder()

        else:

            print("Data Directory Exists")


    def train_model(self):

        self.load_data()
        self.nb = NaiveBayes()
        self.nb.load_model()


    def predict_test_data(self):

        estimated_result = self.nb.predict_test_data()
        report = self.nb.EvaluationMetrics(estimated_result)

        print("Classification Report", end="\n\n")
        print(report)

        return (estimated_result,report)


    def predict_input(self,input):

        label = self.nb.predict_input(input)

        print(label)

        return label
