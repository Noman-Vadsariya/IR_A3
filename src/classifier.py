from msilib.schema import Class

from sympy import evaluate
from preprocessor import Preprocessor
from tfidf import TFIDF
from nouns_topics import TopicTerms
from lexicalChain import LexicalChain
from featureSelector import FeatureSpaceBuilder
from naiveBayes import NaiveBayes

from pathlib import Path
import os

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

        print(report)

        return (estimated_result,report)


    def predict_input(self,input):

        label = self.nb.predict_input(input)

        print(label)

        return label


c = Classifier()
c.train_model()
print(c.predict_test_data())

# input = "The information retrieval part deals with how to find useful information in large textual databases. This part of the course will cover inverted file systems, the vector space model (the SMART system), vector similarity, indexing, weighting, ranking, relevance feedback, phrase generation, term relationships and thesaurus construction, retrieval evaluation, and (if time permits) automatic text structuring and summarization."
# 
# input = " As part of our work on human-centered systems, we study (jointlywith cognitive scientists) human skills in motion planning and spaceorientation. These results are then used for comparison with theperformance of automatic systems and for developong hybrid physical(teleoperated) and computer graphics interaction systems. The majorproperty of such a hybrid system is that it blends together, in asynergistic manner, human and machine intelligences. Ourhardware/experimental work includes systems with massive real-timesensing and control (e.g. with thousands of sensors operating inparallel)."

# input = "My goal in general is to build software systems that improvecommunication among people.  I believe that communication mediums ofthe future will have an increasing understanding of the structure andcontent of the messages they transmit.  They will manipulate,reformat, and even generate that content.  I am interested inhypertext systems, network information access, and collaboration."


# c.predict_input(input)