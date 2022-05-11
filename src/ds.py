from bs4 import BeautifulSoup
import os
from pathlib import Path
import json

class Scrapper:

    def __init__(self):
        self.data = []
        self.dictionary = {}
        self.documents = {}
        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")

    def getFileObj(self):

        filePath = self.dataDir + '\\features.txt'

        if not os.path.isdir(self.dataDir):
            os.mkdir(self.dataDir)

        if not os.path.isfile(filePath):
            fileObj = open(filePath,'w')

            if fileObj:
                return fileObj
            else:
                print("Error in creating file")

        else:
            fileObj = open(filePath,'a')

            if fileObj:
                return fileObj
            else:
                print("Error in opening file")
    
    def Scrape(self,folderNames):

        collectionDir = str(Path(__file__).parent.resolve()).replace("src", "course-cotrain-data")

        print(collectionDir)
        i = 0

        for folder in folderNames:

            temp = collectionDir + '\\' + folder

            print(temp)

            for filename in os.listdir(temp):
            
                if filename.endswith('.html') or filename.endswith('.edu'):
                    
                    fname = os.path.join(temp, filename)
                    # print("Current file name ..", os.path.abspath(fname))
                    
                    with open(fname, 'r') as file:
                        
                        contents = file.read()

                        soup = BeautifulSoup(contents, 'lxml')

                        for script in soup(["script", "style"]):
                            script.decompose()

                        text = list(soup.stripped_strings)
                        text = ' '.join(text)
                        
                        self.dictionary[i] = filename
                        self.documents[i] = text
                
                i+=1


        # print(self.documents)
        self.WriteToDisk(self.dictionary,'dictionary')
        self.WriteToDisk(self.documents,'documents')

        # writing specified index to disk

    def WriteToDisk(self, index, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.dataDir + filename, "w") as filehandle:
            filehandle.write(json.dumps(index))

s = Scrapper()

# s.Scrape(['fulltext\\course'])
s.Scrape(['fulltext\\course','fulltext\\non-course','inlinks\\course','inlinks\\non-course'])
# s.Scrape('fulltext\\non-course')
# s.Scrape('inlinks\\non-course')
