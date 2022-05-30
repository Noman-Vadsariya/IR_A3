import nltk
from nltk.corpus import wordnet
from pathlib import Path
import json

"""
 3rd Feature Selection Strategy => Building Lexical Chains for each document

 - Used wordnet.synsets to get relations for a words and compare words using Wu & Palmer similarity

"""

class LexicalChain:

    def __init__(self):

        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
        self.documents = self.ReadFromDisk("train_data")
        self.chains = {}
        tokens = []

        self.traverseDocs()

        self.WriteToDisk(self.chains,"LexicalChain")

        print("\nLexical Chain Generated => Saved in LexicalChain.txt")


    def getRelations(self,tokens):
        
        relations = {}

        for tok in tokens:

            r = []
            
            for syn in wordnet.synsets(tok):
            
                for l in syn.hyponyms():
                    if l.hyponyms():
                        r.append(l.hyponyms()[0].name().split('.')[0])
            
                for l in syn.hypernyms():
                    if l.hypernyms():
                        r.append(l.hypernyms()[0].name().split('.')[0])
            
                for l in syn.lemmas():
                    r.append(l.name())

            relations[tok] = r

        return relations
        

    def constructLexicalChains(self,tokens,relations):

        threshold = 0.5
        chains = []

        for tok in tokens:

            flag = 0

            if len(chains) > 0:                     # for the first time no chain present

                for chain in chains:

                    for key in chain.keys():        # for comparing each token with each member of a chain

                        if tok == key:

                            chain[key] += 1
                            flag = 1                # freqeuncy count of token incremented
                            break

                        elif tok in relations[key] or key in relations[tok]:

                            synTok = wordnet.synsets(tok)
                            synKey = wordnet.synsets(key)
                            
                            if len(synKey) != 0 and len(synTok) != 0:

                                if synTok[0].wup_similarity(synKey[0]) >= threshold:

                                    chain[tok] = 1
                                    flag = 1            # token added to chain
                                    break

                    
                    if flag == 1:
                        break 

            # Initialize a new chain with the token if token not added to any existing chains 
            if flag == 0:
                
                newChain = { tok : 1 }
                chains.append(newChain)

        return chains


    # delete from the list the chains that only have one element or that word is only repeated once.
    def discard_irrelevant_chains(self,vocab):

        chain = []

        for temp in vocab:
        
            if len(temp.keys()) <= 1:
        
                for val in temp.values():
                    
                    if val != 1: 
                        chain.append(temp)
    
            else:
                chain.append(temp)

        return chain 


    def traverseDocs(self):
        
        docNo = [*self.documents.keys()]

        print("\n Generating Lexical Chains")
        
        #Build lexical Chains for all docs in the training set
        for no in docNo:
            
            relation = self.getRelations(self.documents[no])

            lexical = self.constructLexicalChains(self.documents[no],relation)
            
            chain = self.discard_irrelevant_chains(lexical)

            keys = list(set().union(*(d.keys() for d in chain)))

            self.chains[no] = keys


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