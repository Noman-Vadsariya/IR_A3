import nltk
from nltk.corpus import wordnet
from pathlib import Path
import json

class LexicalChain:

    def __init__(self):

        self.dataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
        self.documents = self.ReadFromDisk("train_data")
        self.chains = {}
        tokens = []

        self.POS_tagging()

        self.WriteToDisk(self.chains,"LexicalChain")


    def getRelations(self,tokens):

        # relations = defaultdict(list)
        
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

        self.WriteToDisk(relations,"relations")

        # print("hello")
        return relations
        

    # def constructLexicalChains(self,tokens,relation):
    #     vocab = []
        
    #     threshold = 0.3
        
    #     for tok in tokens:

    #         flag = 0
            
    #         for chain in vocab:

    #         # for j in range(len(vocab)):
            
    #             if flag == 0:
                    
    #                 keys = chain.keys()

    #                 for key in keys:
    #                 # for key in list(vocab[j]):
            
    #                     if key == tok and flag == 0:
    #                         chain[tok] += 1
    #                         flag = 1
            
    #                     elif key in relation[tok] and flag == 0:
    #                         syns1 = wordnet.synsets(key)
    #                         syns2 = wordnet.synsets(tok)
                            
    #                         if syns1[0].wup_similarity(syns2[0]) >= threshold:
    #                             chain[tok] = 1
    #                             flag = 1
            
    #                     elif tok in relation[key] and flag == 0:
    #                         syns1 = wordnet.synsets(key)
    #                         syns2 = wordnet.synsets(tok)
                            
    #                         if syns1[0].wup_similarity(syns2[0]) >= threshold:
    #                             chain[tok] = 1
    #                             flag = 1
            
    #         # Initialize a new chain with the token because it does not belong to any other present chain 
    #         if flag == 0: 
                
    #             dic = {}
    #             dic[tok] = 1
    #             vocab.append(dic)

    #             # print(vocab)

    #             flag = 1

    #     return vocab

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

                # print(chains)

        return chains


    # delete from the list the chains that only have one element and that word is only repeated once.
    def discardIrrelevant(self,vocab):

        chain = []

        while vocab:
        
            temp = vocab.pop()
        
            if len(temp.keys()) == 1:
        
                for val in temp.values():
                    if val != 1: 
                        chain.append(temp)
        
            else:
                chain.append(temp)

        return chain 


    def POS_tagging(self):
        
        docNo = [*self.documents.keys()]

        for no in docNo:

            # tokens.clear()

            # words = self.documents[no]
            # tagged = nltk.pos_tag(words)

            # for (tok, tag) in tagged:
            #     if (tag == "NNP" or tag == "NN" or tag == "NNS" or tag == "NNPS") and (tok not in tokens):
            #         tokens.append(tok)
            
            # print(tokens)
            # print(no)
            relation = self.getRelations(self.documents[no])
            lexical = self.constructLexicalChains(self.documents[no],relation)
            chain = self.discardIrrelevant(lexical)

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


# l = LexicalChain()