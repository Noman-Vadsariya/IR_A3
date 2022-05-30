# IR_A3

## Dependency
    
### Backend Dependency

    pip install flask

### Frontend Dependency

    npm install react-router-dom@6

    npm install axios

    npm install --save @chakra-ui/react

    npm install --save @chakra-ui/icons
    
    import gensim

## Run

### Start Backend Server

    cd src
    
    backend.py

### Start Frontend Server

    cd frontend

    npm start


## Assignment Desciption

### src

    - preprocessor.py => implements preprocessing chain

    - tfidf.py => implements 1st feature selection startegy using TFIDF

    - nouns_topics.py => implements 2nd feature selection, extract top 50 nouns and topic sets

    - lexicalChain.py => implements 3rd feature selection, generating lexical chains using wordnet

    - featureSelector.py => combines features of all three feature selection strategy 

    - naiveBayes.py => implement multinomial naive bayes classifier

    - classifier.py => wrapper for feature selection, training, testing and prediction 

### data

    - dictionary.txt => document url to docid mapping

    - feature_set.txt => combine features set for training

    - kTopicSets.txt => features based of topic occurrences

    - LexicalChain.txt => lexical chains for all documents in the training set

    - train_data.txt, test_data.txt => training and testing datasets

    - tfidf_topKfeatures.txt => top 100 features based on tfidf

    - trained_model.txt => saving trained model (prior and conditional probabilities)

    - vocablary.txt => overall vocablary before feature selection
