import sys
import numpy as np
import sklearn
import nltk
import io
import string

from sklearn.naive_bayes import GaussianNB
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

class DataPrep:
    X = [[]]
    Y = []

    #Preps data
    def __init__(self, fn1, fn2):
        self.VersionCheck()
        orderedWordBag = self.ReadAndParse(fn1, fn2)
        self.X = self.CreateXVectors(fn1, fn2, orderedWordBag)
        self.Y = self.CreateYVector()

    def VersionCheck(self):
        if sys.version_info[0] < 3:
            raise Exception("Python 3 or a more recent version is required.")
        else:
            print ("Using Python" + str(sys.version_info[0]))

    #Cleans word by lemmatizing or stemming#
    def CleanWord(self, word):
        lemmatizer = WordNetLemmatizer() 
        lemmatizer.lemmatize(word)
        return word

    #Takes two text input and returns an ordered list of words with no repeats or stopwords#
    def ReadAndParse(self, fn1, fn2):
        print("......building the bag of words")
        wordBag = set()
        translator = str.maketrans('', '', string.punctuation)
        stop_words = set(stopwords.words('english'))

        for line in io.open(fn1, encoding='latin-1'):
            wordsInLine = line.translate(translator).split()
            for lineWord in wordsInLine:
                if lineWord not in stop_words:
                    wordBag.add(self.CleanWord(lineWord))
        for line in io.open(fn2, encoding='latin-1'):
            wordsInLine = line.translate(translator).split()
            for lineWord in wordsInLine:
                if lineWord not in stop_words:
                    wordBag.add(self.CleanWord(lineWord))

        orderedWordBag = list()

        for bagWord in wordBag:
            orderedWordBag.append(bagWord)
        orderedWordBag.sort()

        return orderedWordBag

    #Vectorizes X by word count
    def CreateXVectors(self, fn1, fn2, wordBag):
        print("......building X matrix")
        translator = str.maketrans('', '', string.punctuation)
        vectors = np.zeros((5331+5331, len(wordBag)), dtype=int)
        lineNum = 0

        for line in io.open(fn1, encoding='latin-1'):
            if lineNum % 1066 == 0:
                print("......"+str(int(lineNum/1066))+"0 percent complete")
            wordsInLine = line.translate(translator).split()
            wordNum = 0
            for bagWord in wordBag:
                for lineWord in wordsInLine:
                    if bagWord == lineWord:
                        vectors[lineNum][wordNum] += 1
                wordNum += 1
            lineNum += 1
        for line in io.open(fn2, encoding='latin-1'):
            if lineNum % 1066 == 0:
                print("......"+str(int(lineNum/1066))+"0 percent complete")
            wordsInLine = line.translate(translator).split()
            wordNum = 0
            for bagWord in wordBag:
                for lineWord in wordsInLine:
                    if bagWord == lineWord:
                        vectors[lineNum][wordNum] += 1
                wordNum += 1
            lineNum += 1

        return vectors

    #Vectorizes Y with 0 being neg and 1 being pos
    def CreateYVector(self):
        print("......building Y matrix")
        vector = np.zeros(5331+5331, dtype=int)
        count = 0

        for entry in vector:
            if count > 5330:
                vector[count] = 1
            count += 1

        return vector


data = DataPrep("rt-polaritydata/rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polaritydata/rt-polarity.pos")
#print(data.X)
#print(data.Y)
print("......spliting")
X_train, X_test, y_train, y_test = train_test_split(data.X, data.Y, test_size=0.33)
clf = GaussianNB(var_smoothing=.0001)
print("......training")
clf.fit(X_train, y_train)
print("Accuracy:  ")
print(clf.score(X_test,y_test))