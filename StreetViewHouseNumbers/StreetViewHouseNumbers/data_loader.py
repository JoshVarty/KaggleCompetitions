import sys
import pickle

def savePickle(object, filePath):
    with open(filePath, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def openPickle(filepath): 
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None
