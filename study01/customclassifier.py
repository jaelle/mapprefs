from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import random 

class CustomClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, 
                    randomSeed=42,
                    alignmentWeight=0.5, 
                    distanceWeight=0.5, 
                    alignmentCols = ['topleft','left','bottomleft'], 
                    distanceCols = ['close','med','far'], 
                    placements = [['topleft-med','left-far','bottomleft-close'], ['topleft-far','left-close','bottomleft-med']],
                    predictRanking = False,
                    stringParam="defaultValue", 
                    otherParam=None):
        """
        Called when initializing the classifier
        """
        self.alignmentWeight = alignmentWeight
        self.distanceWeight = distanceWeight
        self.alignmentCols = alignmentCols 
        self.distanceCols = distanceCols
        self.placements = placements
        self.predictRanking = predictRanking
        self.ranks_ = []

        self.stringParam = stringParam
        self.otherParam = otherParam

        random.seed(randomSeed)

    def fit(self, X, y=None):
        
        return self.rankOptions(X,y)

    def predict(self, X, y=None):

        return self.rankOptions(X,y)

    def rankOptions(self, X, y=None):
        if "qtype" in X.columns and all(col in self.alignmentCols for col in X.columns) and all(col in self.distanceCols for col in X.columns):
            print("Error: X must contain qtype column and the alignment and distance columns specified")
            print("Alignment:")
            print(self.alignmentCols)
            print("Distance:")
            print(self.distanceCols)

        # get preferences by question type
        q = [0,0]
        alignmentVals = [0,0]
        distanceVals = [0,0]
        prefVals = [0,0]

        allPlacements = []

        for i in [0,1]:
            q[i] = X[X['qtype'] == i]

            alignmentVals[i] = self.alignmentWeight * q[i][self.alignmentCols]
            distanceVals[i] = self.distanceWeight * q[i][self.distanceCols]

            prefVals[i] = {}
            for placement in self.placements[i]:
                parts = placement.split('-')
                alignment = parts[0]
                distance = parts[1]
                prefVals[i][placement] = alignmentVals[i][alignment] + distanceVals[i][distance]
                allPlacements += [placement]
        
        pd_prefVals = pd.concat([pd.DataFrame(prefVals[0]),pd.DataFrame(prefVals[1])]).sort_index()

        if not self.predictRanking:
            pd_prefVals['max'] = pd_prefVals[allPlacements].max(axis=1)
            pd_prefVals['possibleResponses'] = pd_prefVals.apply(lambda p: [col for col in allPlacements if p[col] == p['max']],axis=1)

            return pd_prefVals['possibleResponses'].apply(lambda p: random.choice(p))
        else:

            pd_prefVals['possibleResponses'] = pd_prefVals[allPlacements].apply(lambda p: self.getRanks(p),axis=1)
            return pd_prefVals['possibleResponses']

    def getRanks(self,placements):

        # get top 3 values and indices
        ranks = {}
        count = 0

        while count < 3:
            maxId = placements.idxmax()
            maxValue = placements.max()
            if maxValue not in ranks.keys():
                ranks[maxValue] = [maxId]
            else:
                ranks[maxValue] += [maxId]

            placements = placements.drop(maxId)
            count += 1

        self.ranks_ += [ranks]
        
        ranked = ''
        for k,val in ranks.items():
            random.shuffle(val)
            for v in val:
                ranked += v + ','
        
        ranked = ranked[:-1]

        return ranked

    def score(self, X, y):
        # counts number of X that are the same as y
        if not self.predictRanking:
            correct = 0
            self.y_predict_ = self.predict(X)
            for id,yp in self.y_predict_.iteritems():
                if yp == y[id]:
                    correct += 1
        else:
            correct = 0
            self.rankscores_ = [0,0,0]
            self.y_predict_ = self.predict(X)


            for id,yp in self.y_predict_.iteritems():
                yp_arr = yp.split(',')
                y_arr = y[id].split(',')
                for i in range(3):
                    if yp_arr[i] == y_arr[i]:
                        correct += 1    
                        self.rankscores_[i] += 1

        return correct