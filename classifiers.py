import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss


class xgboostClassifier():
    def __init__(self, **params):
        self.clf = None
        self.progress = {}
        self.params = params

    def fit(self, X, y):
        xg_train = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(self.params, xg_train, self.params['num_rounds'])

    def fit_CV(self, X_train, X_val, y_train, y_val):
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_val = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(xg_train, 'train'), (xg_val, 'eval')]
        self.clf = xgb.train(self.params, xg_train, self.params['num_rounds'],
                         watchlist, early_stopping_rounds=200, evals_result=self.progress)

    def get_eval_res(self):
        return self.progress

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / log_loss(y, Y)

    def predict_proba(self, X_test):
        res = self.clf.predict(xgb.DMatrix(X_test))
        return res.astype(np.float32)

    def predict(self, X_test):
        res = np.argmax(self.clf.predict(xgb.DMatrix(X_test)), axis=1)
        return res 

    def get_params(self, **params):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

    def getSortedImportance(self, features):
        with open('xgb.fmap', 'w') as f:
            for i in range(len(features)):
                f.write('{0}\t{1}\tq\n'.format(i, features[i]))
        importance = self.clf.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        #print(importance)
        return importance

class BaseClassifier(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def set_params(self, **params):
        self.params.update(params)
    