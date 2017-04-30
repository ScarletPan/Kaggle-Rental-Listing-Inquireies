import sys
import time
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from preprocess import coreProcess
from classifiers import xgboostClassifier

TRAIN_FILE_NAME = '~/Kaggle/RLI/input/train.json'
TEST_FILE_NAME = '~/Kaggle/RLI/input/test.json'
target_num_map = {'high': 0, 'medium': 1, 'low': 2}
train_data = pd.read_json(TRAIN_FILE_NAME).reset_index()
test_data = pd.read_json(TEST_FILE_NAME).reset_index()
list_img_time = pd.read_csv("~/Kaggle/RLI/input/listing_image_time.csv")
train_data = train_data.merge(list_img_time, left_on="listing_id", right_on="Listing_Id", how='inner')
test_data = test_data.merge(list_img_time, left_on="listing_id", right_on="Listing_Id", how='inner')
RS = 2016
random.seed(RS)
np.random.seed(RS)
# RS = 0

def validation_score(early_stop=False):
    clf = xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 3,
        eta = 0.04,
        max_depth = 6,
        subsample = 0.7,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.7,
        min_child_weight=1,
        silent = 1,
        num_rounds = 700,
        seed = RS,
    )
    print("*** Validation start ***")
    data = train_data.copy()
    y = data["interest_level"].apply(lambda x: target_num_map[x])
    del data["interest_level"]

    # skf = StratifiedKFold(n_splits=5, random_state=RS, shuffle=True)
    skf = StratifiedKFold(n_splits=3, shuffle=False)
    cv_scores = []
    i = 0
    for train_idx, val_idx in skf.split(data, y):
        i += 1
        X = data.copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train, X_val, feats = coreProcess(X, y_train, train_idx, val_idx)
        clf.fit(X_train, y_train)
        # clf.fit_CV(X_train, X_val, y_train, y_val)
        y_val_pred = clf.predict_proba(X_val)
        loss = log_loss(y_val, y_val_pred)
        print("Iteration {}'s loss: {}".format(i, loss))
        cv_scores.append(loss)
        if early_stop:
            break
    print("*** Validation finished ***\n")
    return cv_scores


def validation_avg_score(clfs):
    print("*** Validation start ***")
    data = train_data.copy()
    y = data["interest_level"].apply(lambda x: target_num_map[x])
    del data["interest_level"]

    # skf = StratifiedKFold(n_splits=5, random_state=RS, shuffle=True)
    skf = StratifiedKFold(n_splits=3)
    cv_scores = {i:[] for i in range(len(clfs))}
    cv_scores["Avg"] = []
    i = 0
    for train_idx, val_idx in skf.split(data, y):
        i += 1
        X = data.copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train, X_val, feats = coreProcess(X, y_train, train_idx, val_idx)
        tmp = []
        preds = []
        j = 0
        for clf in clfs:
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict_proba(X_val)
            tmp.append(y_val_pred)
            loss = log_loss(y_val, y_val_pred)
            cv_scores[j].append(loss)
            preds.append(y_val_pred)
            j += 1
            print("clf_{}, Iteration {}'s loss: {}".format(j, i, loss))
        preds = np.array(preds)
        avg_pred = np.mean(preds, axis=0)
        loss = log_loss(y_val, avg_pred)
        cv_scores["Avg"].append(loss)
        print("Iteration {}'s Avg loss: {}".format(i, loss))
    for i in range(len(clfs)):
        print("clf_{} validation loss : {}".format(i, np.mean(cv_scores[i])))
    print("Average validation loss : {}".format(np.mean(cv_scores["Avg"])))
    print("*** Validation finished ***\n")
    return cv_scores["Avg"]


def paramSearch(clf, param_dict):

    def outer_join(left, right):
        if left == []:
            return right
        if right == []:
            return left
        res = []
        for i in left:
            for j in right:
                if isinstance(i, list):
                    tmp = i[:]
                    tmp.append(j)
                    res.append(tmp)
                else:
                    res.append([i, j])
        return res
    # Creating list of param_dict
    param_list = sorted(param_dict.items(), key=lambda x: x[0])
    param_keys = [ item[0] for item in param_list ]
    param_vals = [ item[1] for item in param_list ]
    all_vals = []
    for val in param_vals:
        all_vals = outer_join(all_vals, val)
    all_param_lists = []
    for vals in all_vals:
        all_param_lists.append(dict(zip(param_keys, vals)))
    # for item in all_param_lists:
    #     print(item)

    # Searching
    best_score = float('inf')
    best_params = None
    scores = []
    i = 0
    for params in all_param_lists:
        print("\n" + "-" * 70)
        for param_name in params.keys():
            print("{} : {}".format(param_name, params[param_name]))
        clf.set_params(**params)
        score = np.mean(validation_score(clf))
        if score < best_score:
            best_score = score
            best_params = params
        i += 1
        print("{} / {}, Done".format(i, len(all_param_lists)))
        print("Score: ", score)
        scores.append(score)
    print(scores)
    print("Best parameters:")
    for param_name in best_params.keys():
        print("{} : {}".format(param_name, best_params[param_name]))
    print("Score: ", best_score)


def gen_sub():
    train = train_data.copy()
    train_idx = [i for i in range(train.shape[0])]
    test = test_data.copy()
    test_idx = [i + train.shape[0] for i in range(test.shape[0])]
    y = train["interest_level"].apply(lambda x: target_num_map[x])
    del train["interest_level"]
    data = pd.concat([train, test]).reset_index()
    X_train, X_test, feats = coreProcess(data, y, train_idx, test_idx)
    xgb_clf = xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 12,
        eta = 0.02,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.8,
        min_child_weight=1,
        silent = 1,
        num_rounds = 1700,
        seed = RS,
    )
    print("Trainning:...")
    xgb_clf.fit(X_train, y)

    preds = xgb_clf.predict_proba(X_test)
    sub = pd.DataFrame(preds)
    # sub.columns = ["high", "medium", "low"]
    sub.columns = [ "high", "medium", "low"]
    sub["listing_id"] = test.listing_id.values
    sub.to_csv("submission.csv", index=False)


def genAvgSub(clfs):
    train = train_data.copy()
    train_idx = [i for i in range(train.shape[0])]
    test = test_data.copy()
    test_idx = [i + train.shape[0] for i in range(test.shape[0])]
    y = train["interest_level"].apply(lambda x: target_num_map[x])
    del train["interest_level"]
    data = pd.concat([train, test]).reset_index()
    X_train, X_test, feats = coreProcess(data, y, train_idx, test_idx)
    print("Trainning:...")
    preds = []
    for i in range(len(clfs)):
        print("Clf_{} fiting".format(i))
        clfs[i].fit(X_train, y)
        print("Clf_{} predicting".format(i))
        pred = clfs[i].predict_proba(X_test)
        preds.append(pred)
    sub = pd.DataFrame(np.mean(preds, axis=0))
    # sub.columns = ["high", "medium", "low"]
    sub.columns = [ "high", "medium", "low"]
    sub["listing_id"] = test.listing_id.values
    sub.to_csv("submission.csv", index=False)
    print("Train done.")


def validate(clfs):
    cv_scores = validation_avg_score(clfs)
    return cv_scores


def search():
    param_dict = {
        'eta' : [0.02],
        'max_depth' : [6],
        'subsample' : [0.8],
        'colsample_bylevel' : [0.7],
        'num_rounds' : [1400, 1500, 1600, 1650],
    }
    clf = xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 12,
        eta = 0.04,
        max_depth = 6,
        subsample = 0.7,
        colsample_bytree = 1.0,
        colsample_bylevel = 1.0,
        min_child_weight=1,
        silent = 1,
        num_rounds = 700,
        seed = RS,
    )
    paramSearch(clf, param_dict)


def write2file(cv_scores, val_desc=None):
    print("*" * 50)
    print("Cross validation loss: ", np.mean(cv_scores))
    with open("results.log", "a") as fp:
        fp.write(time.strftime("%m/%d/%Y %H:%M") + '\n')
        if(val_desc is not None):
            fp.write(val_desc + '\n')
        for score in cv_scores:
            fp.write(str(score) + " ")
        fp.write("\nCross Validation: {}\n".format(np.array(cv_scores).mean()))
        fp.write("*" * 50 + "\n")


def stacking(clfs):
    print("Stacking")
    train = train_data.copy()
    test = test_data.copy()
    y = train["interest_level"].apply(lambda x: target_num_map[x])
    del train["interest_level"]
    train_stackers = []
    for RS in [0, 1, 2, 64, 128, 256, 512, 1024, 2048, 4096]:
        skf = StratifiedKFold(n_splits=10, random_state=RS, shuffle=True)
        #Create Arrays for meta
        train_stacker = [[0.0 for s in range(3)]  for k in range (0,(train.shape[0]))]
        cv_scores = {i:[] for i in range(len(clfs))}
        cv_scores["Avg"] = []
        print("Begin 10-flod cross validation")
        cnt = 0
        for train_idx, val_idx in skf.split(train, y):
            cnt += 1
            X = train.copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            X_train, X_val, feats = coreProcess(X, y_train, train_idx, val_idx)
            X_train.toarray()
            preds = []
            k = 0
            for clf in clfs:
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict_proba(X_val)
                loss = log_loss(y_val, y_val_pred)
                preds.append(y_val_pred)
                cv_scores[k].append(loss)
                k += 1
                print("Clf_{} iteration {}'s loss: {}".format(k, cnt, loss))
            preds = np.array(preds)
            avg_pred = np.mean(preds, axis=0)
            avg_loss = log_loss(y_val, avg_pred)
            cv_scores["Avg"].append(avg_loss)
            print("Iteration {}'s Avg loss: {}".format(cnt, avg_loss))
            no = 0
            for real_idx in val_idx:
                for i in range(3):
                    train_stacker[real_idx][i] = avg_pred[no][i]
                no += 1
        for i in range(len(clfs)):
            print("clf_{} validation loss : {}".format(i, np.mean(cv_scores[i])))
        print("Average validation loss : {}".format(np.mean(cv_scores["Avg"])))
        train_stackers.append(train_stacker)
    train_stacker = np.mean(train_stackers, axis=0)
    print("*** Validation finished ***\n")

    test_stacker = [[0.0 for s in range(3)]   for k in range (0,(test.shape[0]))]
    train_idx = [i for i in range(train.shape[0])]
    test_idx = [i + train.shape[0] for i in range(test.shape[0])]
    data = pd.concat([train, test]).reset_index()
    X_train, X_test, feats = coreProcess(data, y, train_idx, test_idx)
    print(X_train.shape, len(train_stacker))
    print("Begin predicting")
    preds = []
    for i in range(len(clfs)):
        print("Clf_{} fiting".format(i))
        clfs[i].fit(X_train, y)
        print("Clf_{} predicting".format(i))
        pred = clfs[i].predict_proba(X_test)
        preds.append(pred)
    preds = np.mean(preds, axis=0)
    for pr in range (0, len(preds)):  
            for d in range (0,3):            
                test_stacker[pr][d]=(preds[pr][d])   
    print ("merging columns")   
    #stack xgboost predictions
    X_train = np.column_stack((X_train.toarray(),train_stacker))
    # stack id to test
    X_test = np.column_stack((X_test.toarray(),test_stacker))         
    # stack target to train
    X = np.column_stack((y,X_train))
    ids = test.listing_id.values
    X_test = np.column_stack((ids, X_test))
    np.savetxt("./train_stacknet.csv", X, delimiter=",", fmt='%.5f')
    np.savetxt("./test_stacknet.csv", X_test, delimiter=",", fmt='%.5f') 
    print("Write results...")
    output_file = "submission_{}.csv".format(np.mean(cv_scores["Avg"]))
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")   
    f.write("listing_id,high,medium,low\n")# the header   
    for g in range(0, len(test_stacker))  :
      f.write("%s" % (ids[g]))
      for prediction in test_stacker[g]:
         f.write(",%f" % (prediction))    
      f.write("\n")
    f.close()
    print("Done.")


if __name__ == "__main__":
    clfs = []
    # clfs.append(xgboostClassifier(
    #     objective = 'multi:softprob',
    #     eval_metric = 'mlogloss',
    #     num_class = 3,
    #     nthread = 6,
    #     eta = 0.04,
    #     max_depth = 6,
    #     subsample = 0.7,
    #     colsample_bytree = 1.0,
    #     colsample_bylevel = 0.7,
    #     min_child_weight=1,
    #     silent = 1,
    #     num_rounds = 700,
    #     seed = 0,
    # ))
    # clfs.append(xgboostClassifier(
    #     objective = 'multi:softprob',
    #     eval_metric = 'mlogloss',
    #     num_class = 3,
    #     nthread = 6,
    #     eta = 0.02,
    #     max_depth = 6,
    #     subsample = 0.8,
    #     colsample_bytree = 1.0,
    #     colsample_bylevel = 0.8,
    #     min_child_weight=1,
    #     silent = 1,
    #     num_rounds = 1700,
    #     seed = 0,
    # ))
    clfs.append(xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 9,
        eta = 0.02,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.7,
        min_child_weight=1,
        silent = 1,
        num_rounds = 1500,
        seed = 0,
    ))
    clfs.append(xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 9,
        eta = 0.02,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.8,
        min_child_weight=1,
        silent = 1,
        num_rounds = 1500,
        seed = 128,
    ))
    clfs.append(xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 9,
        eta = 0.02,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.8,
        min_child_weight=1,
        silent = 1,
        num_rounds = 1500,
        seed = 512,
    )) 
    clfs.append(xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 9,
        eta = 0.02,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.8,
        min_child_weight=1,
        silent = 1,
        num_rounds = 1500,
        seed = 1024,
    ))   
    clfs.append(xgboostClassifier(
        objective = 'multi:softprob',
        eval_metric = 'mlogloss',
        num_class = 3,
        nthread = 9,
        eta = 0.02,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 1.0,
        colsample_bylevel = 0.8,
        min_child_weight=1,
        silent = 1,
        num_rounds = 1500,
        seed = 2048,
    ))    
    if len(sys.argv) == 1:
        cv_scores = validate(clfs)
        write2file(cv_scores)
    elif len(sys.argv) == 2:
        if sys.argv[1] == '-v':
            cv_scores = validate(clfs)
            write2file(cv_scores)
        elif sys.argv[1] == '-g':
            gen_sub()
        elif sys.argv[1] == '-s':
            search()
        elif sys.argv[1] == '-ga':
            genAvgSub(clfs)
        elif sys.argv[1] == '-stack':
            stacking(clfs)
        elif sys.argv[1] == '-v3':
            cv_scores = validate(clfs)
            val_desc = sys.argv[2]
            write2file(cv_scores, val_desc)
    elif len(sys.argv) == 3:
        if sys.argv[1] == '-v':
            cv_scores = validate(clfs)
            val_desc = sys.argv[2]
            write2file(cv_scores, val_desc)
        elif sys.argv[1] == '-g':
            gen_sub()
        elif sys.argv[1] == '-v3':
            cv_scores = validation_score()
            val_desc = sys.argv[2]
            write2file(cv_scores, val_desc)






