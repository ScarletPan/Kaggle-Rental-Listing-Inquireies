import pandas as pd
import numpy as np


def getAvgSub(subs_in):
    subs = []
    for sub in subs_in:
        sub = sub.sort_values(by=["listing_id"]).reset_index()
        subs.append(sub)
    n = len(subs)
    new_sub = subs[0].copy()
    for i in range(1, n):
        sub = subs[i]
        new_sub["high"] = new_sub["high"] + sub["high"]
        new_sub["medium"] = new_sub["medium"] + sub["medium"]
        new_sub["low"] = new_sub["low"] + sub["low"]
    new_sub["high"] =  new_sub["high"] / n
    new_sub["medium"] = new_sub["medium"] / n
    new_sub["low"] = new_sub["low"] / n
    del new_sub["index"]
    return new_sub

def getWeightedAvgSub(subs_in, weights):
    assert np.sum(weights) == 1, "Sum of weights need to be 1"
    subs = []
    for sub in subs_in:
        sub = sub.sort_values(by=["listing_id"]).reset_index()
        subs.append(sub)
    n = len(subs)
    new_sub = subs[0].copy() 
    new_sub["high"] = new_sub["high"] * weights[0]
    new_sub["medium"] = new_sub["medium"] * weights[0]
    new_sub["low"] = new_sub["low"] * weights[0]
    for i in range(1, n):
        sub = subs[i]
        new_sub["high"] = new_sub["high"] + sub["high"] * weights[i]
        new_sub["medium"] = new_sub["medium"] + sub["medium"] * weights[i]
        new_sub["low"] = new_sub["low"] + sub["low"] * weights[i]
    del new_sub["index"]
    return new_sub

def generateStackSub(test_file_name, sub_file_name):
    test_array = np.loadtxt(test_file_name, delimiter=",") 
    test = pd.DataFrame(test_array)
    sub_array = np.loadtxt(sub_file_name, delimiter=",") 
    sub = pd.DataFrame(sub_array)
    sub.columns = ["high", "medium", "low"]
    sub["listing_id"] = test.iloc[:, 0].apply(lambda x: int(x))
    sub.to_csv("new_sub.csv", index=False)   




def correct(df):
    interest_levels = ['low', 'medium', 'high']

    tau = {
        'low': 0.69195995, 
        'medium': 0.23108864,
        'high': 0.07695141, 
    }

    y = df[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print(a)

    def f(p):
        for k in range(len(interest_levels)):
            p[k] *= a[k]
        return p / p.sum()

    df_correct = df.copy()
    df_correct[interest_levels] = df_correct[interest_levels].apply(f, axis=1)

    y = df_correct[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print(a)

    return df_correct