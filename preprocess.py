#!/usr/bin/python3
#-*- encoding: utf-8 -*-
import sys
import random
import operator
import datetime
import time
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from nltk.metrics import distance as distance


FEATURE_NOT_USE = ['created','description','features','photos', 'index']# ,'bathrooms', 'bedrooms''listing_id',
FEATURE_NOT_USE.append('display_address')
FEATURE_NOT_USE.extend(['low_build_frac', 'high_build_frac', 'medium_build_frac', 'build_count'])# 
FEATURE_NOT_USE.extend(['low_manager_frac', 'high_manager_frac', 'medium_manager_frac','manager_count'])#
FEATURE_NOT_USE.extend(['Listing_Id', 'img_created']) # , 'time_stamp'

def bedroomProcess(data, train_idx, test_idx):
    # Some basic feature from bedrooms
    data["no_bedroom"] = data["bedrooms"].apply(lambda x: 1 if x == 0 else 0)
    data["more_than_5_bedroom"] = data["bedrooms"].apply(lambda x: 1 if x >= 5 else 0)
    data.loc[data["bedrooms"] + data["bathrooms"] == 0, "bedrooms"] = 0.001
    train = data.iloc[train_idx, :].copy()
    test = data.iloc[test_idx, :].copy()
    # remove null value (ugly code)
    train.loc[data["bedrooms"] == 0.001, "bathrooms"] = train["bathrooms"].mean()
    test.loc[data["bedrooms"] == 0.001, "bathrooms"] = test["bathrooms"].mean()
    data.iloc[train_idx, :] = train
    data.iloc[test_idx, :] = test
    data["bedroom_per_room"] = data["bedrooms"] / (data["bedrooms"] + data["bathrooms"])
    data.loc[data["bedrooms"] == 0.001, "bathrooms"] = 0
    data.loc[data["bedrooms"] == 0.001, "bedrooms"] = 0
    return data


def bathroomProcess(data, train_idx, test_idx):
    # Some basic feature from bathrooms
    data.loc[data["bathrooms"] == 112, "bathrooms"] = 1.5
    data.loc[data["bathrooms"] == 10, "bathrooms"] = 1
    data.loc[data["bathrooms"] == 20, "bathrooms"] = 2
    data["1_to_2_bathrooms"] = data["bathrooms"].apply(lambda x : 1if x != 0 and x <= 2 else 0)
    data.loc[data["bedrooms"] + data["bathrooms"] == 0, "bathrooms"] = 0.001
    train = data.iloc[train_idx, :].copy()
    test = data.iloc[test_idx, :].copy()
    # remove null value (ugly code)
    train.loc[data["bathrooms"] == 0.001, "bedrooms"] = train["bedrooms"].mean()
    test.loc[data["bathrooms"] == 0.001, "bedrooms"] = test["bedrooms"].mean()
    data.iloc[train_idx, :] = train
    data.iloc[test_idx, :] = test
    data["bathoom_per_room"] = data["bathrooms"] / (data["bedrooms"] + data["bathrooms"])
    data.loc[data["bathrooms"] == 0.001, "bedrooms"] = 0
    data.loc[data["bathrooms"] == 0.001, "bathrooms"] = 0
    return data


def buildingIdProcess(data, y, train_idx, test_idx):
    # Have tried some ideas but failed
    return data


def createdProcess(data):
    # Some basic features from created
    data["created"] = pd.to_datetime(data['created'])
    data["latest"] = (data["created"]- data["created"].min())
    data["latest"] = data["latest"].apply(lambda x: x.total_seconds())
    data["passed"] = (data["created"].max()- data["created"])
    data["passed"] = data["passed"].apply(lambda x: x.total_seconds())
    # year is weird
    data["year"] = data["created"].dt.year
    data['month'] = data['created'].dt.month
    data['day'] = data['created'].dt.day
    data['hour'] = data['created'].dt.hour
    data['weekday'] = data['created'].dt.weekday
    data['week'] = data['created'].dt.week
    data['quarter'] = data['created'].dt.quarter
    data['weekend'] = ((data['weekday'] == 5) & (data['weekday'] == 6))
    data['weekend'] = data['weekend'].apply(int)
    # data["created_stamp"] = data["created"].apply(lambda x: time.mktime(x.timetuple()))
    #*
    data["latest_list_rank"] = data["latest"] / data["listing_id"]   
    # data["diff_rank_2"] = data["passed"] / data["listing_id"]
    #*

    # image time after leak
    data.loc[data["time_stamp"] > 1490000000, "time_stamp"] = 1478524550
    data["img_created"] = data["time_stamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data["img_latest"] = (data["img_created"]- data["img_created"].min())
    data["img_latest"] = data["img_latest"].apply(lambda x: x.total_seconds())
    data["img_passed"] = (data["img_created"].max()- data["img_created"])
    data["img_passed"] = data["img_passed"].apply(lambda x: x.total_seconds())
    data["img_year"] = data["img_created"].dt.year
    data['img_month'] = data['img_created'].dt.month
    data['img_day'] = data['img_created'].dt.day
    data['img_hour'] = data['img_created'].dt.hour
    # data['img_weekday'] = data['img_created'].dt.weekday
    # data['img_week'] = data['img_created'].dt.week
    # data['img_quarter'] = data['img_created'].dt.quarter
    # data['img_weekend'] = ((data['img_weekday'] == 5) & (data['img_weekday'] == 6))
    # data['img_weekend'] = data['img_weekend'].apply(int)
    data["img_latest_list_rank"] = data["img_latest"] / data["listing_id"] 

    return data


def descriptionProcess(data, train_idx, test_idx):
    data["description_words_num"] = data["description"].apply(lambda x: len(x.split(' ')))
    data["description_len"] = data["description"].apply(len)
    # Some info from descriptions
    desc_feats = {
                  'bedroom_mentions': ['br ', '---', "<a", "a>", "<p>"],
                  'html_tag_1':["<img ", "</a>", "<li>", "</li>", "<ul>", "</ul>", "-->", "<close","<hr"],
                }
    for name, kwords in desc_feats.items():
        data[name] =  data['description'].apply(lambda x: sum([x.count(w)  for w in kwords]))

    data['description'] =  data['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc") 
    # Tf-idf Encode
    tfidfdesc=TfidfVectorizer(min_df=20, max_features=50, strip_accents='unicode',lowercase =True,
                        analyzer='word', token_pattern=r'\w{16,}', ngram_range=(1, 2), use_idf=False,smooth_idf=False, 
    sublinear_tf=True, stop_words = 'english')  
    tr_sparsed = tfidfdesc.fit_transform (data.iloc[train_idx, :]["description"])  
    te_sparsed = tfidfdesc.transform(data.iloc[test_idx, :]["description"])
    feats_names = ["desc_" + x for x in tfidfdesc.get_feature_names()]
    return data, tr_sparsed, te_sparsed, feats_names


def displayAddrProcess(data):
    # disp_price_dict = dict(data.groupby('display_address')['price'].mean())
    # data["mean_disp_price"] = data.apply(lambda row: disp_price_dict[row["display_address"]], axis=1)
    # data["addr_sim"] = data.apply(lambda row: distance.edit_distance(row["display_address"].lower(), row["street_address"].lower()), axis=1)
    return data


def featuresProcess(data, train_idx, test_idx):
    def afterRemoveStr(l, s):
        while s in l:
            l.remove(s)
        return l

    def afterRemoveFirstSpace(l):
        res = []
        for s in l:
            res.append(s.strip())
        return res

    data["features_num"] = data["features"].apply(len)
    mark = "#+-+#"
    data["features"] = data["features"].apply(lambda x: mark.join([i for i in x]))
    data["features"] = data["features"].apply(lambda x: x.lower())

    # Deal with list like data
    data["features"] = data["features"].apply(lambda x: mark.join([i for i in x.split(" * ")]))
    data["features"] = data["features"].apply(lambda x: mark.join([i for i in x.split("**")]))
    data['features']=data['features'].str.replace("✓ hardwood floor ✓ high ceilings ✓ dishwasher",
        "hardwood floor" + mark + "high ceilings" + mark + "dishwasher")
    data['features']=data['features'].str.replace(
        "• on-site lifestyle concierge by luxury attaché " + 
        "•24/7 doorman " + 
        "• state of the art cardiovascular and weight training equipment " +
        "• 24-hour valet parking garage " +
        "• valet services including dry cleaning",
        "on-site lifestyle concierge by luxury attaché" + mark + 
        "24/7 doorman" + mark + 
        "state of the art cardiovascular and weight training equipment" + mark + 
        "24-hour valet parking garage" + mark + 
        "valet services including dry cleaning")
    data['features']=data['features'].str.replace(
        '{     0 = "laundry in unit";     ' + 
        '1 = "cats allowed";     '+
        '10 = hardwood;     '+
        '11 = "high ceilings";     '+
        '12 = renovated;     '+
        '13 = "marble bath";     '+
        '14 = "granite kitchen";     '+
        '15 = light;     '+
        '16 = "no fee";     '+
        '17 = "walk-in closet";     '+
        '2 = "dogs allowed";     '+
        '3 = elevator;     '+
        '4 = exclusive;     '+
        '6 = laundry;     '+
        '7 = subway;     '+
        '8 = dishwasher;     '+
        '9 = washer; }',
        "laundry in unit" + mark + "cats allowed" + mark + "hardwood" + 
        "high ceilings" + mark + "renovated" + mark + "marble bath" + 
        "granite kitchen" + mark + "light" + mark + "no fee" +
        "walk-in closet" + mark + "dogs allowed" + mark + "elevator" +
        "exclusive" + mark + "laundry" + mark + "subway"+
        "dishwasher" + mark + "washer")
    data['features']=data['features'].str.replace("windowed air-conditioned and monitored laundry room",
        "windowed air-conditioned" + mark + "monitored laundry room")
    data['features']=data['features'].str.replace("wall of windows. huge bedrooms",
        "wall of windows" + mark + "huge bedrooms")
    data['features']=data['features'].str.replace("to relax and recharge. this spacious 3 bedroom/2 bath residence also features oak hardwood flooring",
        "spacious" + mark + "3 bedroom" + mark + "2 bath" + mark + "residence" + mark + "oak hardwood flooring")
    data['features']=data['features'].str.replace("stunning 3 bedroom apartment with a terrace! east harlem! the best deal out now! get it now!!!!",
        "stunning" + mark + "3 bedroom" + mark + "a terrace" + mark + "east harlem" + mark + "the best deal out now! get it now!!!!")
    data['features']=data['features'].str.replace("ss appliances - d/w -  m/w - recessed lighting - hardwood floors - high ceilings - marble bath",
        "ss appliances - d/w -  m/w - " + mark + "recessed lighting" + mark + "hardwood floors" + mark + "high ceilings" + mark + "marble bath")
    data['features']=data['features'].str.replace("spacious living room for any kind of entertainment. prime location in theater distric",
        "spacious living room for any kind of entertainment." + mark + "prime location in theater distric")
    data['features']=data['features'].str.replace("spacious living room + home office",
        "spacious living room" + mark + "home office")
    data['features']=data['features'].str.replace("spacious and sunny 1st floor apartment "+
        "overlooking the garden  " + 
        "*great williamsburg location*  "+
        "steps from shopping and cafes "+
        "and 5 minute walk to graham avenue l train (3rd stop from manhattan)  "+
        "*shared back yard * "+
        "large box style rooms * "+
        "huge living room with high ceilings * "+
        "nice bathroom with granite floor & ceramic tile * "+
        "beautiful kitchen with granite counter tops  lots of closet spacehardwood floors *"+
        " heat included in the rent  "+
        "clean quiet building   "+
        "cat ok  "+
        "great location close to shopping",
        "spacious"+ mark +"sunny 1st floor"+ mark+ 
        "overlooking the garden" + mark+ 
        "great williamsburg location"+ mark+ 
        "steps from shopping and cafes"+ mark+ 
        "5 minute walk to graham avenue"+ mark +"train (3rd stop from manhattan)"+ mark+ 
        "shared back yard"+mark+ 
        "large box style rooms"+mark+ 
        "huge living room " + mark + "high ceilings"+ mark+ 
        "nice bathroom" + mark +"granite floor" + mark +"ceramic tile * "+mark+ 
        "beautiful kitchen" + mark +"granite counter tops" + mark +"closet " + mark +"spacehardwood floors"+mark+ 
        "heat included in the rent"+mark+ 
        "clean quiet building"+mark+ 
        "cat ok"+mark+ 
        "close to shopping")
    data['features']=data['features'].str.replace("residents-only " + 
        "fitness center " + 
        "and aerobic room " + 
        "professionally outfitted with a full complement of strength and cardio-training equipment",
        "residents-only"+ mark +"itness center"+ mark+ 
        "and aerobic room" + mark+ 
        "cardio-training equipment")
    data['features']=data['features'].str.replace("owner occupied - " + 
        "3 family townhouse - " + 
        "no realtor fees -"+
        " this beautiful apt is offered below market rate",
        "owner occupied"+ mark +"3 family townhouse"+ mark+ 
        "no realtor fees" + mark+ 
        "this beautiful apt is offered below market rate")
    data['features']=data['features'].str.replace("newly renovated "+
        "w/ oak wood floors   "+
        "mid century modern style interior   "+
        "large closets in every bedroom "+
        "extra storage space in hall. "+
        "large living room",
        "newly renovated"+ mark +"oak wood floors"+ mark+ 
        "mid century modern style interior" + mark+ 
        "large closets in every bedroom" + mark+ 
        "extra storage space in hall"+ mark +"large living room")
    data['features']=data['features'].str.replace("live-in super package room "+
        "smoke-free "+
        "storage available "+
        "virtual doorman "+
        "guarantors accepted",

        "live-in super package room"+ mark +"smoke-free"+ mark+ 
        "storage available" + mark+ 
        "virtual doorman" + mark+ 
        "guarantors accepted")
    data['features']=data['features'].str.replace("live-in super package room "+
        "smoke-free "+
        "storage available "+
        "virtual doorman "+
        "guarantors accepted",

        "live-in super package room"+ mark +"smoke-free"+ mark+ 
        "storage available" + mark+ 
        "virtual doorman" + mark+ 
        "guarantors accepted")

    # Merging some features
    data['features']=data['features'].str.replace("washer/dyer combo","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer inside the unit","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer in-unit","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer in building","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer in bldg","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer hookup","washer/dyer")
    data['features']=data['features'].str.replace("washer/dryer  stove/oven","washer/dyer")
    data['features']=data['features'].str.replace("washer/drier hookups","washer/dyer")
    data['features']=data['features'].str.replace("washer/ dryer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer/ dryer hookups","washer/dyer")
    data['features']=data['features'].str.replace("washer-dryer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer-dryer hookups","washer/dyer")
    data['features']=data['features'].str.replace("washer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer dryer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer dryer hookup","washer/dyer")
    data['features']=data['features'].str.replace("washer dryer hook up","washer/dyer")
    data['features']=data['features'].str.replace("washer and dryer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer and dryer in the unit","washer/dyer")
    data['features']=data['features'].str.replace("washer and dryer","washer/dyer")
    data['features']=data['features'].str.replace("washer / dryer in unit","washer/dyer")
    data['features']=data['features'].str.replace("washer / dryer (hookup only)","washer/dyer")
    data['features']=data['features'].str.replace("washer / dryer","washer/dyer")
    data['features']=data['features'].str.replace("washer & dryer.","washer/dyer")
    data['features']=data['features'].str.replace("washer","washer/dyer")
    data['features']=data['features'].str.replace("wash/dryer","washer/dyer")


    data['features']=data['features'].str.replace("pets: cats/small dogs","pet-friendly")
    data['features']=data['features'].str.replace("pets welcome","pet-friendly")
    data['features']=data['features'].str.replace("pets upon approval","pet-friendly")
    data['features']=data['features'].str.replace("pets on approval","pet-friendly")
    data['features']=data['features'].str.replace("pets ok.","pet-friendly")
    data['features']=data['features'].str.replace("pets ok","pet-friendly")
    data['features']=data['features'].str.replace("pets are welcome","pet-friendly")
    data['features']=data['features'].str.replace("pets allowed","pet-friendly")
    data['features']=data['features'].str.replace("pets accepted (on approval)","pet-friendly")
    data['features']=data['features'].str.replace("pets","pet-friendly")
    data['features']=data['features'].str.replace("pet grooming room","pet-friendly")
    data['features']=data['features'].str.replace("pet friendly building","pet-friendly")
    data['features']=data['features'].str.replace("pet friendly ( case by case )","pet-friendly")
    data['features']=data['features'].str.replace("pet friendly","pet-friendly")
    data['features']=data['features'].str.replace("pet friendly building","pet-friendly")
    data['features']=data['features'].str.replace("pet friendly building","pet-friendly")

    data['features']=data['features'].str.replace("garden/patio","garden")
    data['features']=data['features'].str.replace("patio","garden")
    data['features']=data['features'].str.replace("residents_garden","garden")
    data['features']=data['features'].str.replace("common garden","garden")

    data['features']=data['features'].str.replace("wifi access","wifi")
    data['features']=data['features'].str.replace("wifi included","wifi")
    data['features']=data['features'].str.replace("wifi in resident lounge","wifi")
    data['features']=data['features'].str.replace("wifi + utilities","wifi")
    data['features']=data['features'].str.replace("wi fi work lounge","wifi")
    data['features']=data['features'].str.replace("wi-fi access","wifi")

    data['features']=data['features'].str.replace("24/7","24")
    data['features']=data['features'].str.replace("24-hour","24")
    data['features']=data['features'].str.replace("24hr","24")
    data['features']=data['features'].str.replace("concierge","doorman")
    data['features']=data['features'].str.replace("ft doorman","doorman")
    data['features']=data['features'].str.replace("24 doorman","doorman")
    data['features']=data['features'].str.replace("24 hr doorman","doorman")
    data['features']=data['features'].str.replace("doorman service","doorman")
    data['features']=data['features'].str.replace("full-time doorman","doorman")

    data['features']=data['features'].str.replace("gym/fitness","fitness")
    data['features']=data['features'].str.replace("fitness room","fitness")

    data['features']=data['features'].str.replace("washer","laundry")
    data['features']=data['features'].str.replace("laundry in bldg","laundry")
    data['features']=data['features'].str.replace("laundry in building","laundry")
    data['features']=data['features'].str.replace("laundry in building/dryer","laundry")
    data['features']=data['features'].str.replace("laundry in building_&_dryer","laundry")
    data['features']=data['features'].str.replace("laundry room","laundry")
    data['features']=data['features'].str.replace("laundry & housekeeping","laundry")
    data['features']=data['features'].str.replace("laundry in unit","laundry")
    data['features']=data['features'].str.replace("laundry in-unit","laundry")
    data['features']=data['features'].str.replace("laundry on every floor","laundry")
    data['features']=data['features'].str.replace("laundry on floor","laundry")
    data['features']=data['features'].str.replace("in-unit laundry/dryer","laundry")
    data['features']=data['features'].str.replace("on-site laundry","laundry")
    data['features']=data['features'].str.replace("laundry/dryer","laundry")

    data['features']=data['features'].str.replace("high-speed internet","high_speed_internet")
    data['features']=data['features'].str.replace("high speed internet available","high_speed_internet")

    data['features']=data['features'].str.replace("parking available","parking")
    data['features']=data['features'].str.replace("parking space","parking")
    data['features']=data['features'].str.replace("on-site garage","parking")
    data['features']=data['features'].str.replace("on-site parking","parking")
    data['features']=data['features'].str.replace("on-site parking lot","parking")
    data['features']=data['features'].str.replace("full service garage","parking")
    data['features']=data['features'].str.replace("common parking/garage","parking")
    data['features']=data['features'].str.replace("garage","parking")
    data['features']=data['features'].str.replace("assigned-parking-space","private_parking")

    data['features']=data['features'].str.replace("storage available","storage")
    data['features']=data['features'].str.replace("storage facilities available","storage")
    data['features']=data['features'].str.replace("storage space","storage")
    data['features']=data['features'].str.replace("storage room","storage")
    data['features']=data['features'].str.replace("common storage","storage")

    data['features']=data['features'].str.replace("central a/c","central_air")
    data['features']=data['features'].str.replace("central ac","central_air")
    data['features']=data['features'].str.replace("air conditioning","central_air")

    data['features']=data['features'].str.replace("close to  subway","subway")

    data['features']=data['features'].str.replace("roofdeck","roof-deck")
    data['features']=data['features'].str.replace("roof-deck","roof-deck")
    data['features']=data['features'].str.replace("rooftop terrace","roof-deck")
    data['features']=data['features'].str.replace("rooftop deck","roof-deck")
    data['features']=data['features'].str.replace("roof access","roof-deck")
    data['features']=data['features'].str.replace("common roof deck","roof-deck")
    data['features']=data['features'].str.replace("roof decks","roof-deck")
    data['features']=data['features'].str.replace("roof grilling area","roof-deck")
    data['features']=data['features'].str.replace("roof garden and lounge","roof-deck")
    data['features']=data['features'].str.replace("roof deck with stunning view","roof-deck")
    data['features']=data['features'].str.replace("roof deck with real grass","roof-deck")
    data['features']=data['features'].str.replace("roof deck with grills","roof-deck")
    data['features']=data['features'].str.replace("roof deck w/ grills","roof-deck")
    data['features']=data['features'].str.replace("roof deck / sun deck","roof-deck")
    data['features']=data['features'].str.replace("roof deck","roof-deck")

    data['features']=data['features'].str.replace("swimming pool","pool")
    data['features']=data['features'].str.replace("indoor pool","pool")

    data['features']=data['features'].str.replace("deco fireplace","fireplaces")
    data['features']=data['features'].str.replace("decorative fireplace","fireplaces")

    data['features']=data['features'].str.replace("yoga/pilates studio","yoga")
    data['features']=data['features'].str.replace("yoga studio","yoga")
    data['features']=data['features'].str.replace("yoga room","yoga")
    data['features']=data['features'].str.replace("yoga classes","yoga")
    data['features']=data['features'].str.replace("yoga and spin studios","yoga")
    data['features']=data['features'].str.replace("yoga an pilates class","yoga")
    data['features']=data['features'].str.replace("yoga / dance studio","yoga")


    # data["features"] = data["features"].apply(lambda x: afterRemoveStr(x, ''))
    # data["features"] = data["features"].apply(lambda x: afterRemoveFirstSpace(x))
    data["features"] = data["features"].apply(lambda x: x.split(mark))
    data["features"] = data["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    tfidf = CountVectorizer(stop_words="english", max_features=200)
    tr_sparse_feats = tfidf.fit_transform(data.iloc[train_idx, :]["features"])
    te_sparse_feats = tfidf.transform(data.iloc[test_idx, :]["features"])
    feats_names = ["features_" + x for x in tfidf.get_feature_names()]
    return data, tr_sparse_feats, te_sparse_feats, feats_names


def locationProcess(data, train_idx, test_idx):
    # Clustering

    # train_x = data.iloc[train_idx,:][['new_latitude', 'new_longitude']]
    # stest_x = data.iloc[test_idx,:][['new_latitude', 'new_longitude']]
    train_x = data.iloc[train_idx, :][['latitude', 'longitude']]
    test_x = data.iloc[test_idx, :][['latitude', 'longitude']]
    kmeans_cluster = KMeans(n_clusters=20)
    res = kmeans_cluster.fit(train_x)
    res = kmeans_cluster.predict(pd.concat([train_x, test_x]))
    d = dict(zip(data['listing_id'], res))
    data['cenroid'] = data['listing_id'].apply(lambda x: d[x])
    # Manhattan distance
    center = [data.iloc[train_idx, :]['latitude'].mean(), data.iloc[train_idx, :]['longitude'].mean()]
    data['distance'] = abs(data['latitude'] - center[0]) + abs(data['longitude'] - center[1])
    # data['distance_2'] = np.sqrt((data['latitude'] - center[0]) ** 2 + (data['longitude'] - center[1]) ** 2)

    return data


def managerIdProcess(data, y, train_idx, test_idx):
    manager_lgt_dict = dict(data.groupby('manager_id')['longitude'].mean())
    manager_ltt_dict =  dict(data.groupby('manager_id')['latitude'].mean())

    # Group manager_id with location info
    data["mean_man_longitude"] = data.apply(lambda row: manager_lgt_dict[row["manager_id"]], axis=1)
    data["mean_man_latitude"] = data.apply(lambda row: manager_ltt_dict[row["manager_id"]], axis=1)

    # Group manager_id with time info
    data = group_with_time_features(data, "manager_id")
    data = group_with_img_time_features(data, "manager_id")
    manager_stamp_dict = dict(data.groupby('manager_id')['time_stamp'].mean())
    data["mean_man_timestamp"] = data.apply(lambda row: manager_stamp_dict[row["manager_id"]], axis=1)
    # manager_stamp_dict = dict(data.groupby('manager_id')['created_stamp'].mean())
    # data["mean_man_createdstamp"] = data.apply(lambda row: manager_stamp_dict[row["manager_id"]], axis=1)  
    return data


def photoProcess(data):
    data["photo_num"] = data["photos"].apply(len)
    return data


def priceProcess(data):
    #data["out_price"] = data["price"].apply(lambda x: 1 if x < 700 or x > 15000 else 0)
    # Clean the outlier
    ulimit = 15000#np.percentile(data.price.values, 99)
    data.loc[data["price"] > ulimit, "price"] = ulimit
    dlimit = 350
    data.loc[data["price"] < dlimit, "price"] = dlimit
    data["price_per_room"] = data["price"] / (data["bedrooms"] + data["bathrooms"] + 1.0)
    data["price_per_bed"] = data["price"] / (data["bedrooms"] + 1.0)
    #*
    # data.loc[~np.isfinite(data["price_per_room"]), "price_per_room"] = 0
    # data.loc[~np.isfinite(data["price_per_bed"]), "price_per_bed"] = 0
    data["price_latitude"] = data["price"] / (data["latitude"] + 1.0)
    data["price_longitude"] = data["price"] / (data["longitude"] + 1.0)

    # Grouping price with size or build
    median_list = ['bedrooms', 'bathrooms', 'building_id']
    # median_list = ['month', 'day', 'hour', 'weekday', 'quarter', 'week', 'passed', 'latest']
    for col in median_list:
        median_price = data[[col, 'price']].groupby(col)['price'].median()
        median_price = median_price[data[col]].values.astype(float)
        data['median_' + col] = median_price
        data['ratio_' + col] = data['price'] / median_price
        data['median_' + col] = data['median_' + col].apply(lambda x: np.log(x))
    # data["price"] = data["price"].apply(lambda x: np.log(x))
    return data


def streetAddrProcess(data):
    #data["new_addr"] = data["street_address"].apply(lambda x: ' '.join([x.split()[i] for i in range(1, len(x.split()))]))
    #data["new_addr"] = preprocessing.LabelEncoder().fit_transform(data["new_addr"])
    # data["street_address"] = data["street_address"].apply(lambda x: x.replace('\u00a0', '').strip().lower)
    return data


def listingIdProcess(data):
    # It's weird。
    data["listing_id"] = data["listing_id"] - 68119576.0
    return data


def coreProcess(data, y_train, train_idx, test_idx):
    data = listingIdProcess(data)
    data = bedroomProcess(data, train_idx, test_idx)
    data = bathroomProcess(data, train_idx, test_idx)
    data["room_diff"] = data["bathrooms"] - data["bedrooms"]
    data["room_num"] = data["bedrooms"] + data["bathrooms"]
    data = createdProcess(data)
    data = buildingIdProcess(data, y_train, train_idx, test_idx)
    data, tr_sparsed, te_sparsed, feats_sparsed = descriptionProcess(data, train_idx, test_idx)
    data = displayAddrProcess(data)
    data, tr_sparse, te_sparse, feats_sparse = featuresProcess(data, train_idx, test_idx)
    data = locationProcess(data, train_idx, test_idx)
    data = managerIdProcess(data, y_train, train_idx, test_idx)
    data = photoProcess(data)
    data = priceProcess(data)
    data = streetAddrProcess(data)
    
    categorical = ["display_address", "manager_id", "building_id", "street_address"]
    for f in categorical:
        if data[f].dtype=='object':
            cases=defaultdict(int)
            temp=np.array(data[f]).tolist()
            for k in temp:
                cases[k]+=1
            # print(f, len(cases))
            data[f] = data[f].apply(lambda x: cases[x])
            
    feats_in_use = [col for col in data.columns if col not in FEATURE_NOT_USE]

    data_train = np.array(data.iloc[train_idx, :][feats_in_use])
    data_test  = np.array(data.iloc[test_idx, :][feats_in_use])
    # Feature Scaling
    stda = StandardScaler()  
    data_test = stda.fit_transform(data_test)          
    data_train = stda.transform(data_train)
    #  High cardinality feature
    high_card_feats = ["building_id", "manager_id", "longitude", "room_diff"] # "building_id", "manager_id", 
    # C0 = [3, 12, 0, 4]
    C0 = [feats_in_use.index(f) for f in high_card_feats]
    W_train, W_cv = convert_to_avg(data_train, y_train, data_test, seed=1, cvals=5, roundings=2, columns=C0)
    #  Add Sparse feature
    data_train = sparse.hstack([data_train, tr_sparse, tr_sparsed, W_train[:, C0]]).tocsr()
    data_test = sparse.hstack([data_test, te_sparse, te_sparsed, W_cv[:, C0]]).tocsr()
    feats_in_use.extend(feats_sparse)
    feats_in_use.extend(feats_sparsed)
    feats_in_use.extend(["build_high_card", "manager_high_card"])
    # print(len(feats_in_use))
    # print(tr_sparse.toarray().shape, tr_sparsed.toarray().shape, len(feats_in_use), data_train.shape)
    return data_train, data_test, feats_in_use


# Copy from KazAnova's starter code
def convert_dataset_to_avg(xc,yc,xt, rounding=2,cols=None):
    xc = xc.tolist()
    xt = xt.tolist()
    yc = yc.tolist()
    if cols == None:
        cols =[k for k in range(0,len(xc[0]))]
    woe=[ [0.0 for k in range(0,len(cols))] for g in range(0,len(xt))]
    good=[]
    bads=[]
    for col in cols:
        dictsgoouds=defaultdict(int)        
        dictsbads=defaultdict(int)
        good.append(dictsgoouds)
        bads.append(dictsbads)        
    total_count=0.0
    total_sum =0.0

    for a in range (0,len(xc)):
        target=yc[a]
        total_sum+=target
        total_count+=1.0
        for j in range(0,len(cols)):
            col=cols[j]
            good[j][round(xc[a][col],rounding)]+=target
            bads[j][round(xc[a][col],rounding)]+=1.0  
    #print(total_goods,total_bads)            
    
    for a in range (0,len(xt)):    
        for j in range(0,len(cols)):
            col=cols[j]
            if round(xt[a][col],rounding) in good[j]:
                 woe[a][j]=float(good[j][round(xt[a][col],rounding)])/float(bads[j][round(xt[a][col],rounding)])  
            else :
                 woe[a][j]=round(total_sum/total_count)
    return woe            


def convert_to_avg(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    
    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]    
    #print("it is not!!")        
    X=X.tolist()
    Xt=Xt.tolist() 
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]    
    
    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_avg(X_train,y_train,X_cv, rounding=roundings,cols=columns)
        X_cv=X_cv.tolist()
        no=0
        for real_index in test_index:
            for j in range(0,len(X_cv[0])):
                woetrain[real_index][j]=X_cv[no][j]
            no+=1
        no=0
        for real_index in test_index:
            for j in range(0,len(columns)):
                col=columns[j]
                woetrain[real_index][col]=woecv[no][j]
            no+=1      
    woefinal=convert_dataset_to_avg(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns) 

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):           
            woetest[real_index][j]=Xt[real_index][j]
            
    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]
            
    return np.array(woetrain), np.array(woetest)


# Grouping (Very important)
def group_with_time_features(data, g_feat):
    mean_month_dict = dict(data.groupby(g_feat)['month'].mean())
    data["mean_" + g_feat + "_month"] = data.apply(lambda row: mean_month_dict[row[g_feat]], axis=1)
    mean_day_dict = dict(data.groupby(g_feat)['day'].mean())
    data["mean_" + g_feat + "_day"] = data.apply(lambda row: mean_day_dict[row[g_feat]], axis=1)
    mean_hour_dict = dict(data.groupby(g_feat)['hour'].mean())
    data["mean_" + g_feat + "_hour"] = data.apply(lambda row: mean_hour_dict[row[g_feat]], axis=1)
    mean_weekday_dict = dict(data.groupby(g_feat)['weekday'].mean())
    data["mean_" + g_feat + "_weekday"] = data.apply(lambda row: mean_weekday_dict[row[g_feat]], axis=1)
    mean_quarter_dict = dict(data.groupby(g_feat)['quarter'].mean())
    data["mean_" + g_feat + "_quater"] = data.apply(lambda row: mean_quarter_dict[row[g_feat]], axis=1)
    mean_week_dict = dict(data.groupby(g_feat)['week'].mean())
    data["mean_" + g_feat + "_week"] = data.apply(lambda row: mean_week_dict[row[g_feat]], axis=1)
    mean_passed_dict = dict(data.groupby(g_feat)['passed'].mean())
    data["mean_" + g_feat + "_passed"] = data.apply(lambda row: mean_passed_dict[row[g_feat]], axis=1)
    mean_latest_dict = dict(data.groupby(g_feat)['latest'].mean())
    data["mean_" + g_feat + "_latest"] = data.apply(lambda row: mean_latest_dict[row[g_feat]], axis=1)

    return data


def group_with_img_time_features(data, g_feat):
    mean_month_dict = dict(data.groupby(g_feat)['img_month'].mean())
    data["mean_" + g_feat + "_img_month"] = data.apply(lambda row: mean_month_dict[row[g_feat]], axis=1)
    mean_day_dict = dict(data.groupby(g_feat)['img_day'].mean())
    data["mean_" + g_feat + "_img_day"] = data.apply(lambda row: mean_day_dict[row[g_feat]], axis=1)
    mean_hour_dict = dict(data.groupby(g_feat)['img_hour'].mean())
    data["mean_" + g_feat + "_img_hour"] = data.apply(lambda row: mean_hour_dict[row[g_feat]], axis=1)
    # mean_weekday_dict = dict(data.groupby(g_feat)['img_weekday'].mean())
    # data["mean_" + g_feat + "_img_weekday"] = data.apply(lambda row: mean_weekday_dict[row[g_feat]], axis=1)
    # mean_quarter_dict = dict(data.groupby(g_feat)['img_quarter'].mean())
    # data["mean_" + g_feat + "_img_quater"] = data.apply(lambda row: mean_quarter_dict[row[g_feat]], axis=1)
    # mean_week_dict = dict(data.groupby(g_feat)['img_week'].mean())
    # data["mean_" + g_feat + "_img_week"] = data.apply(lambda row: mean_week_dict[row[g_feat]], axis=1)
    mean_passed_dict = dict(data.groupby(g_feat)['img_passed'].mean())
    data["mean_" + g_feat + "_img_passed"] = data.apply(lambda row: mean_passed_dict[row[g_feat]], axis=1)
    mean_latest_dict = dict(data.groupby(g_feat)['img_latest'].mean())
    data["mean_" + g_feat + "_img_latest"] = data.apply(lambda row: mean_latest_dict[row[g_feat]], axis=1)
    return data






