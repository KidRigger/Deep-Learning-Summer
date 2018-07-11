#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:42:35 2018

@author: nishchay
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 
import re
import csv
import pickle

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def replace_values_in_list(the_list,val,replace_value):
    for i,d in enumerate(the_list):
        if (d==val):
            the_list[i]=replace_value
        else:
            continue
    return the_list

def clean_application_names(applicationNames):
    stop_words = set(stopwords.words('english'))
    doc_clean=[]
    applicationNames[applicationNames.notna()==False]='None'
    for applicationName in applicationNames:
        #tokenize the application names into single words
        text=nltk.word_tokenize(applicationName)
        
        #remove stop words from the application name
        for word in text:
            if word.lower() in (stop_words):
                text.remove(word)
                
        # remove all the words with any character other than alphabets
        text=[re.sub("[^a-zA-Z]+",' ',word) for word in text]
        text=remove_values_from_list(text,' ')
        doc_clean.append(text)
    for i in range(0,len(doc_clean)):
        
        #join all the tokenized words of one application name into one string
        doc_clean[i]=' '.join(doc_clean[i])

    # replace all the empty strings with None
    text=replace_values_in_list(doc_clean,'','None')
    text=replace_values_in_list(doc_clean,' ','None')
    return text

def init_vectorizer(mindf,n_gram_range):
    return CountVectorizer(stop_words='english',min_df=mindf,ngram_range=n_gram_range)

def init_tfidf_vectorizer(mindf,n_gram_range):
    return TfidfVectorizer(stop_words='english',min_df=mindf,ngram_range=n_gram_range)

def fit_vectorizer_application_names(text,vectorizer):
    X = vectorizer.fit_transform(text)
    return X

def vectorize_application_names(text,vectorizer):
    return vectorizer.transform(text)

def kmeansfit(num_clusters,max_iterations,n_initial,X):
    init_centroids=checkLoadedModelCentroids()
    if init_centroids==False:
        modelkmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=max_iterations, n_init=n_initial)
        modelkmeans.fit(X)
        return modelkmeans
    else:
        print ('Using previous centroids')
        modelkmeans = KMeans(n_clusters=num_clusters, init=loadModelCentroids(), max_iter=max_iterations, n_init=n_initial)
        modelkmeans.fit(X)
        return modelkmeans

def predict_labels(X,modelkmeans):
    predicted_labels_kmeans = modelkmeans.predict(X)
    return predicted_labels_kmeans

def saveModelCentroids(modelkmeans):
    pickle.dump(modelkmeans.cluster_centers_,open("/home/nishchay/Documents/Arcon/Final output/centroid.pickle","wb"))

def checkLoadedModelCentroids():
    try:
        pickle.load(open("/home/nishchay/Documents/Arcon/Final output/centroid.pickle", "rb"))
        return True
        print ('Loading previous centroids')
    except (OSError, IOError) as e:
        return False

def loadModelCentroids():
    return pickle.load(open("/home/nishchay/Documents/Arcon/Final output/centroid.pickle", "rb"))

def feature_names(X,mindf,n_gram_range):
    countVectorizer=CountVectorizer(stop_words='english',min_df=mindf,ngram_range=n_gram_range)
    countX=countVectorizer.fit_transform(X)
    terms=countVectorizer.get_feature_names()
    scores = zip(terms,
                 np.asarray(countX.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    features=list()
    for item in sorted_scores:
        features.append(item[0])
    return features[0:6]

def tfidf_feature_names(X,mindf,n_gram_range):
    tfidfVectorizer=TfidfVectorizer(stop_words='english',min_df=mindf,ngram_range=n_gram_range)
    tfidfX=tfidfVectorizer.fit_transform(X)
    terms=tfidfVectorizer.get_feature_names()
    scores = zip(terms,
                 np.asarray(tfidfX.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    features=list()
    for item in sorted_scores:
        features.append(item[0])
    return features[0:6]

def getData(path):
    df=pd.read_csv(path)
    return df

def outputPerPersonClusters(labelled_df,perPersonPath):
    users=labelled_df.currentWindowsIdentity.unique()
    cols=['User','Clusters']
    perPersonClusters=pd.DataFrame(columns=cols)
    for user in users:
        count_array=labelled_df[labelled_df.currentWindowsIdentity==user].groupby('ApplicationLabel').count().sort_values("ApplicationName",ascending=False).ApplicationName
        perPersonClusters=perPersonClusters.append({'User':user,'Clusters':count_array.index.values[0:4]},ignore_index=True)
        print (perPersonClusters)
    perPersonClusters.to_csv(perPersonPath)

def outputPerCluster(labelled_df,perClusterPath):
    labelled_df['Hour']=pd.to_datetime(labelled_df.updateDateTime).dt.hour
    num_clusters=labelled_df.ApplicationLabel.unique()
    num_clusters.sort()
    for j in num_clusters:
        cluster_features=feature_names(labelled_df[labelled_df.ApplicationLabel==j].ApplicationName,0.05,(2,3))
        percluster_people=list()
        percluster_timestamp=list()
        percluster_mostFrequentProcessName=list()
        percluster_mostTimeUsed=list()
        count_array_people=labelled_df[labelled_df.ApplicationLabel==j].groupby('currentWindowsIdentity').count().sort_values("ApplicationName",ascending=False).ApplicationName
        count_array_timestamp=labelled_df[labelled_df.ApplicationLabel==j].groupby('Hour').count().sort_values('ApplicationName',ascending=False).ApplicationName
        count_array_mostFrequentProcessName=labelled_df[labelled_df.ApplicationLabel==j].groupby('ProcessName').count().sort_values('ApplicationName',ascending=False).ApplicationName
        count_array_mostTimeUsed=labelled_df[labelled_df.ApplicationLabel==j].groupby('ProcessName').TotalSeconds.sum().sort_values(ascending=False)
        i=0
        for index in count_array_people.index:
            if(i<10):
                percluster_people.append(index)
            i=i+1
        i=0
        for index in count_array_timestamp.index:
            if(i<10):
                percluster_timestamp.append(index)
            i=i+1
        i=0
        for index in count_array_mostFrequentProcessName.index:
            if(i<10):
                percluster_mostFrequentProcessName.append(index)
            i=i+1
        i=0
        for index in count_array_mostTimeUsed.index:
            if(i<10):
                percluster_mostTimeUsed.append(index)
            i=i+1
        with open(perClusterPath+str(j)+'.csv','w') as csvfile:
            fieldnames = ['Data']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Data': 'Cluster number :'+str(j)})
            writer.writerow({'Data':'Cluster features: '+str(cluster_features)})
            writer.writerow({'Data':'Most frequent users of this cluster: '+str(percluster_people)})
            writer.writerow({'Data':'Most probable timestamp of this cluster: '+str(percluster_timestamp)})
            writer.writerow({'Data':'Most frequently used processes of this cluster: '+str(percluster_mostFrequentProcessName)})
            writer.writerow({'Data':'Most time used processes of this cluster: '+str(percluster_mostTimeUsed)})

def main():
    df=getData('/home/nishchay/Documents/Arcon/naya_uba.csv')
    # Clean application names and vectorize application names
    cleaned_application_names=clean_application_names(df.ApplicationName)
    vectorizer=init_vectorizer(0.04,(2,3))
    vectorized_application_names=fit_vectorizer_application_names(cleaned_application_names,vectorizer)

    # Create model and fit the vectorized values into clusters using kmeans
    model=kmeansfit(7,20000,100,vectorized_application_names)
    saveModelCentroids(model)
    # Extracting cluster labels
    predicted_labels_kmeans=predict_labels(vectorized_application_names,model)

    allusers_application_labels=pd.DataFrame(df.ApplicationName,columns=['ApplicationName'])
    allusers_application_labels['ApplicationLabel']=predicted_labels_kmeans
    #different way of predicting labels
#    cols=['ApplicationLabel','ApplicationName','ActivityID']
#    all_users_applicationlabels=pd.DataFrame(columns=cols)
#    for applicationName in cleaned_application_names:
#        applicationList=list()
#        applicationList.append(applicationName)
#        predictedLabel=predict_labels(vectorize_application_names(applicationList,vectorizer),model)
#        all_users_applicationlabels=all_users_applicationlabels.append({'ApplicationLabel':predictedLabel,'ApplicationName':applicationName,'ActivityID':df[df.ApplicationName==applicationName].ActivityID},ignore_index=True)
#    all_users_applicationlabels.to_csv('/home/nishchay/Documents/Arcon/Final output/trialoutput.csv')
main()