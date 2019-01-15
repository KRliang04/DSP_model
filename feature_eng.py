#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:37:39 2019

@author: kiraliang
"""
from __future__ import division
import sys
import os
import glob
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text_files_dsat = glob.glob("../conversations/dsat/*.txt")
text_files_sat = glob.glob("../conversations/sat/*.txt")
# sentences to delete
dic = {'thankyou':"", 'thank you':"", "goodbye":"", "good bye":"", 'bye':""}

def read_file(file):
    with open (file) as f:
        lines = [line.rstrip('\n') for line in f]
    id_name = file.split("/")[-1].split(".txt")[0]
    return id_name,lines

def number_repetition(lines):
    new_list =[]
    for line in lines:
        if "SYSTEM" in line:
            new_list.append(line.strip())
    #create dictionary and count the repetition of sentence (Feature 1)
    new_dict = {}
    for li in new_list:
        if li in new_dict:
            new_dict[li] += 1
        else:
            new_dict[li] = 1
    num_rep = sum([value for value in new_dict.values() if value > 1])
    #Percentage of the repetition sentence in the entire conversation (Feature 3)
    num_rep_per = num_rep/len(new_list)
    
    return num_rep,num_rep_per, len(lines)

def pos_neg_conv(file):
    # get everything
    
    dict_conv_only = create_user_conv(file)
    compound_score, tot_pos_sen, tot_neg_sen = get_sentiment(dict_conv_only)
    return compound_score, tot_pos_sen, tot_neg_sen
    
    
def get_sentiment(dict_conv_only):
    # generate the compound sentiment score and store in conversation

    for sentence in dict_conv_only.keys():
        compound_sentence = generatesentiment(sentence)
        # add compound score to sentence 
        dict_conv_only[sentence] = compound_sentence
    # add total compound score of sentence to conversation
    compound_score = sum(dict_conv_only.values())/len(dict_conv_only.values())
    tot_pos_sen = len([x for x in dict_conv_only.values() if x > 0])/len(dict_conv_only.values())
    tot_neg_sen = len([x for x in dict_conv_only.values() if x < 0])/len(dict_conv_only.values())
    return compound_score, tot_pos_sen, tot_neg_sen
    
    
def create_user_conv(lines):
    #create a list per conversation of only USER sentences 
    # and 'thankyou goodbye left out'
    conv_only_user = []
    new_list =[]
    for line in lines:
        # only USER input
        if "USER" in line:
            # Remove the word '[USER]'
            line = line.replace("[USER]", "")
            line = line.strip()
            new_list.append(line)
    # Remove 'thank you good bye' if in last sentences
    for i,j in dic.items():
        new_list[-1] = new_list[-1].replace(i, j)
        new_list[-2] = new_list[-2].replace(i, j)
    conv_only_user.append(new_list)

    # convert to dict 
    # with every conversation a dict with sentences as keys
    # and compound pos neg score (as 0)
    for conv in conv_only_user:
        new_dict2 = {}
        for sentence in conv:
            new_dict2[sentence] = 0
    return new_dict2       
        
def generatesentiment(sentence):
    # function for sentiment analysis on sentences
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    return ss['compound']



def main():
  
  
    if sys.argv[1] == str(1):
        print("Training...")
        if not os.path.exists("./features"):
            os.makedirs("./features")
        #create feature dataframe for disatified dataset
        columns = ['id','num_rep','num_rep_per','len_conversation','total_compound_conv', 'tot_pos_sen', 'tot_neg_sen', 'Is_satisfied']
        
        df_rep_dsat = pd.DataFrame(index=range(len(text_files_dsat)), columns=columns)
        for idx,file in enumerate(text_files_dsat):
            id_number,lines = read_file(file)
            num_rep, num_rep_per,len_conversation = number_repetition(lines)
            compound_score, tot_pos_sen, tot_neg_sen = pos_neg_conv(lines)
            df_rep_dsat.iloc[idx] = pd.Series({'id':id_number, 'num_rep':num_rep, 'num_rep_per':num_rep_per,
                                          'len_conversation':len_conversation,
                                               'total_compound_conv':compound_score, 'tot_pos_sen':tot_pos_sen, 
                                               'tot_neg_sen':tot_neg_sen, 'Is_satisfied':0})
        #create feature dataframe for disatified dataset
        df_rep_sat = pd.DataFrame(index=range(len(text_files_sat)), columns=columns)
        for idx,file in enumerate(text_files_sat):
            id_number,lines = read_file(file)
            num_rep, num_rep_per,len_conversation = number_repetition(lines)
            compound_score, tot_pos_sen, tot_neg_sen = pos_neg_conv(lines)
            df_rep_sat.iloc[idx] = pd.Series({'id':id_number, 'num_rep':num_rep, 'num_rep_per':num_rep_per,
                                          'len_conversation':len_conversation,
                                               'total_compound_conv':compound_score, 'tot_pos_sen':tot_pos_sen, 
                                               'tot_neg_sen':tot_neg_sen, 'Is_satisfied':1})
        new_df = pd.concat([df_rep_dsat,df_rep_sat])
        #df_shuffle = new_df.sample(frac=1).reset_index(drop=True)
        new_df.to_csv("features/features.csv",index=False)
        os.system("python train.py")
        
    else:
        print("Predicting...")
        file = sys.argv[2]
        id_number,lines = read_file(file)
        num_rep, num_rep_per,len_conversation = number_repetition(lines)
        compound_score, tot_pos_sen, tot_neg_sen = pos_neg_conv(lines)
        features_df = pd.Series({'id':id_number, 'num_rep':num_rep, 'num_rep_per':num_rep_per,
                                          'len_conversation':len_conversation,
                                               'total_compound_conv':compound_score, 'tot_pos_sen':tot_pos_sen, 
                                               'tot_neg_sen':tot_neg_sen})
        features_df = pd.DataFrame(features_df).transpose()
        features_df.to_csv("./features/features_predict.csv",index=False)
        os.system("python predict.py")
        
if __name__ == "__main__":
    main()
    
    