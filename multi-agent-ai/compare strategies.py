from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import  XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
import numpy as np

BUDGET = [6250000, 6250000]

validation = pd.read_csv('we_data/validation.csv')
# bid price loaded
strategy_1 = pd.read_csv('s1.csv')
strategy_2 = pd.read_csv('s2.csv')
# strategy_3 = pd.read_csv('s3.csv')
# strategy_4 = pd.read_csv('s4.csv')
# strategy_5 = pd.read_csv('s5.csv')
# strategy_6 = pd.read_csv('s6.csv')

price_1 = strategy_1['bidprice']
price_2 = strategy_2['bidprice']
# price_3 = strategy_3['bidprice']
# price_4 = strategy_4['bidprice']
# price_5 = strategy_5['bidprice']
# price_6 = strategy_6['bidprice']

click_set = validation['click']
payprice = validation['payprice']
total_click_1 = 0
total_click_2 = 0

for i in range(len(price_1)):
	if price_1[i] > payprice[i] or price_2[i] > payprice[i]:
		if BUDGET[0] > 0 and BUDGET[1] > 0:
			if price_1[i] > price_2[i] and BUDGET[0] - price_2[i] > 0:
				BUDGET[0] -= price_2[i]
				total_click_1 += click_set[i]
			else if price_2[i] > price_1[i] and BUDGET[1] - price_1[i] > 0:
				BUDGET[1] - price_1[i]
				total_click_2 += click_set[i]
		else:
			break

