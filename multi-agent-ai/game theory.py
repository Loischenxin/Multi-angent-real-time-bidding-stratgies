from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import  XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# dataset = pd.read_csv('out.csv')
# dataset = dataset.drop('useless', axis=1)

# temp = dataset['bidprice']

# for i in range(len(temp)):
# 	if temp[i] == 300:
# 		temp[i] = 100000

# dataset['bidprice'] = temp


# dataset.to_csv('out1.csv', index=False)

BUDGET = 6250000
USELESS_FEATURE = ['adexchange','click', 'bidprice', 'payprice', 'bidid', 'IP', 'userid', 'creative', 'domain', 'url', 'urlid', 'slotid', 'keypage']
USELESS_FEATURE1 = ['adexchange', 'bidid', 'IP', 'userid', 'creative', 'domain', 'url', 'urlid', 'slotid', 'keypage']

# Gradient boosting decision trees GBDT
def preprocess_useragent(dataset):
	# first part of the column
	print "Preprocessing useragent"
	dataset['os_system'] = dataset['useragent'].apply(lambda x: x.split('_')[0])
	# second part of the column
	dataset['browser'] = dataset['useragent'].apply(lambda x: x.split('_')[1])
	# delete user agent
	return dataset.drop('useragent', axis=1)

def preprocess_slotprice(dataset):
	# divide slot price into 3 parts
	print "Preprocessing slotprice"
	dataset['encoded_slotprice'] = pd.cut(dataset['slotprice'], 3, labels=[0,1,2])
	# delete slot price
	return dataset.drop('slotprice', axis=1)

# def preprocess_slotwidth(dataset):
# 	print "Preprocessing slotwidth"
# 	dataset['encoded_slotwidth'] = pd.cut(dataset['slotwidth'], 5, labels=[0,1,2,3,4])
# 	# delete slot price
# 	return dataset.drop('slotwidth', axis=1)

# def preprocess_slotheight(dataset):
# 	print "Preprocessing slotheight"
# 	dataset['encoded_slotheight'] = pd.cut(dataset['slotheight'], 5, labels=[0,1,2,3,4])
# 	# delete slot price
# 	return dataset.drop('slotheight', axis=1)


def preprocess_user_tags(dataset):
	print "Preprocessing user tags"

	data_usertag=dataset.usertag.fillna('0')
	data_usertag=data_usertag.str.replace(',',' ')
	vect=CountVectorizer()
	data_usertag_vect=vect.fit_transform(data_usertag)
	usertag=pd.DataFrame(data_usertag_vect.toarray(),columns=vect.get_feature_names())
	dataset = pd.concat([usertag, dataset], axis=1)
	return dataset.drop('usertag', axis=1)

def data_preprocessing(dataset, signal):
	# one hot encoding and preprocessing
	if signal == True:
		dataset = dataset.drop(USELESS_FEATURE, axis=1)
	else:
		dataset = dataset.drop(USELESS_FEATURE1, axis=1)
	dataset = preprocess_useragent(dataset)
	dataset = preprocess_slotprice(dataset)
	# dataset = preprocess_slotheight(dataset)
	# dataset = preprocess_slotwidth(dataset)

 
	print "Encoding data"
	columns = list(dataset.columns.values)
	for i in range(len(columns)): 
		print "Encoding for:", columns[i]
		if columns[i] not in ['usertag']:
			dataset = pd.concat([dataset, pd.get_dummies(dataset[columns[i]],prefix=columns[i], dummy_na=False, sparse=True, drop_first=True)],axis=1).drop(columns[i], axis=1)

	dataset = preprocess_user_tags(dataset)
	# dataset.to_csv('out.csv')
	return dataset

# def feature_extraction():

# def pCTR_prediction():


print "Loading data"
df_train = pd.read_csv('we_data/train.csv')
df_valid = pd.read_csv('we_data/test.csv')

df_allen = pd.read_csv('we_data/validation.csv')

print "Processing data for training set"
x_train = data_preprocessing(df_train, True)

print "Processing data for validation set"
x_valid = data_preprocessing(df_valid, False)

x_allen = data_preprocessing(df_allen, True)

# # label
y_train = df_train['click']

print "Training data"

# clf = MLPClassifier(verbose=True, hidden_layer_sizes=(300,150,50),solver='adam', alpha=0.0001)


# clf_pCTR = clf.fit(x_train, y_train)

clf = XGBClassifier(max_depth=5, silent=False, gamma=0, min_child_weight =7, colsample_bytree=0.6,
                    subsample=0.95, reg_alpha = 0.05, learning_rate = 0.1, n_estimators=100)

clf_pCTR = clf.fit(x_train, y_train)

print "Predicting data"
pctr = clf_pCTR.predict_proba(x_valid)[:, 1]
maxpctr=max(pctr)

pctr_allen = clf_pCTR.predict_proba(x_allen)[:, 1]

d_allen = pd.DataFrame()
d_allen['click'] = df_allen['click']
d_allen['pctr'] = pctr_allen

mean_pctr = d_allen.loc[d_allen['click']==1]['pctr'].mean()
print "mean pctr", mean_pctr

p = pd.DataFrame()
p['pctr'] = pctr_allen
p.to_csv('pctr.csv', index=False)

avgCTR = df_train['click'].mean()
bidmax=maxpctr*138/avgCTR*2

bid = df_valid['bidid']
bid_prices = []

for i in range(len(df_valid['bidid'])):
	if pctr[i] > mean_pctr:
		bid_price = bidmax
	else:
		bid_price = 200 * pctr[i] / avgCTR
	bid_prices.append(bid_price)

d = pd.DataFrame()
d['bidid'] = bid
d['bidprice'] = bid_prices

d.to_csv('out.csv', index=False)






# accuracy = accuracy_score(y_valid, pctr)
