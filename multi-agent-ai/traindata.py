import csv
import numpy as np
from math import *
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


train=pd.read_csv('train.csv')

#print the number of impressions
count=pd.value_counts(train['advertiser'])
print("the number of the impression of each adv:\n","advertiser","impressions\n",count)
print("\n")

#print the number of clicks
clicks=train[train["click"]==1]
clickNum=pd.value_counts(clicks["advertiser"])
print("the number of clicks of each adv:\n","advertiser","clicks\n",clickNum)
print("\n")

#cost of each advertiser
cost=train.groupby(by=["advertiser"])["payprice"].sum()
print("the cost of each adv:\n",cost)
print("\n")

#CTR of each advertiser
CTR=clickNum/count
print("the CTR of each advertiser:\n",CTR)
print("\n")


#average CPM (cost per impressions)(unit:feng)
CPM=cost/count * 1000
print("the avg CPM of each advertiser:\n ", CPM)
print("\n")


#eCPC (cost per click) (unit:feng)
eCPC= cost/clickNum
print("the eCPC of each advertiser:\n", eCPC)



#User Feedback
# 1. CTR per day of the week for advertiser 1458 and 3358

daily_CTR = pd.DataFrame()
daily_CTR['day'] = np.sort(train.weekday.unique())

#the impression of 1458 in each weekday
imp_1458 = train.groupby('weekday').advertiser.value_counts()
daily_CTR['imps_1458'] = imp_1458.iloc[imp_1458.index.get_level_values('advertiser') == 1458].values

#the impression of 3358 in each weekday
imp_3358 = train.groupby('weekday').advertiser.value_counts()
daily_CTR['imps_3358'] = imp_3358.iloc[imp_3358.index.get_level_values('advertiser') == 3358].values

#the click value of 1458 in each weekday
click = train.groupby(['advertiser','weekday']).click.value_counts()
clickall_1458 = click.iloc[click.index.get_level_values('advertiser') == 1458]
daily_CTR['clicks_1458'] = clickall_1458.iloc[clickall_1458.index.get_level_values('click') == 1].values

#the click value of 3358 in each weekday
clickall_3358 = click.iloc[click.index.get_level_values('advertiser') == 3358]
daily_CTR['clicks_3358'] = clickall_3358.iloc[clickall_3358.index.get_level_values('click') == 1].values

daily_CTR['CTR_1458'] = ((daily_CTR.clicks_1458 / daily_CTR.imps_1458) ).round(5)
daily_CTR['CTR_3358'] = ((daily_CTR.clicks_3358 / daily_CTR.imps_3358) ).round(5)

a = np.mean(daily_CTR['CTR_1458'])
b = np.mean(daily_CTR['CTR_3358'])

#standard value of ctr of 1458 and 3358
daily_CTR['std_CTR_1458'] = np.sqrt( (daily_CTR['CTR_1458'] - a)**2 )
daily_CTR['std_CTR_3358'] = np.sqrt((daily_CTR['CTR_3358'] - b)**2 )


#daily_CTR.to_csv("dailyCTR", sep='\t')


# lot CTR for each weekday of advertiser 1458 and 3358
f, ax = plt.subplots(1)
ax.plot(1+daily_CTR.day.values, daily_CTR.CTR_1458.values, marker = '.',color = 'red', label='1458')
ax.plot(1+daily_CTR.day.values, daily_CTR.CTR_3358.values,marker='.', color='black',label='3358')

plt.legend()
plt.ylabel('CTR')
plt.xlabel('weekdays')
plt.title('CTR for each weekday of advertiser 1458 & 3358')
ax.set_xlim(xmin = 0.5 , xmax = 8)
f.set_size_inches(8,6)
plt.draw()
plt.show()



# 2. Analyzing CTR per hour
hour_CTR = pd.DataFrame()

hour_CTR['hour'] = np.sort(train.hour.unique())

#impression of 1458 per hour
imp_1458 = train.groupby('hour').advertiser.value_counts()
hour_CTR['imps_1458'] = imp_1458.iloc[imp_1458.index.get_level_values('advertiser') == 1458].values

#impression of 3358 per hour
imp_3358 = train.groupby('hour').advertiser.value_counts()
hour_CTR['imps_3358'] = imp_3358.iloc[imp_3358.index.get_level_values('advertiser') == 3358].values

#toctal click of 1458 in
click = train.groupby(['advertiser','hour']).click.value_counts()
clickall_1458 = click.iloc[click.index.get_level_values('advertiser') == 1458]
hour_CTR['clicks_1458'] = clickall_1458.iloc[clickall_1458.index.get_level_values('click') == 1].values

clickall_3358 = click.iloc[click.index.get_level_values('advertiser') == 3358]
click3358= list(clickall_3358.iloc[clickall_3358.index.get_level_values('click') == 1].values)
click3358=click3358[:4] + [0] + click3358[4:]
click3358=click3358[:6] + [0] + click3358[6:]
hour_CTR['clicks_3358'] = click3358

hour_CTR['CTR_1458'] = ((hour_CTR.clicks_1458 / hour_CTR.imps_1458) ).round(5)
hour_CTR['CTR_3358'] = ((hour_CTR.clicks_3358 / hour_CTR.imps_3358) ).round(5)

#hour_CTR.to_csv("hourlyCTR", sep='\t')


# lot CTR graph for each hour of advertiser 1458 and 3358
f, ax = plt.subplots(1)
ax.plot(1+hour_CTR.hour.values, hour_CTR.CTR_1458.values, marker = '.',color = 'red', label='1458')
ax.plot(1+hour_CTR.hour.values, hour_CTR.CTR_3358.values,marker='.', label='3358')
plt.xticks(1+hour_CTR.hour.values)
plt.legend(loc=2)
plt.ylabel('CTR')
plt.xlabel('hours')
plt.title('CTR for each hour of advertiser 1458 & 3358')
ax.set_xlim(xmin = 0.5 , xmax = 24.5)
f.set_size_inches(8,6)
plt.draw()
plt.show()


# 3. Analyzing CTR per region
region_CTR = pd.DataFrame()

region_CTR['region'] = np.sort(train.region.unique())

#impression group by region 1458
imp_1458 = train.groupby('region').advertiser.value_counts()
region_CTR['imps_1458'] = imp_1458.iloc[imp_1458.index.get_level_values('advertiser') == 1458].values

#impression group by region 3358
imp_3358 = train.groupby('region').advertiser.value_counts()
region_CTR['imps_3358'] = imp_3358.iloc[imp_3358.index.get_level_values('advertiser') == 3358].values

#total clicks and CTR for 1458 in each region
click = train.groupby(['advertiser','region']).click.value_counts()
clickall_1458 = click.iloc[click.index.get_level_values('advertiser') == 1458]
clickall_1458.iloc[clickall_1458.index.get_level_values('click') == 0]=0
region_CTR['clicks_1458'] = clickall_1458.groupby(level='region').sum().values

#total clicks and CTR for 3358 in each region
clickall_3358 = click.iloc[click.index.get_level_values('advertiser') == 3358]
clickall_3358.iloc[clickall_3358.index.get_level_values('click') == 0]=0
region_CTR['clicks_3358'] = clickall_3358.groupby(level='region').sum().values


region_CTR['CTR_1458'] = ((region_CTR.clicks_1458 / region_CTR.imps_1458) ).round(5)
region_CTR['CTR_3358'] = ((region_CTR.clicks_3358 / region_CTR.imps_3358) ).round(5)

#region_CTR.to_csv("regionCTR", sep='\t')



# lot CTR graph for each region of advertiser 1458 and 3358
dataframes = []
advs=[3358,1458]
df = None
for adv in advs:
    df = train[train['advertiser'] == adv]
    ctr = df.groupby('region').agg({'click': {'click': sum}, 'bidid': {'imps': 'count'}})
    ctr.columns = ctr.columns.droplevel(0)
    ctr['ctr'] = (ctr.click / ctr.imps) * 100
    dataframes.append(ctr)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
x = np.arange(len(df.region.unique()))
width = 0.35

p1 = ax.bar(x, dataframes[0]['ctr'], width, color="lightblue", label=str(advs[0]))
p2 = ax.bar(x, dataframes[1]['ctr'], width, color="steelblue", bottom=dataframes[0]['ctr'], label=str(advs[1]))
ax.set_xlabel('Regions')
ax.set_ylabel('CTR')
ax.set_xticks(x + width / 2.)
ax.legend(loc='upper center')
plt.title('CTR for each region of advertiser 1458 & 3358')
l = list(df.region.unique())
l.sort()
ax.set_xticklabels(l, rotation=45)

plt.show()



# 4. Analyzing CTR per ad exchange
adexchange_CTR = pd.DataFrame()

adexchange_CTR['ad_exchange'] = np.sort(train.adexchange.unique())
adexchange_CTR = adexchange_CTR.drop([3,4])
imp_1458 = train.groupby('adexchange').advertiser.value_counts()
adexchange_CTR['imps_1458'] = imp_1458.iloc[imp_1458.index.get_level_values('advertiser') == 1458].values

imp_3358 = train.groupby('adexchange').advertiser.value_counts()
adexchange_CTR['imps_3358'] = imp_3358.iloc[imp_3358.index.get_level_values('advertiser') == 3358].values

#total clicks of 1458 for each adexchange
click = train.groupby(['advertiser','adexchange']).click.value_counts()
clickall_1458 = click.iloc[click.index.get_level_values('advertiser') == 1458]
clickall_1458.iloc[clickall_1458.index.get_level_values('click') == 0]=0
adexchange_CTR['clicks_1458'] = clickall_1458.groupby(level='adexchange').sum().values

#total clicks of 3358 for each adexchange
clickall_3358 = click.iloc[click.index.get_level_values('advertiser') == 3358]
clickall_3358.iloc[clickall_3358.index.get_level_values('click') == 0]=0
adexchange_CTR['clicks_3358'] = clickall_3358.groupby(level='adexchange').sum().values

adexchange_CTR['CTR_1458'] = ((adexchange_CTR.clicks_1458 / adexchange_CTR.imps_1458) ).round(5)
adexchange_CTR['CTR_3358'] = ((adexchange_CTR.clicks_3358 / adexchange_CTR.imps_3358) ).round(5)

#adexchange_CTR.to_csv("adexchangeCTR", sep='\t')

# lot CTR graph for each adexchange of advertiser 1458 and 3358
f, ax = plt.subplots(1)
ax.plot(adexchange_CTR.ad_exchange.values, adexchange_CTR.CTR_1458.values, marker = '.',color = 'red', label='1458')
ax.plot(adexchange_CTR.ad_exchange.values, adexchange_CTR.CTR_3358.values,marker='.',color = 'black', label='3358')
plt.xticks(adexchange_CTR.ad_exchange.values)
plt.legend(loc=2)
plt.ylabel('CTR')
plt.xlabel('adexchange')
plt.title('CTR for each adexchange of advertiser 1458 & 3358')
ax.set_xlim(xmin = 0.5 , xmax = 4)
f.set_size_inches(8,6)
plt.draw()
plt.show()


#slot size
advs = [1458, 3358]
dataframes = []
df = None
for adv in advs:
    df = train[train['advertiser'] == adv]
    ctr = df.groupby(('slotwidth','slotheight'), as_index=False).agg({'click':{'click':sum}, 'bidid': {'imps' : 'count'}})
    ctr.columns = ctr.columns.droplevel(0)
    ctr['ctr'] = (ctr.click / ctr.imps) * 100
    ctr.columns = ['slotwidth','slotheight','imps','click','ctr']
    dataframes.append(ctr)

new_df = pd.merge(dataframes[0], dataframes[1], how='outer', on=['slotwidth','slotheight'])
new_df = new_df.fillna(0)

labels = list(new_df[['slotwidth','slotheight']].values)
labels_str = [str(l[0]) + 'x' + str(l[1]) for l in labels]


fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
x = np.arange(len(new_df.slotwidth))
width = 0.3

p1 = ax.bar(x, new_df['ctr_x'], width, color= "lightblue", label=advs[0])
p2 = ax.bar(x, new_df['ctr_y'], width, color="steelblue", bottom=new_df['ctr_x'], label=advs[1])
ax.set_xlabel('Slot Size')
ax.set_ylabel('CTR')
ax.set_xticks(x + width/2.)
plt.title('CTR for each slot size of advertiser 1458 & 3358')
ax.legend()
ax.set_xticklabels(labels_str, rotation=45)

plt.show()