import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('credit_card_transactions.csv')
df = df.head(1296675)

#------------------------------
#-------- Task 2 --------------
#------------------------------

#-------Group by Transaction Category------

merchant_summary = df.groupby('category').agg(
    Total_Amount = ('amt', 'sum'),
    Transaction_Count = ('amt', 'count')
).reset_index()

merchant_summary = merchant_summary.sort_values(by='Transaction_Count', ascending = False)

plt.figure(figsize=(12, 6))
sns.barplot(x = 'Transaction_Count', y = 'category', data=merchant_summary, palette= 'magma')
plt.title('Total Transaction Amount by Category')
plt.xlabel('Total Number of Transactions')
plt.ylabel('Merchant category')

for index, value in enumerate(merchant_summary['Transaction_Count']):
    plt.text(value + 5, index, f'{value}', va='center')

plt.tight_layout()
plt.show()

#---- Group by Transaction Sum Total by Category------

merchant_summary = merchant_summary.sort_values(by='Total_Amount', ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x = 'Total_Amount', y = 'category', data = merchant_summary, palette = 'viridis')
plt.title('Total Transaction Sums by Category')
plt.xlabel('Total Amount spent')
plt.ylabel('Merchant Category')

for index, value in enumerate(merchant_summary['Total_Amount']):
    plt.text(value + 5, index, f'{value:.2}', va='center')

plt.tight_layout()
plt.show()

#---------Analyzing Travel Spending (2019 only)-----------------------
travel_df = df[df['category'] == 'travel'].copy()

travel_df['trans_date_trans_time'] = pd.to_datetime(travel_df['trans_date_trans_time'])

travel_df = travel_df[travel_df['trans_date_trans_time'] < '2020-01-01'] #No full 2020 year end data so cut it out

travel_df['Month'] = travel_df['trans_date_trans_time'].dt.to_period('M').astype(str)

travel_summary = travel_df.groupby('Month').agg(
    Total_Amount = ('amt', 'sum'),
    Transaction_Count = ('amt', 'count')
).reset_index()

plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
sns.lineplot(data=travel_summary, x='Month', y='Total_Amount', marker='o')
plt.title('Total Transaction Amount for Travel Category (Month by Month)')
plt.xlabel('Month')
plt.ylabel('Total Amount')

plt.subplot(2, 1, 2)
sns.lineplot(data=travel_summary, x='Month', y='Transaction_Count', marker='o', color='orange')
plt.title('Transaction Count for Travel Category (Month by Month)')
plt.xlabel('Month')
plt.ylabel('Transaction Count')

plt.tight_layout()
plt.show()

#--------------Task 4------------------
#-----------Data Cleaning--------------

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
df['transaction_day'] = df['trans_date_trans_time'].dt.day
df['transaction_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['transaction_month'] = df['trans_date_trans_time'].dt.month

df = df.dropna()

#function that calculates the distnace with lat & long
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

#------------K Nearest Neighbors Classifier--------------------
df = pd.get_dummies(df, columns = ['category'])

df = df.drop(columns=['Unnamed: 0', 'trans_num', 'first', 'last', 'gender', 'unix_time', 'job', 'dob', 'merchant', 
                      'trans_date_trans_time', 'street', 'city', 'state', 'city_pop', 'street', 'zip', 'job', 'trans_num', 
                      'merch_zipcode'])
#Added Distance to dataset to try and boost performance
df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

#For making the sets even
'''
fraud = df[df['is_fraud'] == 1]
nonfraud = df[df['is_fraud'] == 0]

nonfraud = nonfraud.sample(n = fraud.shape[0], random_state=42)

balanced_data = pd.concat([fraud, nonfraud])
balanced_data = balanced_data.sample(frac=1, random_state=42)

print(balanced_data.shape)

x = balanced_data.drop(columns = ['is_fraud'])
y = balanced_data['is_fraud']
'''

#Default Classifier
x = df.drop(columns = ['is_fraud'])
y = df['is_fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(classification_report(y_test, y_pred))
#--------------------------------------------------

#----------Random Forest Classifier----------------
rf = RandomForestClassifier(n_estimators = 250, random_state = 42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print(classification_report(y_test, y_pred_rf))