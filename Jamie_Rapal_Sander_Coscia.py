import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
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