import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('credit_card_transactions.csv')

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
#----------------------------------------

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
#-------------------------------------------------------


