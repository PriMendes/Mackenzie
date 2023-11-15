import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")




pd.set_option('display.max_columns', None)


# https://www.kaggle.com/code/iamsouravbanerjee/shopping-trends-unveiled-eda-for-beginners
df = pd.read_csv("C:\\Users\\Priscila\\Desktop\\python\\shopping_trends.csv")
print('ini')
# Verificando se existem dados nulos
print(df.isnull().sum())




print(df.columns)


df = df.drop('Item Purchased', axis=1)
df = df.drop('Category', axis=1)
df = df.drop('Location', axis=1)
df = df.drop('Color', axis=1)
df = df.drop('Shipping Type', axis=1)
df = df.drop('Discount Applied', axis=1)
df = df.drop('Frequency of Purchases', axis=1)






# Verificando se existem dados duplicados
print(df.duplicated().sum())
print('fim')
# Inicio da análise exploratória
df.info()


promo_code_count = df[(df['Promo Code Used'] == 'Yes')]['Customer ID'].count()
promo_code_count_male = df[(df['Promo Code Used'] == 'Yes') & (df['Gender'] == 'Male')]['Customer ID'].count()
promo_code_count_female = df[(df['Promo Code Used'] == 'Yes') & (df['Gender'] == 'Female')]['Customer ID'].count()
print("Number of TOTAL Customers who used Promo Code:", promo_code_count)
print("Number of male Customers who used Promo Code:", promo_code_count_male)
print("Number of female Customers who used Promo Code:", promo_code_count_female)






# Procurando por outliers
fig, ax = plt.subplots(1, 4, figsize=(12, 6))


ax[0].boxplot(df['Age'])
ax[1].boxplot(df['Purchase Amount (USD)'])
ax[2].boxplot(df['Review Rating'])
ax[3].boxplot(df['Previous Purchases'])


ax[0].set_title('Age')
ax[1].set_title('Purchase Amount (USD)')
ax[2].set_title('Review Rating')
ax[3].set_title('Previous Purchases')


plt.show()




# Contando se tem mais usuários homens ou mulheres
plt.figure(figsize=(6, 6))
counts = df["Gender"].value_counts()
counts.plot(kind = 'pie',   autopct = '%1.1f%%')
plt.xlabel('Gender',   labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()


# Contando em qual season tem mais compras
plt.figure(figsize=(6, 6))
counts = df["Season"].value_counts()
counts.plot(kind = 'pie',   autopct = '%1.1f%%')
plt.xlabel('Season',   labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()




# Histograma da idade com densidade
fig, ax = plt.subplots(figsize = (10, 5))
df['Age'].plot(kind = 'kde', color = 'red', ax = ax)
ax.hist(df['Age'], bins = 25, edgecolor = 'black', density = True)
ax.set_xlabel('Age')
ax.set_ylabel('Count / Density')
ax.set_title('Age Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
ax.set_xlim(0, 80)
plt.show()


# Histograma da preço com densidade
fig, ax = plt.subplots(figsize = (10, 5))
df['Purchase Amount (USD)'].plot(kind = 'kde', color = 'red', ax = ax)
ax.hist(df['Purchase Amount (USD)'], bins = 25, edgecolor = 'black', density = True)
ax.set_xlabel('Purchase Amount (USD)')
ax.set_ylabel('Count / Density')
ax.set_title('Purchase Amount (USD) Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
ax.set_xlim(0, 120)


plt.show()


# Número de ocorrencias nos tamanhos dos produtos
plt.figure(figsize = (10, 6))
ax = df["Size"].value_counts().plot(kind = 'bar', rot = 0)
ax.set_xticklabels(('Medium', 'Large', 'Small', 'Extra Large'))


for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1))
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Size', weight = "bold", labelpad = 20)
plt.ylabel('Number of Occurrences', labelpad = 20);
plt.show()






# Agrupando Size com Gender
grouped_data = df.groupby(['Size', 'Gender']).size().unstack().fillna(0)
ordered_sizes = ['S', 'M', 'L', 'XL']
grouped_data = grouped_data.reindex(ordered_sizes)
ax = grouped_data.plot(kind='bar',  stacked=True)
plt.xlabel('Size')
plt.ylabel('Count')
plt.title('Gender Distribution by Size')
plt.legend(title='Gender', loc='upper right')
plt.show()


# Agrupando Subscription Status com Gender
grouped_data = df.groupby(['Subscription Status', 'Gender']).size().unstack().fillna(0)
ax = grouped_data.plot(kind='bar',  stacked=True)
plt.xlabel('Subscription Status')
plt.ylabel('Count')
plt.title('Gender Distribution by Subscription Status')
plt.legend(title='Gender', loc='upper right')
plt.show()






# Agrupando Payment Method Status com Gender


grouped_data = df.groupby(['Payment Method', 'Gender']).size().unstack().fillna(0)
ax = grouped_data.plot(kind='bar',  stacked=True)
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.title('Gender Distribution by Payment Method')
plt.legend(title='Gender', loc='upper right')
plt.show()




