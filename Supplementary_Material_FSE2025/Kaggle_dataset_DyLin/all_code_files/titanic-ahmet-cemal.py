#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "/kaggle/input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Görev 1
# 

# ## *Titanik veri setimizi pandas kullanarak import edelim*

# In[ ]:


df = pd.read_csv("/kaggle/input/c/titanic/train.csv")


# ## *Sütunları görüntüleyelim*

# In[ ]:


df.columns


# ## *Sayısal sütunları seçelim ve görüntüleyelim*

# In[ ]:


numerical_columns = df[['Age' ,'Fare', 'PassengerId','Survived' ,'Pclass']]
numerical_columns


# ## *Age ve Fare isimli sayısaal sütunları numpy dizisine çevirelim*

# In[ ]:



numpy_array = numerical_columns.to_numpy()
age_numpy = numpy_array[:,0]
fare_numpy = numpy_array[:,1]


# ## *Oluşturduğumuz numpy dizilerinin ortalamasını bulalım*

# In[ ]:




age_mean = np.mean(age_numpy) 
fare_mean = np.mean(fare_numpy) 



# ## *Görüldüğü gibi 'Age' sütunumuzun ortalamasını aldığımızda nan ifadesi çıkmakta. Çünkü bu sütunun içinde nan olan değerler var.Bu yüzden ortalamayı hesaplayamıyoruz. Fakat 'Fare' sütunumuzda böyle bir problem olmadığı için rahatlıkla ortalamayı buluyoruz.*

# ## *Şimdi 'Age' sütunundaki sayısal değerlerin ortalamasını bulmak için nan olan değerleri dizimizden atalım.*

# In[ ]:


age_numpy = age_numpy[~np.isnan(age_numpy)]


# ## *Şimdi yeni 'Age' dizimizin ortalamasını bulabiliriz*

# In[ ]:


age_mean = np.mean(age_numpy)


# ## *Görüldüğü gibi nan değerlerini çıkardıktan sonra rahatlıkla ortalamayı bulabildik*

# ## *Şimdi 'Age dizimizin ortalama,medyan,varyans,standart sapma, max ve min değerlerini bulalım'*

# In[ ]:


age_mean = np.mean(age_numpy)
age_median = np.median(age_numpy)
age_var = np.var(age_numpy)
age_std = np.std(age_numpy)
age_max = np.max(age_numpy)
age_min = np.min(age_numpy)

calculate_age = np.array([age_mean,age_median ,age_var, age_std, age_max ,age_min])



# ## *Şimdi 'Fare' dizimizin ortalama,medyan,varyans,standart sapma, max ve min değerlerini bulalım'*

# In[ ]:


fare_mean = np.mean(fare_numpy)
fare_median = np.median(fare_numpy)
fare_var = np.var(fare_numpy)
fare_std = np.std(fare_numpy)
fare_max = np.max(fare_numpy)
fare_min = np.min(fare_numpy)

calculate_fare = np.array([fare_mean,fare_median ,fare_var, fare_std, fare_max ,fare_min])



# ## *Şimdi 'Age' sütunumuzun üzerine 5 yıl ekleyip yeni bir numpy dizisinde saklayalım.*

# In[ ]:


age_numpy_5_year = age_numpy + 5


# ## *Şimdi 'Fare' dizimi 2 ile çarpalım*

# In[ ]:


fare_numpy_new = fare_numpy * 2


# # Görev 2:

# ## *1)Veri Setinin Genel Yapısını Anlama*

# ### *a) Veri setinin birkaç satırını görüntüleyelim*

# In[ ]:


df = pd.read_csv('/kaggle/input/c/titanic/train.csv')
df.head()


# ### *b) Sütunlar hakkında bilgi edinelim*

# In[ ]:


df.columns
df.info
df.dtypes


# ### *c) Eksik ve nan değerleri tespit edip temizleyelim.*

# In[ ]:


df.isnull().sum()
df.dropna(inplace = True)
df.isnull().sum()


# ## *2) Veri Filtreleme*

# ### * a) Cinsiyet ve hayatta kalma durumu (Survived) sütunlarını kullanarak veriyi filtreleyin. Örneğin, sadece kadın yolcuların verilerini filtreleyin.*
# 

# In[ ]:


survived_female = df[(df['Sex']== 'female') & (df['Survived'] == 1)]
survived_female.head()


# ### *b) Ücret (Fare) sütunu 50'den büyük olan yolcuları seçin.*

# In[ ]:


fare_more_than_50 = df[(df['Fare'] > 50)]
fare_more_than_50.head()


# ## *3) Veri Gruplama: (groupby)*

# ### *a) Yolcuları hayatta kalma durumuna göre gruplandırın ve her grup için ortalama yaş ve bilet ücreti hesaplayın.*

# In[ ]:


groupby_survived = df.groupby('Survived')[['Age','Fare']].mean()
groupby_survived


# ### *b) Cinsiyet ve hayatta kalma durumu (Survived) sütunlarına göre veriyi gruplandırarak, kadın ve erkek yolcular arasında hayatta kalma oranlarını karşılaştırın.*

# Bu oranı bulabilmemiz için hayatta kalan kadın ya da erkek sayısının toplam kadın ya da erkek sayısına bölünmesi gerekmekte. Ayrı ayrı hem kadın için hem erkek için hayatta kalma oranını bulalım

# In[ ]:


groupby_sex_survived = df.groupby(['Sex','Survived'])['PassengerId'].count() # Hayatta kalan ya da kalmayan erkek ve kadın sayısı


total_passengers = df.groupby('Sex')['PassengerId'].count() # Toplam erkek ve kadın sayısı

calculate_rate = groupby_sex_survived / total_passengers # hayatta kalma oranınını hesaplama



# ## *4)Yeni Sütunlar Ekleme*

# ### *a) Yeni bir sütun ekleyerek, her bir yolcunun bilet ücretinin vergi oranını hesaplayın (örneğin: Fare * 0.10).*

# In[ ]:


fare_tax_rate = df['Fare'] * 0.10

df['FareTaxRate'] = fare_tax_rate

df.head()


# ### *b) Kategorik bir sütun ekleyerek, yaşı genç (18 yaş altı) ve yetişkin olarak sınıflandırın.*

# In[ ]:


df['TeenOrAdult'] = np.where(df['Age'] > 18,'Adult','Teen')
df.head()


# # Görev 3: Matplotlib ile Veri Görselleştirme
# 

# ## *1) Çizgi Grafiği : Yaş (Age) ve ücret (Fare) sütunlarını kullanarak yolcuların yaşa göre bilet ücretini gösteren bir çizgi grafiği çizin.*

# *İlk olarak matplotlib kütüphanemizi import edelim*

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.lineplot(x=df["Age"], y=df["Fare"])


# ## *2) Saçılım Grafiği (Scatter Plot): Yaş ve ücret sütunlarını karşılaştırarak bir saçılım grafiği oluşturun. Bunu yaparken cinsiyet ile renklendirme yaparak farklı kategorileri görsel olarak ayırt edin.*

# In[ ]:


plt.scatter(df['Age'],df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')

sns.scatterplot(x='Age', y='Fare', hue='Sex', data=df,)
plt.show()


# ## *3) Histogram:*
# 
# ### *a) Yaş sütununun dağılımını inceleyin. Histogram grafiği kullanarak yolcuların yaş dağılımını görselleştirin.*

# In[ ]:


sns.histplot(data=df, x="Age")


# ### *b) Bilet ücretinin (Fare) histogramını oluşturun.*

# In[ ]:


sns.histplot(data = df, x = 'Fare',bins = 30)

df['Fare']


# ## *4) Çubuk Grafiği (Bar Plot):*
# ### *a) Hayatta kalan ve ölen yolcuların cinsiyete göre dağılımını gösteren bir çubuk grafiği çizin.*

# In[ ]:


sns.barplot(data=df,x='Sex',y = 'Survived', hue = 'Sex')

