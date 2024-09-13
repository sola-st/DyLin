#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Görev 1: NumPy ile Temel İstatistiksel Analizler
# 
# ## Veri Setini Yükleme 
# Titanic veri setini Pandas kullanarak yükleyin.

# In[ ]:


df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()


# ## NumPy Dizilerine Çevirme: 
# Titanic veri setindeki sayısal sütunları NumPy dizisine çevirin. Örnek sütunlar: Age, Fare.

# In[ ]:


age_np = np.array(df["Age"])
fare_np = np.array(df["Fare"])
age_np[:10]


# Mean, median gibi hesapmaları yapabilmek için nan değerleri çıkaralım:

# In[ ]:


age_np = age_np[~np.isnan(age_np)]
fare_np = fare_np[~np.isnan(fare_np)]
age_np[:10]


# ## İstatistiksel Analiz
# 
# Yaş (Age) ve bilet ücreti (Fare) sütunlarının ortalamasını, medyanını, standart sapmasını ve varyansını NumPy kullanarak hesaplayın. Bu sütunlarda minimum ve maksimum değerleri bulun.

# In[ ]:


np.mean(age_np), np.median(age_np), np.std(age_np), np.var(age_np)


# In[ ]:


np.mean(fare_np), np.median(fare_np), np.std(fare_np), np.var(fare_np)


# ## Matematiksel İşlemler
# 
# Yaş sütunu üzerinde 5 yıl ekleyin ve yeni bir NumPy dizisi oluşturun. Bilet Ücretleri sütununu 2 ile çarpın ve sonucu yeni bir NumPy dizisinde saklayın.

# In[ ]:


age_plus_5 = np.array(df["Age"] + 5)
fare_times_2 = np.array(df["Fare"] * 2)
age_plus_5[:10], fare_times_2[:10]


# # Görev 2: Pandas ile Veri Manipülasyon
# 
# ## Veri Setinin Genel Yapısını Anlama:
# 
#     a. Veri setinin ilk birkaç satırını görüntüleyin
# 
#     b. Veri setindeki sütunlar hakkında bilgi edinin 
# 
#     c. Eksik veri (NaN) olan sütunları tespit edin ve bu eksik verileri temizleyin
# 

# In[ ]:


# a
df.head()


# In[ ]:


# b
df.info()


# In[ ]:


# c
df.isnull().sum()


# Sadece yaş ve kabin sutünları eksik bilgi içeriyor. Impute de edebilirdik fakat şimdili çıkarmak yeterli

# In[ ]:


# inplace=True veri setini direkt değiştirir
df.dropna(axis=0, inplace=True)


# In[ ]:


# temizlemiş miyiz diye kontrol et
df.isnull().sum()


# In[ ]:


df.info()


# Örneklerimiz 891'den 183'e düştü. Bu sebeple eksik bilgileri atmak çoğu zaman yanlış bir harekettir.

# ## Veri Filtreleme
# 
#     a. Cinsiyet ve hayatta kalma durumu (Survived) sütunlarını kullanarak veriyi filtreleyin. Örneğin, sadece kadın yolcuların verilerini filtreleyin.
# 
#     b. Ücret (Fare) sütunu 50'den büyük olan yolcuları seçin.
# 

# In[ ]:


# a 
# hem hayatta kalıp hem de kadın olanlar
df[(df.Survived == 1) & (df.Sex == "female")]


# In[ ]:


# b
# iki yolu vardır. birincisi:
df[df.Fare > 50]


# In[ ]:


# ikincisi
df.query("Fare > 50")


# ## Veri Gruplama: (groupby)
# 
#     a. Yolcuları hayatta kalma durumuna göre gruplandırın ve her grup için ortalama yaş ve bilet ücreti hesaplayın.
# 
#     b. Cinsiyet ve hayatta kalma durumu (Survived) sütunlarına göre veriyi gruplandırarak, kadın ve erkek yolcular arasında hayatta kalma oranlarını karşılaştırın.
# 

# In[ ]:


# a
df.groupby("Survived")[["Age", "Fare"]].mean()


# In[ ]:


# b
df.groupby(["Sex", "Survived"])["Survived"].count()


# ## Yeni Sütunlar Ekleme
# 
#     a. Yeni bir sütun ekleyerek, her bir yolcunun bilet ücretinin vergi oranını hesaplayın (örneğin: Fare * 0.10).
# 
#     b. Kategorik bir sütun ekleyerek, yaşı genç (18 yaş altı) ve yetişkin olarak sınıflandırın.

# In[ ]:


# a
df["Fare_Tax"] = df["Fare"] * 0.10
df.head(3)


# In[ ]:


# b
df["Age_Group"] = df["Age"].apply(lambda x: "Adult" if x >= 18 else "Child")
df.head()


# # Görev 3: Matplotlib ile Veri Görselleştirme
# 
# ## Çizgi Grafiği
# 
# Yaş (Age) ve ücret (Fare) sütunlarını kullanarak yolcuların yaşa göre bilet ücretini gösteren bir çizgi grafiği çizin.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(x=df["Age"], y=df["Fare"])


# ## Saçılım Grafiği (Scatter Plot)
# 
# Yaş ve ücret sütunlarını karşılaştırarak bir saçılım grafiği oluşturun. Bunu yaparken cinsiyet ile renklendirme yaparak farklı kategorileri görsel olarak ayırt edin.
# 

# In[ ]:


sns.scatterplot(x="Age", y="Fare", hue="Sex", data=df)


# ## Histogram
# 
#     a. Yaş sütununun dağılımını inceleyin. Histogram grafiği kullanarak yolcuların yaş dağılımını görselleştirin.
# 
#     b. Ayrıca bilet ücretinin (Fare) histogramını oluşturun.
# 

# In[ ]:


sns.histplot(df["Age"], bins=30)


# In[ ]:


sns.histplot(df["Fare"], bins=50)


# ## Çubuk Grafiği (Bar Plot)
# 
# Hayatta kalan ve ölen yolcuların cinsiyete göre dağılımını gösteren bir çubuk grafiği çizin.

# In[ ]:


# bunu yapamamıştım kopya çektim
sns.barplot(x='Sex', y='Survived', hue='Survived', data=df, estimator=lambda x: len(x))


# ## Alt Grafikler (Subplots)
# 
# Tek bir figürde birden fazla grafik gösterin. Örneğin, çizgi ve histogram grafiğini aynı figürde yan yana gösterin.

# In[ ]:


# bunu da yapamadım
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

sns.lineplot(x=df["Age"], y=df["Fare"], ax=ax1)
ax1.set_title('Yaş ve Bilet Ücreti')

sns.histplot(df["Age"], bins=30, ax=ax2)
ax2.set_title('Yaş')

plt.tight_layout()

plt.show()


# # Ekstra Görev (Araştırınız)
# **Seaborn ile Gelişmiş Görselleştirme**: Seaborn kütüphanesini kullanarak bir heatmap ile korelasyon matrisi oluşturun ve korelasyon matrisinden çıkardığınız sonuçları açıklayan metin yazınız (jupyter notebook üzerine).

# In[ ]:


# tüm veri tiplerini görelim
set(df.dtypes)


# In[ ]:


# sadece nümerik olanlarını seçelim
df_num = df.select_dtypes(include=['int64', 'float64'])
df_num.head()


# In[ ]:


df_num.drop(["PassengerId", "Fare_Tax"],axis=1, inplace=True)
df_num.head()


# In[ ]:


# corelasyon matrisi
corr_matrix = df_num.corr()

# heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True)


# ## Açıklama
# 
# Korelasyon sayısı 1'e akın olanlar pozitif ilişkilidir - biri arttıkça diğeri de artar.
# 
# Korelasyonu -1'e yakın olanlar negatif ilişkilidir.
# 
# 0'a yakın olanların arasında bir ilişki yoktur.
# 
# * **Survived - Fare**: Bir yolcunun hayatta kalma olasılığı verdiği bilet ücretiyle zayıf da olsa pozitif bir korelasyonu var. (daha çok para verenlerin kurtarılması öncelikli görülmüş)
# 
# * **Survived - Parch**: Bir yolcunun yanında ebeveyn veya çocuklarının olmasıyla hayatta kalması korela değildir (0'a yakın)
# 
# * **Survived - SibSp**: Bir yolcunun hayatta kalmasıyla yanında kardeş veya eşinin olması doğru orantılı
# 
# * **Survived - Age**: Yolcunun yaşıyla hayatta kalışı negatif orantılı - bu demek oluyor ki daha genç olanların kurtarılmış olması daha olasılı.
# 
# * **Survivived - PClass**: Yolcunun hayatta kalmasıyla sosyoekonomik sınıfı ilişkili değil.

# In[ ]:




