#!/usr/bin/env python
# coding: utf-8

# ***Data Science 101: \\
#  Proyecto Final:*** \\
#  Elaborado por: Alejandro Salazar C.I.26.678.712.

# In[ ]:


#Proyecto Final:DS101
#Elaborado por: Alejandro Salazar C.I.26678712


# **Paquetes Requeridos:**
# En este bloque encontramos todos los paquetes utilizados:
# 

# In[ ]:


import math #Herramienta matematicas (Truncar)
import numpy as np #Herramientas matemáticas (Sobretodo álgebra lineal)
import statistics as stat #Herramientas para sacar estadisticos
import matplotlib.pyplot as plt #Paquete para la graficación
import pandas as pd # Paquete de procesamiento de data
from sklearn import preprocessing #Objeto para estandarizar
from sklearn.linear_model import LinearRegression #Herramientas para realizar un modelo de regresión lineal
import statsmodels.api as sm #Paquete para estadísticas del modelo de regresión
from sklearn.linear_model import LogisticRegression #Herramientas para realizar un modelo de regresión logística
from sklearn.model_selection import train_test_split #Herramienta para dividir la data
from sklearn.metrics import confusion_matrix


# **Data:** Descargamos la data aportada por la competencia desde su correspondiente ruta

# In[ ]:


#Importación de la data requerida
trainset= pd.read_csv('/kaggle/input/titanic/train.csv') #Training set para el modelo
testset= pd.read_csv('/kaggle/input/titanic/test.csv') #Testing set para el modelo


# **Exploración:** A partir de este punto realizamos una serie de pruebas sobre la data para familiarizarnos con sus características.
# Primero con el Training Set:
# 

# In[ ]:


#Reconocimiento de la data
#Training Set


# Ahora realizamos los procedimientos análogos con el Testing Set

# In[ ]:


#Testing Set


# Ahora, en vista de que estamos en un problema de clasificación binaria, en la cual se clasifica los datos en dos grupos exclusivos (En este caso, sí sobrevivieron, ó no sobrevivieron al evento del Titanic), procederemos a usar uno de los modelos más recomendado para esto, un modelo de Regresión logística. Su simplicidad recae en el hecho de que es muy similar a la regresión lineal, de hecho genera un modelo de regresión lineal para realizar su estimación, pero en lugar de estimar directamente el valor de la variable dependiente buscada, éste modelo estima la posibilidad "p" de que la variable dependiente que se busca estimar tome uno de los dos valores esperados, y procede a tratarla como una variable binomial con base en ese parámetro "p" para clasificarla (Básicamente, en realidad este proceso conlleva un trasfondo matemático más robusto). Se puede interpretar este modelo como un intento de usar una regresión lineal para un problema de clasificación binaria, pero la regresión lineal tendría el problema de que la estimación tomaría valores distintos para cada dato, pudiendo tomar valores intermedios entre los dos estados buscados (considerando tales estados como 0s y 1s, la estimación podría tomar valores como 0.5, 1.4, -3... según el dato con el que se trabaje), así que la regresión logística, usando principios matemáticos muy similares para la estimación, los clasifica directamente en dos estados (El 0 o el 1 buscado).

# Al igual que en un modelo de regresión lineal, para evaluar variables categóricas en un modelo de regresión logística, hace falta darles un peso numérico (pues, el modelo solo toma valores numéricos como parámetros), por ende, para aquellas variables categóricas que vayamos a utilizar generamos variables Dummys, que consisten en desglosar las variables categóricas con $n$ valores en $n-1$ variables booleanas, uno por cada valor, indicando si el dato toma (1) o no (0) tal valor. Algo importante a recalcar, es que uno de los valores tomados por la variable no genera una variable nueva, porque el valor de esta se puede deducir del resto (1 si el resto es 0, ó 0 de lo contrario), por lo cual si se tomara esta última en cuenta en el modelo generaría multicolinealidad entre las variables dependientes, lo cual debe evitarse para mantener la precisión más alta posible en cada estimador. 

# Generamos las correspondientes variables Dummys. Primero en el Training Set:
# 

# In[ ]:


#Generación de variables Dummys
#Training Set
train_set=pd.get_dummies(trainset,columns=["Sex","Embarked"],drop_first= True)


# Ahora en el Testing Set

# In[ ]:


#Testing Set
test_set=pd.get_dummies(testset,columns=["Sex","Embarked"],drop_first= True)


# Nótese que no tomamos en cuenta las variables categóricas "Ticket"y "Cabin", en el caso de "Ticket" porque toma demasiados valores para trabajarlos como una variable Dummy, además que es una variable que no toma valores claros en su formato, lo cual complica la obtención de información a partir de ésta (esto no implica que realizando un preprocesamiento más  en profundidad no se podría obtener información a raíz de esta variable), y prescindimos de la variable "Cabin" porque notamos una alta presencia de valores NAs en ella, más del 75% en ambos set de datas, a pesar de que quizas podría ser data de utilidad, primero tendríamos que estimar el valor de todos los valores NAs faltantes, y hacer esto a partir de menos del 25% de data original, se considera, como una práctica que puede generar información falsa o ficticia, involucrando conclusiones sesgadas.

# **Rellenado de valores NAs:**

# Training Set:

# Ante la presencia de algunos valores NA's en ciertas variables que, consideramos, podría aportar información util para el modelado, procedemos a estimar tales valores.

# Para la variable **embarked** notamos que solo hacen falta 2 datos por tomar valores, por lo cual estimarlos puede aportar más al modelo, completando los datos, que por el poco sesgo que aumenta, notamos que esta variable toma solo tres valores S, C, Q:

# In[ ]:


#Llenado de NAs
#Training Set
#Variable Embarked

trainset['Embarked']=trainset['Embarked'].fillna(stat.mode(trainset["Embarked"]))


# Y más del 70% de los datos toma el valor "S", con lo cual procedemos a asignarle dicho valor, la moda, a los valores faltantes:

# In[ ]:


trainset['Embarked']=trainset['Embarked'].fillna(stat.mode(trainset["Embarked"]))


# Para la variable **Edad** como se trata de una variable continua, cuya información podría ser de relevancia para el modelo, con un porcentaje menor de 20% de datos faltantes, que además, sería esperable, que tomara valores diferentes para cada dato faltante (no debemos asignar el mismo valor a todos los NAs), consideramos que se puede estimar mediante una regresión lineal tomando en cuenta el resto de valores, sobretodo tomando en cuenta sus correlaciones con el resto de las variables, las cuales son relativamente significativas con respecto a ,por ejemplo, las variables Parch, SibSp y Pclass (alrededor de 0.3 y 0.2, lo ideal sería superior a 0.8, pero siendo datos reales que podrían ser dependientes de varios factores, eso sería muy sorprendente, 0.3 ya lo consideramos significativo).

# In[ ]:


#Estudio de correlaciones con variables dummys


# A pesar de que notamos algunos valores no esperados en la edad que no son enteros (30.5 o valores similares); por lo pronto, no truncaremos tales valores, porque el modelo de regresión igual devolverá valores con decimales, y como se trata de una aproximación por cercanía o magnitud del resto de los valores, consideramos valioso tomar en cuenta aquellos datos que están más cerca de un valor que de otro, con lo cual los decimales nos son de utilidad. Procedemos a ajustar el modelo, para la selección de las variables empezaremos usando todas las dependientes y usaremos Selección hacia atrás para descartar aquellas que no aporten al modelo:

# In[ ]:


X_Age_tr=train_set[train_set["Age"].notna()].drop(["Name","Age","Survived","Ticket","Cabin"], axis=1)
Y_Age_tr=train_set["Age"][train_set["Age"].notna()]
regresion_lineal=LinearRegression()
regresion_lineal.fit(X_Age_tr,Y_Age_tr)


# A partir de aquí evaluamos el modelo generado y hacemos la selección hacia atrás eliminando aquellas variables que no sean significativas para el modelo, con criterio de consideración que el p-valor asociado a tal variable sean mayores a 0.05 para su eliminación (menos de 95% de significancia):

# In[ ]:


#Selección hacia atrás 1er paso: Eliminamos PassengerID
X_Age_tr=sm.add_constant(X_Age_tr) #Agregamos una columna constante al DF de data dependiente por requerimiento del paquet SM
model = sm.OLS(endog=Y_Age_tr, exog=X_Age_tr).fit()
X_Age_tr=X_Age_tr.drop(["PassengerId"],axis=1)


# In[ ]:


#Selección hacia atrás 2do paso: Eliminamos Parch
model = sm.OLS(endog=Y_Age_tr, exog=X_Age_tr).fit()
X_Age_tr=X_Age_tr.drop(["Parch"],axis=1)


# In[ ]:


#Selección hacia atrás 3er paso: Eliminamos Embarked
model = sm.OLS(endog=Y_Age_tr, exog=X_Age_tr).fit()
X_Age_tr=X_Age_tr.drop(["Embarked_Q","Embarked_S"],axis=1)


# En este paso la variable embarked cuenta con el p-valor más alto, eliminamos todas las variables dummys asociadas a ella, Embarked_Q y Embarked_S, recordando que forman parte de una variable y una sin la otra pierde sentido.
# 

# In[ ]:


#Selección hacia atrás 4to paso: Eliminamos Fare
model = sm.OLS(endog=Y_Age_tr, exog=X_Age_tr).fit()
X_Age_tr=X_Age_tr.drop(["Fare"],axis=1)


# In[ ]:


#Selección hacia atrás 5to paso: Obtenemos el modelo buscado
model = sm.OLS(endog=Y_Age_tr, exog=X_Age_tr).fit()


# En este último modelo los p-valores son practicamente nulos, con los cuales consideramos que las variables tomadas son las más significativas en base a nuestro criterio.

# Procedemos a implementar el modelo en el testing set para comparar y verificar su validez, previo a implementarlo para el rellandado de NAs.

# In[ ]:


X_Age_tr=X_Age_tr.drop("const",axis=1)
regresion_lineal=LinearRegression()
regresion_lineal.fit(X_Age_tr,Y_Age_tr)


# In[ ]:


Age_test_X=test_set[test_set["Age"].notna()][["Pclass","SibSp","Sex_male"]]
Age_Com_Test=regresion_lineal.predict(Age_test_X)
plt.scatter(test_set[test_set["Age"].notna()]["PassengerId"][0:100],test_set[test_set["Age"].notna()]["Age"][0:100],color="red")
plt.plot(test_set[test_set["Age"].notna()]["PassengerId"][0:100],Age_Com_Test[0:100])
plt.title("PassengerId vs Age test comparation")
plt.xlabel("PassengerId")
plt.ylabel("Age")
plt.show()


# En esta última gráfica la linea azul indica el valor esperado por el modelo en 100 valores de muestra, y en rojo el valor real medido de su edad, observamos que el modelo se ajusta apropiadamente para la mayoría de los valores (a excepción de algunos outliers, como es de esperar)y lo usaremos para el rellenado de NAs.

# Procedemos a rellenar los NAs. Posteriormente, en el testing set, para aplicar nuestro modelo, haremos lo  mismo si notamos en el training set que es necesario.

# In[ ]:


Relleno_Age_tr=regresion_lineal.predict(train_set[["Pclass","SibSp","Sex_male"]])
train_set["Age"]=np.where(train_set['Age'].notna(), train_set['Age'], Relleno_Age_tr)


# Ahora procederemos a preparar la data para generar el modelo de regresión logística. En este caso, como utilizaremos la edad como un factor a tomar en cuenta (será variable dependiente), consideramos que, de cara a saber si un pasajero sobrevive o no, no genera diferencia si tiene una edad exacta, o esta más cercano a un valor u otro, por lo cual, los decimales pierden relevancia; de hecho, consideramos que deben ser tratados por igual un pasajero (dato, o fila) cuya edad sea 30, que uno cuya edad sea 30.41, y así respectivamente con cada edad, ya que si ese es un factor de relevancia en la supervivencia del pasajero, seguro se deba a una consideración humana con respecto a la edad, y habitualmente el humano no diferencia los decimales en la edad, solo su parte entera, por ello truncaremos la variable "Age" para erradicar los decimales:

# In[ ]:


train_set["Age"]=train_set["Age"].apply(math.trunc)


# Ahora, separamos las variables numéricas que serán las tomadas para el modelo, y las normalizamos para mayor precisión en el modelo (aunque este último paso no es realmente necesario):

# In[ ]:


X_tr=train_set.drop(["PassengerId","Name","Survived","Ticket","Cabin"], axis=1)
sc_X=preprocessing.StandardScaler()
X_tr=sc_X.fit_transform(X_tr)


# Como el Testing Set no cuenta con la variable "Survived", no podremos probar la precisión de nuestro modelo en el, por ello tomaremos parte del training Set para probarlo (100 datos), y generaremos el modelo con el resto (791 datos, que consideramos siguen siendo suficientes para generar un modelo fiable):

# In[ ]:


X_model, X_pr, Y_model, Y_pr=train_test_split(X_tr,train_set["Survived"],test_size=100/891) 


# Ajustamos nuestro modelo de regresión logística:

# In[ ]:


regresion_logistica=LogisticRegression()
regresion_logistica.fit(X_model,Y_model)


# Ahora, implementaremos el modelo en los 100 datos tomados para evaluar su precisión:

# In[ ]:


y_pred_tr=regresion_logistica.predict(X_pr)#Predicciones
cm=confusion_matrix(Y_pr,y_pred_tr)#matriz de confusion para comparar


# En la matriz de confusión observamos que el modelo acertó en 84 datos de los 100 tomados, 59 que no sobrevivieron y 25 que sí, mientras que erró en 19, con lo cual esperamos alrededor de un poco más de 80% de precisión en la predicción del modelo, nos conformamos, y procedemos a ejecutarlo en el testing Set:

# Primero, es necesario rellenar los NAs que tiene el testing set, empezamos con la variable "Fare", que tiene un NA, observamos que la media y la mediana difieren relativamente, y que su histograma es muy asimétrico, con lo cual, es evidente que su distribución no es normal, y sería errado asignarle un valor de un estadístico de tendencia central a tal NA, porque es muy poco probable que el verdadero valor se asemeje a alguno en particular

# In[ ]:


#Testing Set
#Variable Fare
plt.hist(testset["Fare"] )
plt.title('Histograma del Testing Set')
plt.xlabel('Variable Fare')
plt.ylabel('Frecuencia')
plt.show()


# Para asignarle un valor más realista al correspondiente NA hacemos uso de la observación de que la variable "Fare" está relativamente  correlacionada con la variable "Pclass" porque se trata de una tarifa y en primera clase se tiene mayor tendencia a una mayor tarifa, como el dato con el NA se encuentra en tercera clase, supondremos que su valor de tarifa (o "Fare") sondea la media de aprox 10.11 de las tarifas en dicha clase, excluyendo los outliers de pasajeros en 3ra clase cuya tarifa es mayor a 25, que veremos en la siguiente gráfica que se trata de la minoría

# In[ ]:



testset_3rd_class=testset[testset["Pclass"]==3]
plt.hist(testset_3rd_class["Fare"] )
plt.title('Histograma del Testing Set en 3rd Class')
plt.xlabel('Variable Fare')
plt.ylabel('Frecuencia')
plt.show()
testset_3rd_class_comun=testset_3rd_class[testset_3rd_class["Fare"]<=25]
plt.hist(testset_3rd_class_comun["Fare"] )
plt.title('Histograma del Testing Set en 3rd Class sin outliers')
plt.xlabel('Variable Fare')
plt.ylabel('Frecuencia')
plt.show()
test_set['Fare']=test_set['Fare'].fillna(testset_3rd_class_comun["Fare"].mean())


# Escogiendo el valor para rellenar el NA de esta forma, aunque lo más probable es que estuviera al rededor de 7 como se ve en el histograma, al dejarlo en 10 reducimos el sesgo posible ante la posibilidad de que el verdadero valor sondeara los 15, como lo hacen varios datos de su tipo, y al tratarse de un intervalo tan reducido, la posible diferencia que podría tener con el valor tomado de 10 es menos considereble, y, por ende, relevante para el modelo. Ahora, rellenamos la variable "Age" como indicamos previamente, usando el modelo de regresión lineal ya generado:

# In[ ]:


Relleno_Age_test=regresion_lineal.predict(test_set[["Pclass","SibSp","Sex_male"]])
test_set["Age"]=np.where(test_set['Age'].notna(), test_set['Age'], Relleno_Age_test)
test_set["Age"]=test_set["Age"].apply(math.trunc)


# In[ ]:


X_test=test_set.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)#Descartamos las variables nominales
sc_Y=preprocessing.StandardScaler()
X_test=sc_Y.fit_transform(X_test)#Standarizamos el testing set

test_set["Survived"]=regresion_logistica.predict(X_test)#Predicciones



# Ahora, revisamos los valores obtenidos:

# In[ ]:




# Nuestro modelo arroja que del testing set no sobrevivieron 270 de los datos y sobrevivieron 148. Como esperamos que el modelo tenga más de 80% de precisión por nuestras pruebas, podemos conjeturar con alto grado de certeza que al menos 0.8x270=216 de los datos efectivamente no sobrevivieron, y que  al menos 118 (0.8x148=118.4) sí sobrevivieron. El modelo indica que 64.6% aprox de los datos en el Testing Set no sobrevivieron, el resto sí.

# Ahora, hechas las predicciones preparamos el archivo de subida a ser entregado con las predicciones:

# In[ ]:


submission=test_set[["PassengerId","Survived"]]
submission.to_csv('submission.csv',index=False)


# Ahora, procedemos a explorar los resultados obtenidos:

# In[ ]:


plt.hist(x=[train_set["Survived"],test_set["Survived"]],bins=[0,0.5,1],rwidth=0.4,histtype="barstacked",label=["Training Set","Testing Set"])
plt.legend()
plt.title("Survived Variable")
plt.xticks(ticks=[0.25,0.75],labels=["Not Survived","Survived"])
plt.show()
plt.pie(train_set["Survived"].value_counts(),labels=["Not Survived","Survived"],autopct="%1.1f%%")
plt.title("Trainig Set")
plt.show()
plt.pie(test_set["Survived"].value_counts(),labels=["Not Survived","Survived"],autopct="%1.1f%%")
plt.title("Testing Set")
plt.show()


# Aquí observamos que el modelo preserva las proporciones entre pasajeros que sobrevivieron y que no, y que la proporción general, como de ambos sets, es de 2 a 1, con respecto a pasajeros que no sobrevivieron contra los que sí, es decir, que aproximadamente dos tercios no lo hicieron (en el training set, dicha estadistica ya lo denota, y el modelo lo preserva de igual forma en las predicciones del testing set).

# In[ ]:


plt.hist(x=[train_set["Sex_male"],train_set[train_set["Survived"]==1]["Sex_male"]],bins=[0,0.5,1],rwidth=0.4,label=["Total","Sobrevivieron"])
plt.legend()
plt.title("Comparación por Género (Training Set)")
plt.xticks(ticks=[0.25,0.75],labels=["Mujeres","Hombres"])
plt.show()
plt.hist(x=[test_set["Sex_male"],test_set[test_set["Survived"]==1]["Sex_male"]],bins=[0,0.5,1],rwidth=0.4,label=["Total","Sobrevivieron"])
plt.legend()
plt.title("Comparación por Género (Testing Set)")
plt.xticks(ticks=[0.25,0.75],labels=["Mujeres","Hombres"])
plt.show()


# En la grafica inicial se hace notaria la gran relevancia que tomo el factor del genero para la supervivencia, mientras que las mujeres sobrevivieron en su mayoría, ni la cuarta parte de los hombres sobrevivieron, de hecho, notamos que en principio la población de hombres era practicamente el doble de la de las mujeres, y no sobrevivieron ni la mitad de los hombres con respecto a las mujeres que sí. En la segunda gráfica notamos que el algoritmo claramente le dió relevancia a este factor, y colocó en su gran mayoría que los hombres no sobrevivieron, mientras mantujo la proporción de mujeres que sí lo hicieron.

# In[ ]:


plt.hist(x=[train_set["Pclass"],train_set[train_set["Survived"]==1]["Pclass"]],bins=[0,1.1,2.2,3.1],rwidth=0.8,label=["Total","Sobrevivieron"])
plt.legend()
plt.title("Comparación por Clase de Pasajero (Training Set)")
plt.xticks(ticks=[0.5,1.5,2.5],labels=["1ra clase","2da clase", "3ra clase"])
plt.show()
plt.hist(x=[test_set["Pclass"],test_set[test_set["Survived"]==1]["Pclass"]],bins=[0,1.1,2.2,3.1],rwidth=0.8,label=["Total","Sobrevivieron"])
plt.legend()
plt.title("Comparación por Clase de Pasajero (Testing Set)")
plt.xticks(ticks=[0.5,1.5,2.5],labels=["1ra clase","2da clase", "3ra clase"])
plt.show()


# Nuevamente notamos que el modelo tomó en cuenta el factor de clase del viaje, visto que preservó sus proporciones, en la data notamos que, en cuanto a proporción en la primera clase sobrevivió la mayoría, mientras que en segunda clase un porcentaje ligeramente inferior a la mitad, y de tercera clase la mayoría no sobrevivió, sin embargo, en cuanto a cantidad, en las 3 clases sobrevivieron la misma cantidad de personas, notando que en la 3ra clase les perjudicó su alta masa de población. Parece que no se dió prioridad por clases, a pesar de que esto perjudica gravemente a los de tercera clase, que eran mayoría, mientras que garantizaba que la mayoría de primera clase sobrevivían.

# In[ ]:


Grupo_edad=np.where(train_set['Age']<13, 1, 2)#1=niño,2=adolescente,3=adulto
Grupo_edad=np.where(train_set['Age']>17, 3, Grupo_edad)
Grupo_edad_test=np.where(test_set['Age']<13, 1, 2)#1=niño,2=adolescente,3=adulto
Grupo_edad_test=np.where(test_set['Age']>17, 3, Grupo_edad_test)
plt.hist([Grupo_edad,Grupo_edad[train_set["Survived"]==1]],bins=[0,1.1,2.1,3.1],rwidth=0.8,label=["Total","Sobrevivieron"])
plt.legend()
plt.title("Comparación por Edad (Training Set)")
plt.xticks(ticks=[0.5,1.5,2.5],labels=["Niños","Adolescente", "Adulto"])
plt.show()
plt.hist([Grupo_edad_test,Grupo_edad_test[test_set["Survived"]==1]],bins=[0,1.1,2.1,3.1],rwidth=0.8,label=["Total","Sobrevivieron"])
plt.legend()
plt.title("Comparación por Edad (Testing Set)")
plt.xticks(ticks=[0.5,1.5,2.5],labels=["Niños","Adolescente", "Adulto"])
plt.show()


# Finalmente, notamos que entre niños y adolescentes sobrevivieron en proporciones similares a los adultos, 1/3 en cada caso aproximadamente, a vista por la gráfica, y que la gran mayoría de la población era adulta. Estas últimas tablas no son del todo confiable debido a que muchas de las edades no fueron obtenidas por los datos originales, sino generadas mediante un modelo para rellenar tales datos faltantes, y el modelo clasificó en su mayoría a los datos como adultos, a pesar de que el modelo presentaba un alto nivel de significancia, estas tablas no están sujutas a los valores reales, y por eso presentan un sesgo mayor al de las anteriores. En cualquier caso, notamos que el modelo preserva las proporciones, y no arroja conclusiones inesperadas para el caso del estudio por edades.
