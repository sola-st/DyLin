import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import pandas
from sklearn.feature_extraction.text import CountVectorizer

text = [
    "the house had a tiny little mouse",
    "the cat saw the mouse",
    "the mouse ran away from the house",
    "the cat finally ate the mouse",
    "the end of the mouse story",
]
y = [True, False, False, True, True]
# unknown words in test data leak into training data
wordsVectorizer = CountVectorizer().fit(text)
# transforms into a matrix of token counts
wordsVector = wordsVectorizer.transform(text)
invTransformer = TfidfTransformer().fit(wordsVector)
# normalize count matrix
invFreqOfWords = invTransformer.transform(wordsVector)
X = pandas.DataFrame(invFreqOfWords.toarray())
# split training and test data
train, test, spamLabelTrain, spamLabelTest = train_test_split(X, y, test_size=0.5)  # DyLin warn


# Second test case, Figure 1 in Yang et. al
# generate random data
n_samples, n_features, n_classes = 200, 10000, 2
rng = np.random.RandomState(42)
X = rng.standard_normal((n_samples, n_features))
y = rng.choice(n_classes, n_samples)

# leak test data through feature selection
X_selected = SelectKBest(k=25).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, random_state=42)  # DyLin warn

gbc = GradientBoostingClassifier(random_state=1)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)

accuracy_score(y_test, y_pred)
# expected accuracy ~0.5; reported accuracy 0.76
