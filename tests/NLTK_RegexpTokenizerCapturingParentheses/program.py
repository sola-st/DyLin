from nltk.tokenize import RegexpTokenizer


RegexpTokenizer(r'(?:\w+)|(?:[^\w\s]+)') # OK



RegexpTokenizer(r'(\w+)|([^\w\s]+)') # DyLin warn