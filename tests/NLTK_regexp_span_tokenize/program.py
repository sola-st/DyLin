from nltk.tokenize.util import regexp_span_tokenize

s = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
regexp_span_tokenize(s, 's') # OK


s = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
regexp_span_tokenize(s, '') # DyLin warn