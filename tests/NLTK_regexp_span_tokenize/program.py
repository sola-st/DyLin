from nltk.tokenize.util import regexp_span_tokenize

# Fixture for empty regular expressions passed to regexp_span_tokenize().
# Control case: a non-empty pattern produces meaningful token spans.
s = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
regexp_span_tokenize(s, 's') # OK


# Warning case: an empty pattern can match everywhere and is treated as invalid here.
s = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
regexp_span_tokenize(s, '') # DyLin warn
