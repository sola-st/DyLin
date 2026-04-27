import requests

# Fixture for requests APIs that should receive file handles opened in binary mode.

# Create a small file so the fixture can exercise both upload modes locally.
with open('test.txt', 'w') as f:
    f.write('test')


# Control case: requests receives a binary stream, which is the expected upload form.
with open('test.txt', 'rb') as f:
    s = requests.post('https://github.com/', data=f) # OK


# Warning case: passing a text-mode stream can corrupt byte-oriented request payload handling.
with open('test.txt', 'r') as f:
    s = requests.post('https://github.com/', data=f) # DyLin warn
