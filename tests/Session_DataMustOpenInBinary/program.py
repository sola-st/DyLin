from requests import Session

with open('test.txt', 'w') as f:
    f.write('test')

with open('test.txt', 'rb') as f:
    session = Session()
    s = session.post('https://github.com/', data=f) # OK

with open('test.txt', 'r') as f:
    session = Session()
    s = session.post('https://github.com/', data=f) # DyLin warn