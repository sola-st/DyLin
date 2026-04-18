from requests import Session

# Fixture for Session.post() calls that should receive binary file handles.
# Create a local file so the session-based upload examples have concrete input data.
with open('test.txt', 'w') as f:
    f.write('test')

# Control case: Session.post() receives a binary stream, which matches the expected request body type.
with open('test.txt', 'rb') as f:
    session = Session()
    s = session.post('https://github.com/', data=f) # OK

# Warning case: a text-mode handle is passed through the session API instead of raw bytes.
with open('test.txt', 'r') as f:
    session = Session()
    s = session.post('https://github.com/', data=f) # DyLin warn
