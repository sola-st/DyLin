import hashlib
from argon2 import PasswordHasher
from flask import Flask, make_response, request

# source: https://codeql.github.com/codeql-query-help/python/py-weak-sensitive-data-hashing/

d = {"Insecure hash function used for storing cookies": "ObjectMarkingAnalysis",
     "configName": "weak_hash"}

def insecure_hash(input: str, salt: str):
    i = input.encode('utf-8')
    s = salt.encode('utf-8')
    return hashlib.sha256(i + s).hexdigest() # BAD for passwords

def secure_hash(password: str):
    ph = PasswordHasher()
    return ph.hash(password) # GOOD for passwords

app = Flask("Weak hash")

@app.route('/insecure')
def insecure():
    password = request.args.get("password")
    hashed = insecure_hash(password, "salt")
    resp = make_response("")
    f'START;'
    resp.set_cookie("password", hashed)
    f'END; used insecure hash function'
    return resp

@app.route('/looks_insecure')
def looks_insecure():
    password = request.args.get("password")
    # interesting case:
    # hash returns marked value, secure_hash function
    # removes mark from return value internally
    # goal: if instrumented methods overwrite markings
    #       internally, use those and don't union as per default

    # other way around TODO
    hashed = secure_hash(insecure_hash(password, "salt"))
    resp = make_response("")
    resp.set_cookie("password", hashed)
    return resp

@app.route('/secure')
def secure():
    password = secure_hash(request.args.get("password"))
    resp = make_response("")
    resp.set_cookie("password", password)
    return resp

client = app.test_client()
response = client.get('/insecure?password=super_secret')
#response = client.get('/looks_insecure?password=super_secret')
#response = client.get('/secure?password=super_secret')