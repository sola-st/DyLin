import hashlib
from argon2 import PasswordHasher
from flask import Flask, make_response, request

# Fixture for tracking insecure hashes that flow into security-sensitive cookie storage.

# source: https://codeql.github.com/codeql-query-help/python/py-weak-sensitive-data-hashing/

d = {"Insecure hash function used for storing cookies": "ObjectMarkingAnalysis",
     "configName": "weak_hash"}

def insecure_hash(input: str, salt: str):
    # SHA-256 with a plain concatenated salt is still not a password hashing scheme.
    i = input.encode('utf-8')
    s = salt.encode('utf-8')
    return hashlib.sha256(i + s).hexdigest() # BAD for passwords

def secure_hash(password: str):
    # Argon2 is used here as the "mark-clearing" secure alternative.
    ph = PasswordHasher()
    return ph.hash(password) # GOOD for passwords

app = Flask("Weak hash")

@app.route('/insecure')
def insecure():
    # Insecure flow: a weakly hashed password is written directly into a cookie.
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
    # Interesting propagation case: the inner weak hash should be neutralized by the outer secure hash.
    hashed = secure_hash(insecure_hash(password, "salt"))
    resp = make_response("")
    resp.set_cookie("password", hashed)
    return resp

@app.route('/secure')
def secure():
    # Safe flow: the cookie receives a value produced only by the secure hash function.
    password = secure_hash(request.args.get("password"))
    resp = make_response("")
    resp.set_cookie("password", password)
    return resp

# Exercise the insecure endpoint by default; the alternatives remain as manual control cases.
client = app.test_client()
response = client.get('/insecure?password=super_secret')
#response = client.get('/looks_insecure?password=super_secret')
#response = client.get('/secure?password=super_secret')
