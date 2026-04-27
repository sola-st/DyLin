from requests import Session

# Fixture for Session.mount hostnames that omit the required trailing slash.
s = Session()

# Valid mount prefixes end with "/" so requests can match the mounted adapter correctly.
s.mount('https://youtube.com/', None) # OK

# Missing the trailing slash changes the prefix semantics and should be flagged.
s.mount('https://github.com', None) # DyLin warn

# Another invalid prefix variant to confirm the rule is not host-specific.
s.mount('https://google.com', None) # DyLin warn
