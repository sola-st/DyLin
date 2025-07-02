from requests import Session

s = Session()


s.mount('https://youtube.com/', None) # OK


s.mount('https://github.com', None) # DyLin warn


s.mount('https://google.com', None) # DyLin warn
