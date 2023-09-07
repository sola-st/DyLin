d = { "Side Effects Dunder Methods": "SideEffectsDunderAnalysis"}

global g
g = 0
class UsesSelf():
    def __init__(self):
        self.a = ""
        self.b = 1
        self.c = []
    def __eq__(self, other):
        f'START;'
        self.a = "1"
        f'END; __eq__ self.a = "1"'
        f'START;'
        self.b = 2
        f'END; __eq__ self.b = 2'
        f'START;'
        self.c = ["2"]
        f'END; __eq__ self.c = ["2"]'
        d = 2
        e = ""
        f = []
        return True
    def __abs__(self):
        f'START;'
        self.a = "1"
        f'END; __abs__ self.a = "1"'
        f'START;'
        self.b = 2
        f'END; __abs__ self.b = 2'
        f'START;'
        self.c = ["2"]
        f'END; __abs__ self.c = ["2"]'
        d = 2
        e = ""
        f = []
        return 1
    def __hash__(self):
        f'START;'
        self.a = "1"
        f'END; __hash__ self.a = "1"'
        f'START;'
        self.b = 2
        f'END; __hash__ self.b = 2'
        f'START;'
        self.c = ["2"]
        f'END; __hash__ self.c = ["2"]'
        d = 2
        e = ""
        f = []
        return 1
    def __len__(self):
        f'START;'
        self.a = "1"
        f'END; __len__ self.a = "1"'
        f'START;'
        self.b = 2
        f'END; __len__ self.b = 2'
        f'START;'
        self.c = ["2"]
        f'END; __len__ self.c = ["2"]'
        d = 2
        e = ""
        f = []
        return 1
class UsesGlobal():
    def __init__(self):
        global g
        g = 2
        d = 2
        e = ""
        f = []
    def __eq__(self, other):
        global g
        f'START;'
        g = 2
        f'END; __eq__ g = 2'
        d = 2
        e = ""
        f = []
        return True
    def __abs__(self):
        global g
        f'START;'
        g = 2
        f'END; __abs__ g = 2'
        d = 2
        e = ""
        f = []
        return 1
    def __hash__(self):
        global g
        f'START;'
        g = 2
        f'END; __hash__ g = 2'
        d = 2
        e = ""
        f = []
        return 1
    def __len__(self):
        global g
        f'START;'
        g = 2
        f'END; __len__ g = 2'
        d = 2
        e = ""
        f = []
        return 1

UsesSelf() == UsesSelf()
abs(UsesSelf())
hash(UsesSelf())
len(UsesSelf())

UsesGlobal() == UsesGlobal()
abs(UsesGlobal())
hash(UsesGlobal())
len(UsesGlobal())