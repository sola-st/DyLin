global g
g = 0


class UsesSelf:
    def __init__(self):
        self.a = ""
        self.b = 1
        self.c = []

    def __eq__(self, other):
        self.a = "1"  # DyLin warn
        self.b = 2  # DyLin warn
        self.c = ["2"]  # DyLin warn
        d = 2
        e = ""
        f = []
        return True

    def __abs__(self):
        self.a = "1"  # DyLin warn
        self.b = 2  # DyLin warn
        self.c = ["2"]  # DyLin warn
        d = 2
        e = ""
        f = []
        return 1

    def __hash__(self):
        self.a = "1"  # DyLin warn
        self.b = 2  # DyLin warn
        self.c = ["2"]  # DyLin warn
        d = 2
        e = ""
        f = []
        return 1

    def __len__(self):
        self.a = "1"  # DyLin warn
        self.b = 2  # DyLin warn
        self.c = ["2"]  # DyLin warn
        d = 2
        e = ""
        f = []
        return 1


class UsesGlobal:
    def __init__(self):
        global g
        g = 2
        d = 2
        e = ""
        f = []

    def __eq__(self, other):
        global g
        g = 2  # DyLin warn
        d = 2
        e = ""
        f = []
        return True

    def __abs__(self):
        global g
        g = 2  # DyLin warn
        d = 2
        e = ""
        f = []
        return 1

    def __hash__(self):
        global g
        g = 2  # DyLin warn
        d = 2
        e = ""
        f = []
        return 1

    def __len__(self):
        global g
        g = 2  # DyLin warn
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
