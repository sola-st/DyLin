class NonSymmetric:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.x <= other.x
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class NonStable:
    def __init__(self):
        self.toggle = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        res = self.toggle == other.toggle
        if self.toggle < 9:
            self.toggle += 1
            return res
        return not res

    def __ne__(self, other):
        return self.__eq__(other)


class BadIdentiy:
    def __eq__(self, other):
        if other == None:
            return True
        else:
            return id(self) == id(other)

    def __nq__(self, other):
        return not self.__eq__(other)


class BadReflexivity:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.x != other.x

    def __ne__(self, other):
        return not self.__eq__(other)


# symmetry
NonSymmetric(1) == NonSymmetric(2)  # DyLin warn
NonSymmetric(1) != NonSymmetric(2)  # DyLin warn

# stability
NonStable() == NonStable()  # DyLin warn
NonStable() != NonStable()  # DyLin warn

# identity
BadIdentiy() == BadIdentiy()  # DyLin warn
BadIdentiy() != BadIdentiy()  # DyLin warn

# reflexivity
BadReflexivity(1) == BadReflexivity(1)  # DyLin warn
BadReflexivity(1) != BadReflexivity(1)  # DyLin warn
