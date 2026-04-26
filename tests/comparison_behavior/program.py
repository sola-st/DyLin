# Fixture for custom equality implementations that break comparison invariants.
# This class uses <= inside __eq__, so a == b can differ from b == a.
class NonSymmetric:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.x <= other.x
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


# This class mutates internal state while comparing, so repeated comparisons can change result.
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


# This class treats None as equal and falls back to object identity for everything else.
class BadIdentiy:
    def __eq__(self, other):
        if other == None:
            return True
        else:
            return id(self) == id(other)

    def __nq__(self, other):
        return not self.__eq__(other)


# This class defines equality as "different payload", so even self-comparison can fail.
class BadReflexivity:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.x != other.x

    def __ne__(self, other):
        return not self.__eq__(other)


# Symmetry violations: swapping operands should not change equality semantics.
NonSymmetric(1) == NonSymmetric(2)  # DyLin warn
NonSymmetric(1) != NonSymmetric(2)  # DyLin warn

# Stability violations: the same comparison should not flip just because it ran before.
NonStable() == NonStable()  # DyLin warn
NonStable() != NonStable()  # DyLin warn

# Identity violations: distinct instances should not compare equal by a broken identity rule.
BadIdentiy() == BadIdentiy()  # DyLin warn
BadIdentiy() != BadIdentiy()  # DyLin warn

# Reflexivity violations: any value should compare equal to itself.
BadReflexivity(1) == BadReflexivity(1)  # DyLin warn
BadReflexivity(1) != BadReflexivity(1)  # DyLin warn
