import random

random.lognormvariate(0, 1) # OK
random.vonmisesvariate(0, 0) # OK
random.vonmisesvariate(0, 1) # OK

random.lognormvariate(0, 0) # DyLin warn
random.lognormvariate(0, -1) # DyLin warn
random.vonmisesvariate(0, -1) # DyLin warn