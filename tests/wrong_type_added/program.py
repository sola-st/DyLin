def test(length=1001):
    l = [10] * length
    l.append("test")  # DyLin warn
    l = [10] * length
    l.extend(["testi", 10])  # DyLin warn
    l = [10] * length
    l.insert(20, "test")  # DyLin warn
    l = [10] * length
    l += ["test"]  # DyLin warn
    l = [10] * length
    l = l + ["test"]  # DyLin warn

    l = set(range(0, length))
    l.add("test")  # DyLin warn

    l = [10] * length
    l.append(10)
    l.extend([10])
    l.insert(20, 10)
    l += [10]
    l = l + [10]

    l = set(range(0, length))
    l.add(32131313)


test()
