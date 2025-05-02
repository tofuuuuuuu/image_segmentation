def coord_to_int(i, j, n, m) :
    return i * m + j

def int_to_coord(a, n, m) :
    return [a // m, a % m]

def dot(c1, c2) :
    return c1[0] * c2[0] + c1[1] * c2[1]

def abs(c1) :
    return (c1[0]**2 + c1[1]**2) **0.5