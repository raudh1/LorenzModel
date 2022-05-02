def chebyshev_distance(v1, v2):
    #Return the Chebyshev distance between equal-length vectors
    if len(v1) != len(v2):
        raise ValueError("Undefined for vectors of unequal length")
    return max(abs(e1-e2) for e1, e2 in zip(v1, v2))

print(chebyshev_distance([1,2],[3,5]),'[1,2],[3,5]')
