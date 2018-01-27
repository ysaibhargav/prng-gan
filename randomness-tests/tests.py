import numpy as np
from scipy.stats import chisquare
import pdb

# group x into chunks of size s and transform
# each chunk to index
def transform(x, s, base = 2):
    x = "".join([str(_x) for _x in x])
    x = x[:(len(x) / s) * s]
    x_tr = []

    def bits_to_idx(val):
        idx = 0
        for i, _val in enumerate(val):
            idx += int(_val) * (base ** i)
        return idx

    for i in range(len(x) / s):
        val = x[i*s:(i+1)*s]
        x_tr.append(bits_to_idx(val))
    
    return x_tr

def find_in_A(A, val):
   for i, A_ in enumerate(A[::-1]):
       if val >= A_:
           return len(A) - i - 1

def book_stack(x, s = 4, base = 2, A = None):
    num_alp = base ** s
    indices = range(num_alp)    
    x = transform(x, s, base)

    if A is None:
        A = range(0, num_alp, s)

    n = [0 for _ in A]

    for val in x:
        n[find_in_A(A, indices[val])] += 1
        for i, idx in enumerate(indices):
            if idx < indices[val]:
                indices[i] += 1
        indices[val] = 0 

    A.append(num_alp)
    n_exp = [len(x)*(float(b-a)/num_alp) \
            for a, b in zip(A, A[1:])]

    return chisquare(n, n_exp)

def order_test(x, s = 4, base = 2, A = None):
    num_alp = base ** s
    indices = range(num_alp)    
    num_occ = [0 for _ in range(num_alp)]    
    x = transform(x, s, base)

    if A is None:
        A = range(0, num_alp, s)

    n = [0 for _ in A]

    for val in x:
        n[find_in_A(A, indices[val])] += 1
        val_idx = indices[val]
        for i, idx in enumerate(indices):
            if num_occ[i] <= num_occ[val] \
                    and idx < indices[val]:
                indices[i] += 1
                val_idx -= 1
        indices[val] = val_idx
        num_occ[val] += 1

    A.append(num_alp)
    n_exp = [len(x)*(float(b-a)/num_alp) \
            for a, b in zip(A, A[1:])]

    return chisquare(n, n_exp)
