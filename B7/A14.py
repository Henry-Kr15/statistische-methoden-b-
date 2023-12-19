#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.optimize import minimize

Messreihe = np.array([1, 2, 3, 4, 5, 6, 7])
Zählungen = np.array([4135, 4202, 4203, 4218, 4227, 4231, 4310])
df = pd.DataFrame({"Messreihe": Messreihe, "Zählungen": Zählungen})
N = len(df)

# Aufgabenteil a
Lambda_Hut = 1 / N * sum(df["Zählungen"])
print(Lambda_Hut)
print(df)


# Aufgabenteil b
def F(params):
    a = params[0]
    b = params[1]
    Term = [
        k_i * np.log(a * t_i + b) - (a * t_i)
        for t_i, k_i in zip(df["Messreihe"], df["Zählungen"])
    ]
    return -np.sum(Term, axis=0)


params_mins = minimize(F, x0=[100, 4000], bounds=((0, 4200), (0, 4200)))
# rennt leider immer in den oberen Bound für b, egal für welchen Wert
a, b = params_mins.x
print(f"Die Parameter sind a={a}, b={b}.")
