#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Erstellen sie ein Dataframe  mit 10^5 uniform zwischen 0 und 1 verteilten Zufallszahlen x_1, x_2.
num_randoms = 10**5
data = pd.DataFrame(
    {"x_1": np.random.rand(num_randoms), "x_2": np.random.rand(num_randoms)}
)
# print(data)
