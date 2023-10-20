#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Erstellen sie ein Dataframe  mit 10^5 uniform zwischen 0 und 1 verteilten Zufallszahlen x_1, x_2.
num_randoms = 10**5
data = pd.DataFrame(
    {"x_1": np.random.rand(num_randoms), "x_2": np.random.rand(num_randoms)}
)
# print(data)


# Berechnen sie aus diesen Attributen ein drittes Attribut x_3 mit der Funktionsvorschrift


def funktion(x_1, x_2):
    x_3 = 15 * np.sin(4 * np.pi * x_1) + 60 * (x_2 - 0.5) ** 2
    return x_3


# Addieren Sie auf diese Zahl eine standardnormalverteilte Zufallszahl, um Rauschen zu simulieren.
data["x_3"] = funktion(data["x_1"], data["x_2"]) + np.random.normal()
# Das x_3 Attribut ist von nun an ihr Zielattribut.

print(data.head())

# Teilen Sie das Dataframe in einen Trainings- und Test-Datensatz auf.
X_train, X_test, y_train, y_test = train_test_split(
    data[["x_1", "x_2"]], data["x_3"], test_size=0.33, random_state=42
)

# Wählen Sie einen Random-Forest-Regressor mit 200 Bäumen und trainieren Sie diesen auf dem Trainingsdatensatz um x_3
# zu schätzen.

rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
rf_regressor.fit(X_train, y_train)

# Vorhersagen auf den Testdaten machen
predictions = rf_regressor.predict(X_test)

# Die Mean Squared Error (MSE) Metrik zur Bewertung verwenden
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Stellen Sie die erstellten Daten und die Vorhersagen des Regressors in einem dreidimensonalen Plot und mehreren
# 2-dimensionalen Projektionen dar um die Vorhersage mit der Wahrheit zu vergleichen. Geben Sie außerdem den mean
# squared error der Vorhersage zu den wahren Werten an.
