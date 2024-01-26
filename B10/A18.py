import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import lombscargle

# Daten einlesen 
df = pd.read_csv('Temperaturdaten_Dortmund.csv')

# Plot
plt.plot(df['Messzeit'], df['Temperatur'])
plt.xlabel('t [year]')
plt.ylabel('T [C]')
plt.savefig('Temperaturplot.pdf')


# Überprüfen auf fehlende Werte
anzahl_nan = df.isna().sum().sum()
print("Anzahl der NaN-Werte:", anzahl_nan)
df = df.dropna() # nan Werte droppen

# Überprüfung der Zeiltintervalle 
plt.figure()
plt.plot(df['Messzeit'], df['Messzeit'].diff(), '-')
plt.xlabel('t [year]')
plt.ylabel('$\Delta$t [year]')
plt.yscale('log')
plt.savefig('Zeitdifferenz.pdf')

# Datenvorbereitung

#in Numpy Array, damit die funktionen nicht meckern
T = df['Temperatur'].to_numpy()
t = df['Messzeit'].to_numpy()

# Temperaturverteilung um T=0 zentrieren
T_center = T - np.mean(T) 
if np.mean(T_center)>1e-6:
    print('Error: Daten sind nicht richtig zentriert')

# Lomb-Scargle Periodogramm
mins_per_year = 365*24*60
delta_t = 15/mins_per_year 
Delta_t = t[-1] -t[0]

f_min = 1/Delta_t
f_max= 1/(2*delta_t)

frequenzen = np.logspace(np.log10(0.1), np.log10(1000), num=1000)
power = lombscargle(t, T_center, 2*np.pi*frequenzen)
plt.figure()
plt.plot(frequenzen, power)
plt.axvline(x=1, color='gray', linestyle='--', label='1 Year')
plt.axvline(x=365, color='gray', linestyle='--', label='1 Day')
plt.xlabel('f [1/year]')
plt.ylabel('Power')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Lomb-Scargle Periodogramm')
plt.savefig('Lomb-Scargle.pdf')

frequenzen = np.arange(0.001, 2, 0.001)
power = lombscargle(t, T_center, 2*np.pi*frequenzen)
plt.figure()
plt.plot(frequenzen, power)
plt.axvline(x=1, color='gray', linestyle='--', label='1 Year')
plt.xlabel('f [1/year]')
plt.ylabel('Power')
plt.yscale('log')
plt.grid()
plt.title('Lomb-Scargle Periodogramm')
plt.savefig('Lomb-Scargle_year.pdf')

frequenzen = np.arange(360, 370, 0.01)
power = lombscargle(t, T_center, 2*np.pi*frequenzen)
plt.figure()
plt.plot(frequenzen, power)
plt.axvline(x=365, color='gray', linestyle='--', label='1 Day')
plt.xlabel('f [1/year]')
plt.ylabel('Power')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Lomb-Scargle Periodogramm')
plt.savefig('Lomb-Scargle_day.pdf')