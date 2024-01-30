import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import lombscargle
from scipy.interpolate import interp1d 
from scipy.fft import fft, fftfreq
from scipy.fft import ifft

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
plt.axvline(x=1, color='gray', linestyle='--', label='1 Year')
plt.axvline(x=365, color='gray', linestyle='--', label='1 Day')
plt.plot(frequenzen, power)
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
plt.axvline(x=1, color='gray', linestyle='--', label='1 Year')
plt.plot(frequenzen, power)
plt.xlabel('f [1/year]')
plt.ylabel('Power')
plt.yscale('log')
plt.grid()
plt.title('Lomb-Scargle Periodogramm')
plt.savefig('Lomb-Scargle_year.pdf')

frequenzen = np.arange(360, 370, 0.01)
power = lombscargle(t, T_center, 2*np.pi*frequenzen)
plt.figure()
plt.axvline(x=365, color='gray', linestyle='--', label='1 Day')
plt.plot(frequenzen, power)
plt.xlabel('f [1/year]')
plt.ylabel('Power')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Lomb-Scargle Periodogramm')
plt.savefig('Lomb-Scargle_day.pdf')

# Fouriertrafo 

t_reg = np.linspace(t.min(), t.max(), num=len(t))
interpolator = interp1d(t, T_center, kind='linear')
T_reg = interpolator(t_reg)

plt.figure()
plt.title('Interpolation')
plt.plot(t_reg, T_reg)
plt.xlabel("t [year]")
plt.ylabel("$T-T_{mean}$ [C]")
plt.savefig("grid.pdf")

# Berechnung der FFT und der Frequenzen
fft_result = fft(T_reg)
fft_freq = fftfreq(len(T_reg), d=(t_reg[1] - t_reg[0]))
# Nur die positive Hälfte des Spektrums ist von Interesse
n = len(fft_freq) // 2

# Plot des Frequenzspektrums
plt.figure()
plt.axvline(x=1, color='gray', linestyle='--', label='1 Year')
plt.axvline(x=365, color='gray', linestyle='--', label='1 Day')
plt.plot(fft_freq[:n], np.abs(fft_result[:n]))
plt.xlabel('f [1/year]')
plt.ylabel('Power')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Fourier')
plt.legend()
plt.savefig('Fourier_Transformiert.pdf')

# 1. Identifizierung der maximalen Frequenzen (ohne Gleichanteil)
# Wir ignorieren die erste Komponente, da sie dem Gleichanteil entspricht
indices = np.argsort(np.abs(fft_result[1:n]))[-2:] + 1  # +1, um den Gleichanteil zu überspringen

# 2. Isolation der Peaks
fft_filtered = np.zeros_like(fft_result)
fft_filtered[indices] = fft_result[indices]
fft_filtered[-indices] = fft_result[-indices]  # Berücksichtigung der konjugierten Frequenzen für die Rücktransformation

# 3. Rücktransformation
T_filtered = ifft(fft_filtered)

# 4. Plot der rekonstruierten Zeitreihe zusammen mit den originalen Daten
plt.figure()
plt.plot(t_reg, T_reg, label='Interpolation', linewidth=0.1)
plt.plot(t_reg, T_filtered.real, label='Reconstruction (2 freq)', linestyle='-', linewidth=0.1)
plt.xlabel('t [year]')
plt.ylabel('T [C]')
plt.title('Fourier Reconstruction')
plt.legend()
plt.savefig('Vergleich_Interpolation_Rekonstruktion.pdf')