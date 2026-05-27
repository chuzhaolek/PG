import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# ETAP 1: POBRANIE DANYCH
# ==========================================
spolki = ['AMZN', 'ORCL', 'NVDA', 'HIMS', 'AAPL', 'JNJ', 'KO', 'TLT', 'AGG', 'GLD']
print("Pobieranie danych z Yahoo Finance...")
dane = yf.download(spolki, start="2020-01-01", end="2022-05-01")['Adj Close']
stopy_zwrotu = dane.pct_change().dropna()

# ==========================================
# ETAP 2: BAZA ALGEBRAICZNA
# ==========================================
DNI_SESYJNE = 252

# Przechodzimy od razu na czyste struktury numpy dla wydajności macierzowej
mu = (stopy_zwrotu.mean() * DNI_SESYJNE).to_numpy()
Sigma = (stopy_zwrotu.cov() * DNI_SESYJNE).to_numpy()
liczba_aktywow = len(spolki)


def statystyki_portfela(wagi):
    """Oblicza zysk, ryzyko i Sharpe'a dla wektora wag (w^T * mu oraz w^T * Sigma * w)"""
    zysk = np.dot(wagi, mu)
    ryzyko = np.sqrt(np.dot(wagi.T, np.dot(Sigma, wagi)))
    return zysk, ryzyko, zysk / ryzyko


# ==========================================
# ETAP 3: CZYSTA MATEMATYKA (UKŁAD RÓWNAŃ Ax = b)
# ==========================================
def optymalizuj_algebraicznie(docelowy_zysk):
    """
    Rozwiązuje układ równań macierzy blokowej Lagrange'a.
    Zwraca idealne wagi portfela dla zadanego docelowego zysku.
    """
    mu_col = mu.reshape(liczba_aktywow, 1)
    jedynki = np.ones((liczba_aktywow, 1))
    zero = np.array([[0]])

    # Budowa macierzy blokowej A (wymiar: 12x12)
    A = np.block([
        [Sigma, mu_col, jedynki],
        [mu_col.T, zero, zero],
        [jedynki.T, zero, zero]
    ])

    # Budowa wektora celów b (wymiar: 12x1)
    b = np.vstack([
        np.zeros((liczba_aktywow, 1)),
        [[docelowy_zysk]],
        [[1]]
    ])

    # Algebra liniowa: x = A^-1 * b
    x = np.linalg.solve(A, b)

    # Zwracamy tylko pierwsze 10 elementów (wagi spółek)
    return x[:liczba_aktywow].flatten()


# ==========================================
# ETAP 4: OBLICZENIA I SYMULACJA
# ==========================================

# 1. Obliczanie matematycznej Granicy Efektywnej
print("\nRozwiązywanie układów równań liniowych dla 50 poziomów zysku...")
# Wybieramy zakres zysków od najgorszej do najlepszej spółki
zakres_zyskow = np.linspace(mu.min(), mu.max(), 50)
matematyczne_ryzyko = []
matematyczne_zyski = []

for docelowy_zysk in zakres_zyskow:
    # Wyliczamy optymalne wagi ze wzoru analitycznego
    wagi_opt = optymalizuj_algebraicznie(docelowy_zysk)
    # Sprawdzamy ryzyko dla tych wag
    zysk, ryzyko, _ = statystyki_portfela(wagi_opt)
    matematyczne_ryzyko.append(ryzyko)
    matematyczne_zyski.append(zysk)

# 2. Symulacja Monte Carlo dla porównania (10 000 losowych portfeli)
print("Generowanie chmury 10 000 losowych portfeli (Monte Carlo)...")
liczba_portfeli = 10000
mc_zyski = np.zeros(liczba_portfeli)
mc_ryzyko = np.zeros(liczba_portfeli)

for i in range(liczba_portfeli):
    wagi = np.random.random(liczba_aktywow)
    wagi /= np.sum(wagi)  # normalizacja do 1
    mc_zyski[i], mc_ryzyko[i], _ = statystyki_portfela(wagi)

# ==========================================
# ETAP 5: WIZUALIZACJA
# ==========================================
plt.figure(figsize=(12, 8))

# Rysujemy chmurę losowych prób
plt.scatter(mc_ryzyko, mc_zyski, c='lightgray', marker='o', s=15, alpha=0.5, label='Losowe portfele (Monte Carlo)')

# Rysujemy idealną linię wyznaczoną przez matematykę (Rozwiązania Ax = b)
plt.plot(matematyczne_ryzyko, matematyczne_zyski, color='crimson', linewidth=4,
         label='Matematyczna Granica Efektywna (Lagrange)')

# Wyszukanie matematycznego Portfela Minimalnego Ryzyka (MVP) na naszej krzywej
index_mvp = np.argmin(matematyczne_ryzyko)
plt.scatter(matematyczne_ryzyko[index_mvp], matematyczne_zyski[index_mvp], color='blue', marker='*', s=400, zorder=5,
            label='Minimalne Ryzyko (MVP)')

# Formatyzowanie wykresu
plt.title('Dowód Algebraiczny Markowitza: Lagrange vs Monte Carlo', fontsize=16)
plt.xlabel('Ryzyko (Odchylenie standardowe)', fontsize=12)
plt.ylabel('Oczekiwana Stopa Zwrotu', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# Wyświetlamy konkretny portfel dla przykładowego zysku np. 15%
przyklad_zysk = 0.15
przyklad_wagi = optymalizuj_algebraicznie(przyklad_zysk)
print("-" * 50)
print(f"ANALYTYCZNE WAGI DLA ZYSKU {przyklad_zysk * 100}% (Oparte na macierzy):")
for ticker, waga in zip(spolki, przyklad_wagi):
    # Wyświetlamy wszystkie, by zobaczyć, że algebra pozwala na krótką sprzedaż (wagi < 0)
    print(f"{ticker}: {waga * 100:>7.2f}%")

plt.show()