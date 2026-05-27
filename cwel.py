import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# 1. POBRANIE DANYCH (Skrócone dla czytelności)
# =========================================================
spolki = ['AMZN', 'ORCL', 'NVDA', 'HIMS', 'AAPL', 'JNJ', 'KO', 'TLT', 'AGG', 'GLD']
dane = yf.download(spolki, start='2020-01-01', end='2022-05-01')['Close']
dzienne_zwroty = dane.pct_change().dropna()

mu = dzienne_zwroty.mean().to_numpy() * 252           # Wektor mu
Sigma = dzienne_zwroty.cov().to_numpy() * 252         # Macierz kowariancji Sigma
wektor_jedynek = np.ones(len(spolki))                 # Wektor 1

# =========================================================
# 2. ALGEBRA LINIOWA (Rozwiązanie z układu Lagrange'a)
# =========================================================
print("Rozwiązuję analitycznie (odwracanie macierzy)...")

# Krok 1: Odwracamy macierz kowariancji (Sigma^-1)
# To najtrudniejsza operacja matematyczna w tym kodzie
Sigma_inv = np.linalg.inv(Sigma)

# Krok 2: Liczymy licznik ze wzoru (Sigma^-1 * mu)
licznik = Sigma_inv @ mu

# Krok 3: Liczymy mianownik ze wzoru (1^T * Sigma^-1 * mu)
# W Pythonie mnożenie wektora 1D przez wektor daje nam skalar (pojedynczą liczbę)
mianownik = wektor_jedynek.T @ Sigma_inv @ mu

# Krok 4: Wyznaczamy idealne wagi (zgodnie z ujęciem Lagrange'a)
wagi_optymalne = licznik / mianownik

# Krok 5: Wyliczamy końcowe parametry dla znalezionego punktu
zysk_optymalny = wagi_optymalne.T @ mu
ryzyko_optymalne = np.sqrt(wagi_optymalne.T @ Sigma @ wagi_optymalne)
sharpe_optymalny = zysk_optymalny / ryzyko_optymalne

# =========================================================
# 3. WYŚWIETLENIE WYNIKÓW
# =========================================================
print("\n--- ZNALEZIONY PORTFEL (Czysta Algebra / Lagrange) ---")
print(f"Oczekiwany zysk roczny: {zysk_optymalny * 100:.2f}%")
print(f"Ryzyko (zmienność):     {ryzyko_optymalne * 100:.2f}%")
print(f"Wskaźnik Sharpe'a:      {sharpe_optymalny:.4f}")
print("\nPodział kapitału (ZAUWAŻ UJEMNE WAGI - KRÓTKA SPRZEDAŻ):")

for i in range(len(spolki)):
    znak = "KUP" if wagi_optymalne[i] > 0 else "SPRZEDAJ (KRÓTKO)"
    print(f" -> {spolki[i]:<5}: {wagi_optymalne[i] * 100:>6.2f}%  [{znak}]")