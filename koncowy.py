import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# FUNKCJE WŁASNE (ALGEBRA LINIOWA W PRAKTYCE)
# =========================================================
def wyznacz_macierz_kowariancji(stopy_zwrotu: pd.DataFrame) -> pd.DataFrame:
    """
    Ręczna implementacja macierzy kowariancji z wykorzystaniem
    własności algebry liniowej (iloczyn macierzy wycentrowanych).
    """
    R = stopy_zwrotu.to_numpy()
    T, n = R.shape
    X = R - np.mean(R, axis=0)
    Sigma = (X.T @ X) / (T - 1)

    macierz_df = pd.DataFrame(Sigma, index=stopy_zwrotu.columns, columns=stopy_zwrotu.columns)
    return macierz_df


# =========================================================
# 1. POBRANIE DANYCH I OBLICZENIA LINIOWE
# =========================================================
spolki = ['AMZN', 'ORCL', 'NVDA', 'HIMS', 'AAPL', 'JNJ', 'KO', 'TLT', 'AGG', 'GLD']
print("Ściągam dane z giełdy dla 10 aktywów...")

dane = yf.download(spolki, start='2024-01-01', end='2026-05-01')['Close']
dzienne_zwroty = dane.pct_change().dropna()

print("Licze oczekiwane zwroty i macierz kowariancji w skali rocznej...")
roczne_zwroty = dzienne_zwroty.mean() * 252
macierz_kowariancji = wyznacz_macierz_kowariancji(dzienne_zwroty)

# =========================================================
# 2. SYMULACJA MONTE CARLO
# =========================================================
liczba_prob = 500000
liczba_aktywow = len(spolki)
print(f"Odpalam Monte Carlo dla {liczba_prob} portfeli. To może chwilę zająć...")

wyniki_zwroty = np.zeros(liczba_prob)
wyniki_ryzyko = np.zeros(liczba_prob)
wyniki_sharpe = np.zeros(liczba_prob)
zapisane_wagi = np.zeros((liczba_prob, liczba_aktywow))

Sigma_np = macierz_kowariancji.to_numpy()

for i in range(liczba_prob):
    # Losowanie i normalizacja wag
    wagi = np.random.random(liczba_aktywow)
    wagi = wagi / np.sum(wagi)

    # Algebra Liniowa
    zwrot_portfela = wagi.T @ roczne_zwroty
    ryzyko_portfela = np.sqrt(wagi.T @ Sigma_np @ wagi)

    # Zapis
    wyniki_zwroty[i] = zwrot_portfela
    wyniki_ryzyko[i] = ryzyko_portfela
    wyniki_sharpe[i] = zwrot_portfela / ryzyko_portfela
    zapisane_wagi[i, :] = wagi

# =========================================================
# 3. WYZNACZENIE ZŁOTEGO ŚRODKA
# =========================================================
najlepszy_indeks = np.argmax(wyniki_sharpe)
najlepsze_wagi = zapisane_wagi[najlepszy_indeks]
najlepszy_zwrot = wyniki_zwroty[najlepszy_indeks]
najlepsze_ryzyko = wyniki_ryzyko[najlepszy_indeks]

print("\n--- ZNALEZIONO OPTYMALNY PORTFEL (Max Sharpe) ---")
print(f"Oczekiwany zysk roczny: {najlepszy_zwrot * 100:.2f}%")
print(f"Ryzyko (zmienność):     {najlepsze_ryzyko * 100:.2f}%")
print(f"Wskaźnik Sharpe'a:      {wyniki_sharpe[najlepszy_indeks]:.4f}")

# =========================================================
# 4. WIZUALIZACJA - ZBIORCZY DASHBOARD
# =========================================================
print("\nGeneruje zbiorczy panel z wykresami (Dashboard)...")

# Tworzymy główną figurę o odpowiednich proporcjach
fig = plt.figure(figsize=(16, 12))

# Definiujemy siatkę (GridSpec) - 2 wiersze, 2 kolumny
# Górny wiersz jest szerszy (height_ratios=[1.5, 1]) żeby główny wykres był dobrze widoczny
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

# --- WYKRES 1: Granica Efektywna (Zajmuje cały górny wiersz) ---
ax1 = fig.add_subplot(gs[0, :])
scatter = ax1.scatter(wyniki_ryzyko * 100, wyniki_zwroty * 100, c=wyniki_sharpe, cmap='viridis', marker='o', s=5,
                      alpha=0.3)
fig.colorbar(scatter, ax=ax1, label="Wskaźnik Sharpe'a (Zysk / Ryzyko)")
ax1.scatter(najlepsze_ryzyko * 100, najlepszy_zwrot * 100, color='red', marker='*', s=400,
            label='Złoty Środek (Max Sharpe)')
ax1.set_title('Symulacja Monte Carlo - Granica Efektywna Markowitza', fontsize=16, fontweight='bold')
ax1.set_xlabel('Ryzyko (Odchylenie standardowe) [%]', fontsize=12)
ax1.set_ylabel('Oczekiwany Zwrot [%]', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- WYKRES 2: Mapa Cieplna (Dolny wiersz, lewa strona) ---
ax2 = fig.add_subplot(gs[1, 0])
sns.heatmap(dzienne_zwroty.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, ax=ax2,
            annot_kws={"size": 9})
ax2.set_title('Macierz Korelacji', fontsize=14)

# --- WYKRES 3: Wykres Kołowy (Dolny wiersz, prawa strona) ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.pie(najlepsze_wagi, labels=spolki, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
ax3.set_title('Struktura Złotego Portfela', fontsize=14)

# Automatyczne dopasowanie marginesów, żeby nic na siebie nie nachodziło
plt.tight_layout()

# Zapisujemy cały wygenerowany kokpit do jednego pliku obrazu w wysokiej rozdzielczości
nazwa_pliku = 'projekt_awad_dashboard.png'
plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nGotowe! Wszystkie wykresy zapisano do jednego pliku: {nazwa_pliku}")

# Wyświetlenie na ekranie
plt.show()