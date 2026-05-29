import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


# =========================================================
# FUNKCJE WLASNE (ALGEBRA LINIOWA W PRAKTYCE)
# =========================================================
def wyznacz_macierz_kowariancji(stopy_zwrotu: pd.DataFrame) -> pd.DataFrame:
    """
    Reczna implementacja macierzy kowariancji z wykorzystaniem
    wlasnosci algebry liniowej (iloczyn macierzy wycentrowanych).
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
print("Sciagam dane z gieldy dla 10 aktywow...")

dane = yf.download(spolki, start='2024-01-01', end='2026-05-01')['Close']
dzienne_zwroty = dane.pct_change().dropna()

print("Licze oczekiwane zwroty i macierz kowariancji w skali rocznej...")
roczne_zwroty = dzienne_zwroty.mean() * 252
macierz_kowariancji = wyznacz_macierz_kowariancji(dzienne_zwroty) * 252

# Stopa wolna od ryzyka (roczna)
r_f = 0.042

# =========================================================
# 2. SYMULACJA MONTE CARLO
# =========================================================
liczba_prob = 50000
liczba_aktywow = len(spolki)
print(f"Odpalam Monte Carlo dla {liczba_prob} portfeli. To moze chwile zajac...")

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
    wyniki_sharpe[i] = (zwrot_portfela - r_f) / ryzyko_portfela
    zapisane_wagi[i, :] = wagi

# =========================================================
# 3. WYZNACZENIE ZLOTEGO SRODKA (Monte Carlo)
# =========================================================
najlepszy_indeks = np.argmax(wyniki_sharpe)
najlepsze_wagi = zapisane_wagi[najlepszy_indeks]
najlepszy_zwrot = wyniki_zwroty[najlepszy_indeks]
najlepsze_ryzyko = wyniki_ryzyko[najlepszy_indeks]

print("\n--- ZNALEZIONO OPTYMALNY PORTFEL Monte Carlo (Max Sharpe) ---")
print(f"Oczekiwany zysk roczny: {najlepszy_zwrot * 100:.2f}%")
print(f"Ryzyko (zmiennosc):     {najlepsze_ryzyko * 100:.2f}%")
print(f"Wskaznik Sharpe'a:      {wyniki_sharpe[najlepszy_indeks]:.4f}")

print("Wagi portfela Monte Carlo:")
for t, wi in zip(spolki, najlepsze_wagi):
    if wi < 1e-6:
        print(f"    {t:>5s}: {wi * 100:>6.2f}%  << WYRZUCONE (KKT: gamma_i > 0, w_i = 0)")
    else:
        print(f"    {t:>5s}: {wi * 100:>6.2f}%  -- aktywne   (KKT: gamma_i = 0, w_i > 0)")


# =========================================================
# 4. OPTYMALIZACJA KWADRATOWA MARKOWITZA (SLSQP / KKT)
# =========================================================
#
#   Problem Markowitza:
#     min_w  (1/2) w^T Sigma w
#     s.t.   w^T 1  = 1       (pelna inwestycja)
#            w^T mu = mu*     (cel zysku)
#            w >= 0           (brak krotkiej sprzedazy)
#
#   Lagrangian z warunkami KKT:
#     L(w, lam1, lam2, gamma) = (1/2) w^T Sigma w
#                               - lam1 (w^T 1 - 1)
#                               - lam2 (w^T mu - mu*)
#                               - gamma^T w
#
#   Gradient L po w = 0:
#     Sigma w - lam1*1 - lam2*mu - gamma = 0
#
#   Warunek komplementarnosci:
#     gamma_i * w_i = 0   =>  albo w_i > 0 (aktywo w grze)
#                              albo w_i = 0 (wyrzucone z portfela)

print("\n--- OPTYMALIZACJA KWADRATOWA (SLSQP / KKT) ---")

mu_np = roczne_zwroty.to_numpy()


def wariancja_portfela(w, Sigma):
    """Funkcja celu: (1/2) w^T Sigma w"""
    return 0.5 * w @ Sigma @ w


def ujemny_sharpe(w, mu_vec, Sigma):
    """Ujemny Sharpe (minimalizujemy => maksymalizacja Sharpe)."""
    ret = w @ mu_vec
    vol = np.sqrt(w @ Sigma @ w)
    return -(ret - r_f) / vol


# --- Portfel o MAKSYMALNYM SHARPE (tangent portfolio) ---
#    min_w  -Sharpe(w)
#    s.t.   sum(w) = 1    (w^T 1 = 1)
#           w >= 0        (brak krotkiej sprzedazy)

ograniczenia = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
granice = tuple((0, 1) for _ in range(liczba_aktywow))
w0 = np.ones(liczba_aktywow) / liczba_aktywow

wynik_slsqp = minimize(
    ujemny_sharpe,
    w0,
    args=(mu_np, Sigma_np),
    method='SLSQP',
    bounds=granice,
    constraints=ograniczenia,
    options={'maxiter': 1000, 'ftol': 1e-12}
)

wagi_opt = wynik_slsqp.x
zwrot_opt = wagi_opt @ mu_np
ryzyko_opt = np.sqrt(wagi_opt @ Sigma_np @ wagi_opt)
sharpe_opt = (zwrot_opt - r_f) / ryzyko_opt

print(f"Oczekiwany zysk roczny: {zwrot_opt * 100:.2f}%")
print(f"Ryzyko (zmiennosc):     {ryzyko_opt * 100:.2f}%")
print(f"Wskaznik Sharpe'a:      {sharpe_opt:.4f}")
print()
print("Wagi portfela (SLSQP):")
for t, wi in zip(spolki, wagi_opt):
    if wi < 1e-6:
        print(f"    {t:>5s}: {wi * 100:>6.2f}%  << WYRZUCONE (KKT: gamma_i > 0, w_i = 0)")
    else:
        print(f"    {t:>5s}: {wi * 100:>6.2f}%  -- aktywne   (KKT: gamma_i = 0, w_i > 0)")


# --- Granica efektywna (analityczna, 200 punktow) ---
#    Dla kazdego docelowego mu*:
#      min_w  (1/2) w^T Sigma w
#      s.t.   w^T 1  = 1,  w^T mu = mu*,  w >= 0

docelowe_zwroty = np.linspace(mu_np.min(), mu_np.max(), 200)
granica_ryzyko = []
granica_zwrot  = []

for mu_star in docelowe_zwroty:
    ogr = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, t=mu_star: w @ mu_np - t}
    ]
    wynik = minimize(
        wariancja_portfela, w0, args=(Sigma_np,),
        method='SLSQP', bounds=granice, constraints=ogr,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    if wynik.success:
        granica_ryzyko.append(np.sqrt(wynik.x @ Sigma_np @ wynik.x))
        granica_zwrot.append(mu_star)

granica_ryzyko = np.array(granica_ryzyko)
granica_zwrot  = np.array(granica_zwrot)

print(f"\nWyznaczono {len(granica_ryzyko)} punktow granicy efektywnej (SLSQP).")


# --- Analiza KKT ---
n_aktywne = sum(1 for w in wagi_opt if w > 1e-6)
n_wyrzucone = liczba_aktywow - n_aktywne

print(f"\nAnaliza KKT (gamma_i * w_i = 0):")
print(f"  Aktywow w portfelu (w_i > 0):    {n_aktywne}")
print(f"  Aktywow wyrzuconych (w_i = 0):   {n_wyrzucone}")


# --- Porownanie MC vs SLSQP ---
print("\n--- POROWNANIE: Monte Carlo vs SLSQP (KKT) ---")
print(f"  {'Metoda':<18s}  {'mu_p':>8s}  {'sigma_p':>8s}  {'Sharpe':>8s}")
print('  ' + '-' * 46)
print(f"  {'Monte Carlo':<18s}  {najlepszy_zwrot:>7.2%}  {najlepsze_ryzyko:>7.2%}  {wyniki_sharpe[najlepszy_indeks]:>8.4f}")
print(f"  {'SLSQP (KKT)':<18s}  {zwrot_opt:>7.2%}  {ryzyko_opt:>7.2%}  {sharpe_opt:>8.4f}")

roznica = ((sharpe_opt - wyniki_sharpe[najlepszy_indeks]) / wyniki_sharpe[najlepszy_indeks]) * 100
print(f"\n  SLSQP daje Sharpe wyzszy o {roznica:+.2f}% wzgledem MC.")


# =========================================================
# 5. WIZUALIZACJA - ZBIORCZY DASHBOARD
# =========================================================
print("\nGeneruje zbiorczy panel z wykresami (Dashboard)...")

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.35, wspace=0.35)

# --- WYKRES 1: Granica Efektywna (caly gorny wiersz) ---
ax1 = fig.add_subplot(gs[0, :])

# Chmura Monte Carlo
scatter = ax1.scatter(
    wyniki_ryzyko * 100, wyniki_zwroty * 100,
    c=wyniki_sharpe, cmap='viridis', marker='o', s=5, alpha=0.3
)
fig.colorbar(scatter, ax=ax1, label="Wskaznik Sharpe'a (Zysk / Ryzyko)")

# Granica efektywna (SLSQP) -- czerwona linia
ax1.plot(
    granica_ryzyko * 100, granica_zwrot * 100,
    color='red', linewidth=2.5, linestyle='-',
    label='Granica efektywna (SLSQP / KKT)'
)

# Zloty Srodek MC (srebrna gwiazda)
ax1.scatter(
    najlepsze_ryzyko * 100, najlepszy_zwrot * 100,
    color='silver', marker='*', s=400, edgecolors='black', linewidths=1,
    label=f'Max Sharpe MC = {wyniki_sharpe[najlepszy_indeks]:.3f}', zorder=5
)

# Zloty Srodek SLSQP (zlota gwiazda)
ax1.scatter(
    ryzyko_opt * 100, zwrot_opt * 100,
    color='gold', marker='*', s=500, edgecolors='black', linewidths=1.5,
    label=f'Max Sharpe SLSQP = {sharpe_opt:.3f}', zorder=6
)

ax1.set_title('Granica Efektywna Markowitza\nMonte Carlo (50 000) + Optymalizacja SLSQP (KKT)',
              fontsize=16, fontweight='bold')
ax1.set_ylabel('Oczekiwany Zwrot [%]', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# --- WYKRES 2: Mapa Cieplna (dolny lewy) ---
ax2 = fig.add_subplot(gs[1, 0])
sns.heatmap(dzienne_zwroty.corr(), annot=True, cmap='coolwarm', fmt='.2f',
            vmin=-1, vmax=1, ax=ax2, annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8})
ax2.set_title('Macierz Korelacji', fontsize=14, pad=12)
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.tick_params(axis='y', rotation=0, labelsize=8)

# --- WYKRES 3: Wykres Kolowy wag Monte Carlo (dolny srodkowy) ---
ax3 = fig.add_subplot(gs[1, 1])

# Filtrujemy aktywa z waga > 0.1% (male wagi nie pokazujemy)
maska_mc = najlepsze_wagi > 0.001
pie_wagi_mc = najlepsze_wagi[maska_mc]
pie_spolki_mc = [s for s, m in zip(spolki, maska_mc) if m]
n_wyrzucone_mc = liczba_aktywow - sum(maska_mc)

wedges3, texts3, autotexts3 = ax3.pie(
    pie_wagi_mc, labels=pie_spolki_mc, autopct='%1.1f%%',
    startangle=140, pctdistance=0.75, labeldistance=1.15,
    textprops={'fontsize': 9}
)
for at in autotexts3:
    at.set_fontsize(7)

tytul_pie_mc = f'Portfel optymalny (Monte Carlo)\nSharpe = {wyniki_sharpe[najlepszy_indeks]:.3f}'
ax3.set_title(tytul_pie_mc, fontsize=13, pad=12)

# --- WYKRES 4: Wykres Kolowy wag SLSQP (dolny prawy) ---
ax4 = fig.add_subplot(gs[1, 2])

# Filtrujemy aktywa z waga > 0.1% (wyrzucone przez KKT nie pokazujemy)
maska = wagi_opt > 0.001
pie_wagi = wagi_opt[maska]
pie_spolki = [s for s, m in zip(spolki, maska) if m]

wedges4, texts4, autotexts4 = ax4.pie(
    pie_wagi, labels=pie_spolki, autopct='%1.1f%%',
    startangle=140, pctdistance=0.75, labeldistance=1.15,
    textprops={'fontsize': 9}
)
for at in autotexts4:
    at.set_fontsize(7)

tytul_pie = f'Portfel optymalny (SLSQP)\n{n_wyrzucone} aktywow wyrzuconych przez KKT'
ax4.set_title(tytul_pie, fontsize=13, pad=12)

plt.tight_layout(pad=2.0)

nazwa_pliku = 'projekt_awad_dashboard.png'
plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nGotowe! Wszystkie wykresy zapisano do: {nazwa_pliku}")

plt.show()
