"""
================================================================================
  OPTYMALIZACJA PORTFELA INWESTYCYJNEGO METODĄ MONTE CARLO
  Analiza granicy efektywnej Markowitza dla portfela wieloaktywowego
  
  Aktywa: AMZN, ORCL, NVDA, HIMS, AAPL, JNJ, KO, TLT, AGG, GLD
  Okres:  2021-01-01 — 2024-01-01
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORT BIBLIOTEK
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja wizualizacji
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

TRADING_DAYS = 252        # liczba sesji giełdowych w roku
NUM_PORTFOLIOS = 50_000   # liczba symulacji Monte Carlo
np.random.seed(42)        # powtarzalność wyników


# ──────────────────────────────────────────────────────────────────────────────
# 2. POBRANIE DANYCH Z YAHOO FINANCE
# ──────────────────────────────────────────────────────────────────────────────
tickers = ['AMZN', 'ORCL', 'NVDA', 'HIMS', 'AAPL',   # tech / wzrostowe
           'JNJ', 'KO',                                 # defensywne
           'TLT', 'AGG', 'GLD']                          # bezpieczne przystanie

print('Pobieranie danych z Yahoo Finance...')
raw_data = yf.download(tickers, start='2021-01-01', end='2024-01-01', auto_adjust=True)

# Ceny zamknięcia
prices = raw_data['Close'][tickers].dropna()
print(f'Pobrano {len(prices)} sesji dla {len(tickers)} aktywow.')
print(f'Zakres dat: {prices.index[0].date()} -- {prices.index[-1].date()}')
print()


# ──────────────────────────────────────────────────────────────────────────────
# 3. DZIENNE LOGARYTMICZNE STOPY ZWROTU (WEKTORYZACJA)
# ──────────────────────────────────────────────────────────────────────────────
daily_returns = np.log(prices / prices.shift(1)).dropna()

print('Statystyki dziennych stop zwrotu:')
print(daily_returns.describe().round(5))
print()


# ──────────────────────────────────────────────────────────────────────────────
# 4. ANNUALIZACJA — WEKTOR μ I MACIERZ Σ
# ──────────────────────────────────────────────────────────────────────────────
mu = daily_returns.mean() * TRADING_DAYS           # wektor μ (annualizowany)
cov_matrix = daily_returns.cov() * TRADING_DAYS    # macierz Σ (annualizowana)

print('=== Annualizowane oczekiwane stopy zwrotu mu ===')
for ticker, ret in mu.items():
    print(f'  {ticker:>5s}: {ret:>8.2%}')
print()


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAPA CIEPLNA KORELACJI
# ──────────────────────────────────────────────────────────────────────────────
corr_matrix = daily_returns.corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(240, 10, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap=cmap,
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.8,
    square=True,
    cbar_kws={'shrink': 0.8, 'label': 'Korelacja Pearsona'},
    ax=ax
)
ax.set_title('Macierz korelacji dziennych stop zwrotu (2021-2024)',
             fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 6. SYMULACJA MONTE CARLO (50 000 ITERACJI)
# ──────────────────────────────────────────────────────────────────────────────
print(f'Uruchamiam symulację Monte Carlo ({NUM_PORTFOLIOS:,} portfeli)...')

num_assets = len(tickers)
mu_arr = mu.values             # wektor μ: shape (n,)
cov_arr = cov_matrix.values    # macierz Σ: shape (n, n)

# Pre-alokacja tablic wynikowych
results_return = np.zeros(NUM_PORTFOLIOS)
results_risk   = np.zeros(NUM_PORTFOLIOS)
results_sharpe = np.zeros(NUM_PORTFOLIOS)
weights_record = np.zeros((NUM_PORTFOLIOS, num_assets))

for i in range(NUM_PORTFOLIOS):
    # Losowanie wag z rozkładu Dirichleta (sumują się do 1)
    w = np.random.dirichlet(np.ones(num_assets))

    # Oczekiwany zwrot portfela:  μ_p = w^T · μ
    port_return = w @ mu_arr

    # Ryzyko portfela:  σ_p = sqrt(w^T · Σ · w)
    port_risk = np.sqrt(w @ cov_arr @ w)

    # Wskaźnik Sharpe'a (r_f = 0)
    port_sharpe = port_return / port_risk

    results_return[i] = port_return
    results_risk[i]   = port_risk
    results_sharpe[i] = port_sharpe
    weights_record[i] = w

print('Symulacja zakonczona.')
print(f'  Zakres zwrotow:   [{results_return.min():.2%}, {results_return.max():.2%}]')
print(f'  Zakres ryzyka:    [{results_risk.min():.2%}, {results_risk.max():.2%}]')
print(f'  Najwyzszy Sharpe: {results_sharpe.max():.4f}')
print()


# ──────────────────────────────────────────────────────────────────────────────
# 7. PORTFEL OPTYMALNY — MAKSYMALNY SHARPE
# ──────────────────────────────────────────────────────────────────────────────
idx_max_sharpe = results_sharpe.argmax()

opt_return  = results_return[idx_max_sharpe]
opt_risk    = results_risk[idx_max_sharpe]
opt_sharpe  = results_sharpe[idx_max_sharpe]
opt_weights = weights_record[idx_max_sharpe]

print('=' * 60)
print('  PORTFEL OPTYMALNY -- Maksymalny wskaznik Sharpe\'a')
print('=' * 60)
print(f'  Oczekiwany zwrot roczny (mu_p):  {opt_return:>8.2%}')
print(f'  Ryzyko roczne (sigma_p):         {opt_risk:>8.2%}')
print(f'  Wskaznik Sharpe\'a (S_p):         {opt_sharpe:>8.4f}')
print('-' * 60)
print('  Optymalne wagi portfela:')
print('-' * 60)

weights_df = pd.DataFrame({
    'Aktywo': tickers,
    'Waga (%)': (opt_weights * 100).round(2)
}).sort_values('Waga (%)', ascending=False).reset_index(drop=True)

print(weights_df.to_string(index=False))
print(f'\n  Suma wag: {opt_weights.sum():.4f}')
print()


# ──────────────────────────────────────────────────────────────────────────────
# 8. WYKRES GRANICY EFEKTYWNEJ
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))

# Chmura portfeli
scatter = ax.scatter(
    results_risk * 100,
    results_return * 100,
    c=results_sharpe,
    cmap='viridis',
    marker='o',
    s=5,
    alpha=0.5,
    edgecolors='none'
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Wskaźnik Sharpe\'a', fontsize=12)

# „Złoty Środek" — portfel optymalny
ax.scatter(
    opt_risk * 100,
    opt_return * 100,
    c='gold',
    marker='*',
    s=600,
    edgecolors='black',
    linewidths=1.5,
    zorder=5,
    label=f'Złoty Środek (Sharpe = {opt_sharpe:.3f})'
)

ax.annotate(
    f'  Maks. Sharpe\n  mu={opt_return:.1%}  sigma={opt_risk:.1%}',
    xy=(opt_risk * 100, opt_return * 100),
    xytext=(opt_risk * 100 + 2, opt_return * 100 + 3),
    fontsize=10,
    fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='gray', alpha=0.9)
)

ax.set_title('Granica Efektywna Markowitza -- Symulacja Monte Carlo (50 000 portfeli)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Ryzyko roczne sigma_p [%]', fontsize=13)
ax.set_ylabel('Oczekiwany zwrot roczny mu_p [%]', fontsize=13)
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 9. WYKRES KOŁOWY WAG OPTYMALNYCH
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))

sorted_idx = np.argsort(opt_weights)[::-1]
sorted_weights = opt_weights[sorted_idx]
sorted_tickers = [tickers[i] for i in sorted_idx]

color_map = {
    'AMZN': '#FF6B35', 'ORCL': '#FF8C61', 'NVDA': '#FFB088',
    'HIMS': '#FFD4BC', 'AAPL': '#E85D04',
    'JNJ': '#2EC4B6', 'KO': '#52D1C6',
    'TLT': '#3A86FF', 'AGG': '#6DA6FF', 'GLD': '#FFD700'
}
colors = [color_map[t] for t in sorted_tickers]
explode = [0.05 if w == sorted_weights[0] else 0 for w in sorted_weights]

wedges, texts, autotexts = ax.pie(
    sorted_weights,
    labels=sorted_tickers,
    colors=colors,
    explode=explode,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
    pctdistance=0.75,
    startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)

for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

ax.set_title('Struktura portfela o maksymalnym wskazniku Sharpe\'a',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 10. TABELA PODSUMOWUJĄCA
# ──────────────────────────────────────────────────────────────────────────────
asset_class = {
    'AMZN': 'Tech', 'ORCL': 'Tech', 'NVDA': 'Tech', 'HIMS': 'Tech', 'AAPL': 'Tech',
    'JNJ': 'Defensywne', 'KO': 'Defensywne',
    'TLT': 'Obligacje', 'AGG': 'Obligacje', 'GLD': 'Złoto'
}

summary = pd.DataFrame({
    'Aktywo': tickers,
    'Klasa': [asset_class[t] for t in tickers],
    'Roczny zwrot [%]': (mu * 100).round(2).values,
    'Roczne ryzyko [%]': (daily_returns.std() * np.sqrt(TRADING_DAYS) * 100).round(2).values,
    'Waga optymalna [%]': (opt_weights * 100).round(2)
})

print('=' * 75)
print('  TABELA PODSUMOWUJACA')
print('=' * 75)
print(summary.to_string(index=False))
print()
print('-' * 80)
print(f'  Portfel optymalny: mu_p = {opt_return:.2%} | sigma_p = {opt_risk:.2%} | Sharpe = {opt_sharpe:.4f}')
print('=' * 80)
