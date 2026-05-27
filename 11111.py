import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Pobranie danych giełdowych (szereg czasowy cen zamknięcia)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
print(f"Pobieranie danych dla: {tickers}...")
data = yf.download(tickers, start='2018-01-01', end='2024-01-01')['Close']

# Obliczenie dziennych logarytmicznych stóp zwrotu
returns = np.log(data / data.shift(1)).dropna()

# 2. Podział na próbę uczącą i testową
split_date = '2022-01-01'
train_returns = returns.loc[:split_date]
test_returns = returns.loc[split_date:]

# 3. Estymacja parametrów modelu Markowitza (tylko na próbie uczącej)
N = len(tickers)
# Wektor oczekiwanych stóp zwrotu (wymiar N x 1), annualizowany (252 dni sesyjne)
mu = train_returns.mean().values.reshape(-1, 1) * 252
# Macierz kowariancji (wymiar N x N), annualizowana
Sigma = train_returns.cov().values * 252

# Wektor jedynek (wymiar N x 1) do warunku w_i = 1
ones = np.ones((N, 1))

# 4. Konstrukcja i rozwiązanie układu równań liniowych
# Założenie docelowej stopy zwrotu (np. średnia z oczekiwanych zwrotów)
mu_p = np.mean(mu)

# Budowa macierzy blokowej A o wymiarach (N+2) x (N+2)
# Postać macierzy z metody mnożników Lagrange'a:
# A = [ Sigma,  -mu, -ones ]
#     [ mu.T,    0,     0  ]
#     [ ones.T,  0,     0  ]
top_row = np.hstack((Sigma, -mu, -ones))
mid_row = np.hstack((mu.T, [[0]], [[0]]))
bot_row = np.hstack((ones.T, [[0]], [[0]]))
A = np.vstack((top_row, mid_row, bot_row))

# Budowa wektora b o wymiarach (N+2) x 1
# b = [ 0, 0, ..., 0, mu_p, 1 ].T
b = np.vstack((np.zeros((N, 1)), [[mu_p]], [[1]]))

# Numeryczne rozwiązanie układu Ax = b
x = np.linalg.solve(A, b)

# Wyodrębnienie wektora optymalnych wag z wektora wynikowego x
w_opt = x[:N]

print("\nWektor optymalnych wag (Markowitz):")
for ticker, weight in zip(tickers, w_opt):
    print(f"{ticker}: {weight[0]:.4f}")

# 5. Weryfikacja empiryczna na próbie testowej (out-of-sample)
# Konstrukcja portfela naiwnego jako benchmarku (wagi po równo: 1/N)
w_eq = np.ones((N, 1)) / N

# Operacja mnożenia macierzy: R_p = R_test * w
port_opt_returns = test_returns.values @ w_opt
port_eq_returns = test_returns.values @ w_eq

# Obliczenie skumulowanych zwrotów dla wizualizacji
cumulative_opt = np.exp(np.cumsum(port_opt_returns))
cumulative_eq = np.exp(np.cumsum(port_eq_returns))

# 6. Wizualizacja wyników
plt.figure(figsize=(12, 6))
plt.plot(test_returns.index, cumulative_opt, label='Portfel Zoptymalizowany (Markowitz)', color='blue')
plt.plot(test_returns.index, cumulative_eq, label='Portfel Naiwny (1/N)', color='orange', linestyle='--')
plt.title('Backtesting - Weryfikacja Empiryczna Portfela na Próbie Testowej')
plt.xlabel('Data')
plt.ylabel('Wartość portfela (Kapitał początkowy = 1)')
plt.legend()
plt.grid(True)
plt.show()