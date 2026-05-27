import numpy as np
import pandas as pd


def wyznacz_macierz_kowariancji(stopy_zwrotu: pd.DataFrame, annualizuj: bool = True,
                                dni_sesyjne: int = 252) -> pd.DataFrame:


    # Wyciągamy surowe wartości z DataFrame do szybkiej macierzy numpy
    R = stopy_zwrotu.to_numpy()

    # Pobieramy wymiary: T (liczba okresów/dni), n (liczba spółek)
    T, n = R.shape

    # KROK 1 i 2: Obliczamy średnią dla każdej kolumny i centrujemy dane
    # R - np.mean odejmuje średnią od każdego elementu w danej kolumnie
    X = R - np.mean(R, axis=0)

    # KROK 3: Mnożenie macierzy (X^T * X) podzielone przez (T - 1)
    Sigma = (X.T @ X) / (T - 1)

    if annualizuj:
        Sigma = Sigma * dni_sesyjne

    # Pakujemy wynik z powrotem do DataFrame, żeby zachować ładne nazwy kolumn i wierszy
    macierz_df = pd.DataFrame(Sigma, index=stopy_zwrotu.columns, columns=stopy_zwrotu.columns)

    return macierz_df