import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score


def treinamento_media_simples(df, k=5):
    resultados = []

    zonas = df['regiao'].unique()
    horas = sorted(df['hour'].unique())

    for zona in zonas:
        for hora in horas:
            # Filtra os dados da zona e hora atual
            dados = df[(df['regiao'] == zona) & (df['hour'] == hora)]
            dados = dados.dropna(subset=['temperatura', 'carga'])

            X = dados[['temperatura']].values
            y = dados['carga'].values

            # Divide em treino, teste e validação
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Regressão quadrática: x², x, constante
            X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
            coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

            # Previsões
            y_pred_train = X_train_poly @ coef
            X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
            y_pred_test = X_test_poly @ coef
            X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
            y_pred_val = X_val_poly @ coef

            # Métricas
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
            rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            r2_val = r2_score(y_val, y_pred_val)

            # Validação cruzada
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            rmse_cv_scores = []

            for train_idx, test_idx in kf.split(X_val):
                X_cv_train, X_cv_test = X_val[train_idx], X_val[test_idx]
                y_cv_train, y_cv_test = y_val[train_idx], y_val[test_idx]

                X_cv_train_poly = np.hstack((X_cv_train**2, X_cv_train, np.ones_like(X_cv_train)))
                X_cv_test_poly = np.hstack((X_cv_test**2, X_cv_test, np.ones_like(X_cv_test)))

                coef_cv, *_ = np.linalg.lstsq(X_cv_train_poly, y_cv_train, rcond=None)
                y_cv_pred = X_cv_test_poly @ coef_cv
                rmse_cv_scores.append(mean_squared_error(y_cv_test, y_cv_pred, squared=False))

            rmse_cv_mean = np.mean(rmse_cv_scores)
            rmse_cv_std = np.std(rmse_cv_scores)

            resultados.append({
                'regiao': zona,
                'hora': hora,
                'equacao': f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}",
                'RMSE Treino': rmse_train,
                'R² Treino': r2_train,
                'RMSE Teste': rmse_test,
                'R² Teste': r2_test,
                'RMSE Validação': rmse_val,
                'R² Validação': r2_val,
                'CV RMSE Médio': rmse_cv_mean,
                'CV RMSE DP': rmse_cv_std
            })

    return pd.DataFrame(resultados)


def treinamento_ponderado_por_zona_e_hora(df, k=5):
    resultados = []

    zonas = df['regiao'].unique()
    horas = sorted(df['hour'].unique())

    for zona in zonas:
        for hora in horas:
            dados = df[(df['regiao'] == zona) & (df['hour'] == hora)]
            dados = dados.dropna(subset=['temp_ponderada_pop', 'carga'])

            X = dados[['temp_ponderada_pop']].values
            y = dados['carga'].values

            # Divisão 70/15/15
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Quadrática
            X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
            coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

            # Previsões
            y_pred_train = X_train_poly @ coef
            X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
            y_pred_test = X_test_poly @ coef
            X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
            y_pred_val = X_val_poly @ coef

            # Métricas
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            r2_train = r2_score(y_train, y_pred_train)
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
            r2_val = r2_score(y_val, y_pred_val)

            # Cross-validation na validação
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            rmse_cv_scores = []

            for train_idx, test_idx in kf.split(X_val):
                X_cv_train, X_cv_test = X_val[train_idx], X_val[test_idx]
                y_cv_train, y_cv_test = y_val[train_idx], y_val[test_idx]

                X_cv_train_poly = np.hstack((X_cv_train**2, X_cv_train, np.ones_like(X_cv_train)))
                X_cv_test_poly = np.hstack((X_cv_test**2, X_cv_test, np.ones_like(X_cv_test)))

                coef_cv, *_ = np.linalg.lstsq(X_cv_train_poly, y_cv_train, rcond=None)
                y_cv_pred = X_cv_test_poly @ coef_cv
                rmse_cv_scores.append(mean_squared_error(y_cv_test, y_cv_pred, squared=False))

            rmse_cv_mean = np.mean(rmse_cv_scores)
            rmse_cv_std = np.std(rmse_cv_scores)

            resultados.append({
                'regiao': zona,
                'hora': hora,
                'equacao': f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}",
                'RMSE Treino': rmse_train,
                'R² Treino': r2_train,
                'RMSE Teste': rmse_test,
                'R² Teste': r2_test,
                'RMSE Validação': rmse_val,
                'R² Validação': r2_val,
                'CV RMSE Médio': rmse_cv_mean,
                'CV RMSE DP': rmse_cv_std
            })

    return pd.DataFrame(resultados)


def treinamento_por_weather_zone_e_hora_cv(df, weekday=None, k=5):
    zonas = ['coast', 'east', 'far_west', 'north', 'north_central',
             'south_central', 'southern', 'west']
    
    horas = sorted(df['hour'].unique())
    resultados = []

    for zona in zonas:
        temp_col = f"{zona}_tc"
        carga_col = f"{zona}_carga"

        for hora in horas:
            dados = df.copy()

            # Filtros
            if weekday is not None:
                dados = dados[dados['weekday'] == weekday]
            dados = dados[dados['hour'] == hora]
            dados = dados[dados[temp_col].notna() & dados[carga_col].notna()]

            X = dados[[temp_col]].values
            y = dados[carga_col].values

            # Divisão dos dados
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, X_val = train_test_split(X_temp, test_size=0.5, random_state=42)
            y_test, y_val = train_test_split(y_temp, test_size=0.5, random_state=42)

            # Quadrática
            X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
            coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

            # Previsões
            y_pred_train = X_train_poly @ coef
            X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
            y_pred_test = X_test_poly @ coef
            X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
            y_pred_val = X_val_poly @ coef

            # Métricas
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            r2_train = r2_score(y_train, y_pred_train)
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
            r2_val = r2_score(y_val, y_pred_val)

            # Validação cruzada na base de validação
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            rmse_cv_scores = []

            for train_idx, test_idx in kf.split(X_val):
                X_cv_train, X_cv_test = X_val[train_idx], X_val[test_idx]
                y_cv_train, y_cv_test = y_val[train_idx], y_val[test_idx]

                X_cv_train_poly = np.hstack((X_cv_train**2, X_cv_train, np.ones_like(X_cv_train)))
                X_cv_test_poly = np.hstack((X_cv_test**2, X_cv_test, np.ones_like(X_cv_test)))

                coef_cv, *_ = np.linalg.lstsq(X_cv_train_poly, y_cv_train, rcond=None)
                y_cv_pred = X_cv_test_poly @ coef_cv
                rmse_cv_scores.append(mean_squared_error(y_cv_test, y_cv_pred, squared=False))

            rmse_cv_mean = np.mean(rmse_cv_scores)
            rmse_cv_std = np.std(rmse_cv_scores)

            # Armazenar resultados
            resultados.append({
                'weather_zone': zona,
                'hora': hora,
                'equacao': f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}",
                'RMSE Treino': rmse_train,
                'R² Treino': r2_train,
                'RMSE Teste': rmse_test,
                'R² Teste': r2_test,
                'RMSE Validação': rmse_val,
                'R² Validação': r2_val,
                'CV RMSE Médio': rmse_cv_mean,
                'CV RMSE DP': rmse_cv_std
            })

    return pd.DataFrame(resultados)



