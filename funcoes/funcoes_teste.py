import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score


def treinamento_estado(df, column_x, column_y, k=5):
    resultados = []

    horas = sorted(df['hour'].unique())

    for hora in horas:
        # Filtra os dados para a hora atual
        dados = df[df['hour'] == hora]
        dados = dados.dropna(subset=[column_x, column_y, 'weekday'])

        X_temp = dados[[column_x]].values
        x2 = X_temp ** 2
        weekday = dados[['weekday']].values  # 0 = fim de semana, 1 = dia útil

        # Empilha as variáveis para regressão múltipla: x², x, weekday, constante
        X = np.hstack((x2, X_temp, weekday, np.ones_like(X_temp)))

        y = dados[column_y].values

        # Divide em treino, teste e validação
        X_train, X_temp_split, y_train, y_temp_split = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp_split, y_temp_split, test_size=0.5, random_state=42)

        # Ajuste dos coeficientes
        coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

        # Previsões
        y_pred_train = X_train @ coef
        y_pred_test = X_test @ coef
        y_pred_val = X_val @ coef

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

            coef_cv, *_ = np.linalg.lstsq(X_cv_train, y_cv_train, rcond=None)
            y_cv_pred = X_cv_test @ coef_cv
            rmse_cv_scores.append(mean_squared_error(y_cv_test, y_cv_pred, squared=False))

        rmse_cv_mean = np.mean(rmse_cv_scores)
        rmse_cv_std = np.std(rmse_cv_scores)

        resultados.append({
            'hora': hora,
            'equacao': f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}weekday {coef[3]:+.2f}",
            'RMSE Treino': rmse_train,
            'R² Treino': r2_train,
            'RMSE Teste': rmse_test,
            'R² Teste': r2_test,
            'RMSE Validação': rmse_val,
            'R² Validação': r2_val,
            # 'CV RMSE Médio': rmse_cv_mean,
            # 'CV RMSE DP': rmse_cv_std
        })

    return pd.DataFrame(resultados)


def treinamento_regioes(df, k=5):
    resultados = []

    # Definindo as colunas de carga e temperatura para cada região
    regioes = {
        'coast': ('coast_carga', 'coast_tc'),
        'east': ('east_carga', 'east_tc'),
        'far_west': ('far_west_carga', 'far_west_tc'),
        'north': ('north_carga', 'north_tc'),
        'north_central': ('north_central_carga', 'north_central_tc'),
        'south_central': ('south_central_carga', 'south_central_tc'),
        'southern': ('southern_carga', 'southern_tc'),
        'west': ('west_carga', 'west_tc')
    }

    horas = sorted(df['hour'].unique())

    # Itera sobre as horas
    for hora in horas:
        # Inicializa a lista de resultados para cada hora
        resultado_hora = {'hora': hora}

        # Itera sobre as regiões
        for regiao, (col_carga, col_temp) in regioes.items():
            # Filtra os dados para a hora e a região atual
            dados = df[df['hour'] == hora]
            dados = dados.dropna(subset=[col_carga, col_temp])  # Filtrando os dados

            X = dados[[col_temp]].values  # Temperatura da região
            y = dados[col_carga].values  # Carga da região

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

            # Métricas de erro
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
            rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            r2_val = r2_score(y_val, y_pred_val)
            
            # Adiciona a equação para a região
            resultado_hora[f'Equação {regiao}'] = f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}"

            # Adiciona as métricas para a região
            resultado_hora[f'RMSE {regiao} Treino'] = rmse_train
            resultado_hora[f'R² {regiao} Treino'] = r2_train
            resultado_hora[f'RMSE {regiao} Teste'] = rmse_test
            resultado_hora[f'R² {regiao} Teste'] = r2_test
            resultado_hora[f'RMSE {regiao} Validação'] = rmse_val
            resultado_hora[f'R² {regiao} Validação'] = r2_val



        # Adiciona os resultados para a hora
        resultados.append(resultado_hora)

    return pd.DataFrame(resultados)



def treinamento_regioes_formatado(df, k=5):
    resultados = []

    regioes = {
        'coast': ('coast_carga', 'coast_tc'),
        'east': ('east_carga', 'east_tc'),
        'far_west': ('far_west_carga', 'far_west_tc'),
        'north': ('north_carga', 'north_tc'),
        'north_central': ('north_central_carga', 'north_central_tc'),
        'south_central': ('south_central_carga', 'south_central_tc'),
        'southern': ('southern_carga', 'southern_tc'),
        'west': ('west_carga', 'west_tc')
    }

    horas = sorted(df['hour'].unique())

    for hora in horas:
        for regiao, (col_carga, col_temp) in regioes.items():
            dados = df[df['hour'] == hora].dropna(subset=[col_carga, col_temp])

            if len(dados) < k:
                continue  # Pula se não houver dados suficientes

            X = dados[[col_temp]].values
            y = dados[col_carga].values

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
            coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

            y_pred_train = X_train_poly @ coef
            X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
            y_pred_test = X_test_poly @ coef
            X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
            y_pred_val = X_val_poly @ coef

            resultados.append({
                'hora': hora,
                'região': regiao,
                'equação': f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}",
                'RMSE Treino': mean_squared_error(y_train, y_pred_train, squared=False),
                'R² Treino': r2_score(y_train, y_pred_train),
                'RMSE Teste': mean_squared_error(y_test, y_pred_test, squared=False),
                'R² Teste': r2_score(y_test, y_pred_test),
                'RMSE Validação': mean_squared_error(y_val, y_pred_val, squared=False),
                'R² Validação': r2_score(y_val, y_pred_val),
            })

    return pd.DataFrame(resultados)