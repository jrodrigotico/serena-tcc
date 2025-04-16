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
        dados = dados.dropna(subset=[column_x, column_y])

        X_temp = dados[[column_x]].values
        x2 = X_temp ** 2
        # weekday = dados[['weekday']].values  # 0 = fim de semana, 1 = dia útil

        # Empilha as variáveis para regressão múltipla: x², x, weekday, constante
        X = np.hstack((x2, X_temp, np.ones_like(X_temp)))

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
        
        cv_rmse_percent = (rmse_cv_std / rmse_cv_mean) * 100 if rmse_cv_mean != 0 else 0        

        resultados.append({
            'hora': hora,
            'equacao': f"y = {coef[0]:.2f}x² {coef[1]:+.2f}x {coef[2]:+.2f}",
            'RMSE Treino': rmse_train,
            'R² Treino': r2_train,
            'RMSE Teste': rmse_test,
            'R² Teste': r2_test,
            'RMSE Validação': rmse_val,
            'R² Validação': r2_val,
            'CV RMSE Médio': rmse_cv_mean,
            'CV RMSE DP': rmse_cv_std,
            'CV RMSE %': cv_rmse_percent
        })

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

            X = dados[[col_temp]].values
            y = dados[col_carga].values

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Regressão quadrática
            X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
            coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

            # Previsões
            y_pred_train = X_train_poly @ coef
            X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
            y_pred_test = X_test_poly @ coef
            X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
            y_pred_val = X_val_poly @ coef

            # Validação cruzada no conjunto de validação
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
            cv_rmse_percent = (rmse_cv_std / rmse_cv_mean) * 100

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
                'CV RMSE Médio': rmse_cv_mean,
                'CV RMSE DP': rmse_cv_std,
                'CV RMSE %': cv_rmse_percent
            })

    return pd.DataFrame(resultados)


def avaliar_modelo_por_hora_cv(df, k=5):
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
    resultados = []

    for hora in horas:
        dados_hora = df[df['hour'] == hora]
        skip = False

        # Inicializa dicionários de treino/teste
        X_train_dict, y_train_dict = {}, {}
        X_test_dict, y_test_dict = {}, {}
        coef_dict = {}

        for regiao, (col_carga, col_temp) in regioes.items():
            dados = dados_hora.dropna(subset=[col_carga, col_temp])
            if len(dados) < 10:
                skip = True
                break

            X = dados[[col_temp]].values
            y = dados[col_carga].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
            coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

            X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
            X_val_poly = np.hstack((X**2, X, np.ones_like(X)))

            coef_dict[regiao] = coef
            X_train_dict[regiao] = X_train_poly
            X_test_dict[regiao] = X_test_poly
            y_train_dict[regiao] = y_train
            y_test_dict[regiao] = y_test

        if skip:
            continue

        # Soma das previsões por base
        y_train_pred_soma = np.zeros_like(y_train_dict['coast'])
        y_test_pred_soma = np.zeros_like(y_test_dict['coast'])
        y_val_pred_soma = np.zeros(len(dados_hora))

        y_train_soma = np.zeros_like(y_train_dict['coast'])
        y_test_soma = np.zeros_like(y_test_dict['coast'])
        y_val_soma = dados_hora[[r[0] for r in regioes.values()]].sum(axis=1).values

        for regiao in regioes:
            coef = coef_dict[regiao]
            y_train_pred = (X_train_dict[regiao] @ coef).flatten()
            y_test_pred = (X_test_dict[regiao] @ coef).flatten()

            y_train_pred_soma += y_train_pred
            y_test_pred_soma += y_test_pred

            y_train_soma += y_train_dict[regiao]
            y_test_soma += y_test_dict[regiao]

            X_val = dados_hora[[regioes[regiao][1]]].values
            X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
            y_val_pred = (X_val_poly @ coef).flatten()
            y_val_pred_soma += y_val_pred

        # Métricas
        rmse_train = mean_squared_error(y_train_soma, y_train_pred_soma, squared=False)
        rmse_test = mean_squared_error(y_test_soma, y_test_pred_soma, squared=False)
        rmse_val = mean_squared_error(y_val_soma, y_val_pred_soma, squared=False)

        r2_train = r2_score(y_train_soma, y_train_pred_soma)
        r2_test = r2_score(y_test_soma, y_test_pred_soma)
        r2_val = r2_score(y_val_soma, y_val_pred_soma)

        # Validação cruzada
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmse_cv_scores = []

        for train_idx, test_idx in kf.split(np.hstack((X_val_poly, dados_hora[['hour']].values))): # adicionando a hora como feature
            X_cv_train, X_cv_test = np.hstack((X_val_poly, dados_hora[['hour']].values))[train_idx], np.hstack((X_val_poly, dados_hora[['hour']].values))[test_idx]
            y_cv_train, y_cv_test = y_val_soma[train_idx], y_val_soma[test_idx]

            coef_cv, *_ = np.linalg.lstsq(X_cv_train, y_cv_train, rcond=None)
            y_cv_pred = X_cv_test @ coef_cv
            rmse_cv_scores.append(mean_squared_error(y_cv_test, y_cv_pred, squared=False))

        rmse_cv_mean = np.mean(rmse_cv_scores)
        rmse_cv_std = np.std(rmse_cv_scores)
        cv_rmse_percent = (rmse_cv_std / rmse_cv_mean) * 100 if rmse_cv_mean != 0 else 0

        resultado_hora = {
            'hora': hora,
            'rmse_treino': rmse_train,
            'rmse_teste': rmse_test,
            'rmse_validacao': rmse_val,
            'r2_treino': r2_train,
            'r2_teste': r2_test,
            'r2_validacao': r2_val,
            'CV RMSE Médio': rmse_cv_mean,
            'CV RMSE DP': rmse_cv_std,
            'CV RMSE %': cv_rmse_percent
        }
        resultados.append(resultado_hora)

    return pd.DataFrame(resultados)