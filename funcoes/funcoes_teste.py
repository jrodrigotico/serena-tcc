import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score


def treinamento_media_simples(df, weekday=None, hour=None, k=5):
    resultados = []

    # Aplicar filtros antes do agrupamento
    if weekday is not None:
        df = df[df['weekday'] == weekday]
    if hour is not None:
        df = df[df['hour'] == hour]

    # Agrupar por horário e tirar a média simples das temperaturas por horário
    df_media = df.groupby('interval_start_utc').agg({
        'temperatura': 'mean',
        'carga': 'first'  # A carga já está associada ao horário (não varia por região)
    }).reset_index()

    X = df_media[['temperatura']].values
    y = df_media['carga'].values

    # Preparar termos quadráticos
    X_poly = np.hstack((X**2, X, np.ones_like(X)))

    # Divisão: 70% treino | 15% teste | 15% validação
    X_train, X_temp, y_train, y_temp = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Ajuste com MQO (usando base de treino)
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

    # Cross-validation na base de validação
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_cv_scores = []
    for train_index, test_index in kf.split(X_val):
        X_cv_train, X_cv_test = X_val[train_index], X_val[test_index]
        y_cv_train, y_cv_test = y_val[train_index], y_val[test_index]

        coef_cv, *_ = np.linalg.lstsq(X_cv_train, y_cv_train, rcond=None)
        y_cv_pred = X_cv_test @ coef_cv
        rmse_cv = mean_squared_error(y_cv_test, y_cv_pred, squared=False)
        rmse_cv_scores.append(rmse_cv)

    rmse_cv_mean = np.mean(rmse_cv_scores)
    rmse_cv_std = np.std(rmse_cv_scores)

    # Resultados para tabela
    resultados.append({
        'equacao': f"y = {int(round(coef[0]))}x² + {int(round(coef[1]))}x + {int(round(coef[2]))}",
        'RMSE Treino': rmse_train,
        'RMSE Teste': rmse_test,
        'RMSE Validação': rmse_val,
        'R² Treino': r2_train,
        'R² Teste': r2_test,
        'R² Validação': r2_val,
        'CV RMSE Médio': rmse_cv_mean,
        'CV RMSE DP': rmse_cv_std
    })

    return pd.DataFrame(resultados)




def treinamento_ponderado_por_zona_cv(df_long, weekday=None, hour=None, k=5):
    resultados = []

    # Filtros opcionais
    if weekday is not None:
        df_long = df_long[df_long['weekday'] == weekday]
    if hour is not None:
        df_long = df_long[df_long['hour'] == hour]

    zonas = df_long['regiao'].unique()

    for zona in zonas:
        dados = df_long[df_long['regiao'] == zona].dropna(subset=['temp_ponderada_pop', 'carga'])
        X = dados[['temp_ponderada_pop']].values
        y = dados['carga'].values

        # Divisão 70% treino | 15% teste | 15% validação
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Regressão quadrática na base de treino
        X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
        coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

        # --- Previsões nas 3 bases ---
        # Treino
        y_pred_train = X_train_poly @ coef
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        r2_train = r2_score(y_train, y_pred_train)

        # Teste
        X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
        y_pred_test = X_test_poly @ coef
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        r2_test = r2_score(y_test, y_pred_test)

        # Validação (sem cross)
        X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
        y_pred_val = X_val_poly @ coef
        rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
        r2_val = r2_score(y_val, y_pred_val)

        # --- Validação cruzada (k-fold) na base de validação ---
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

        # --- Armazenar resultados ---
        resultados.append({
            'regiao': zona,
            'equacao': f"y = {coef[0]:.2f}x² {coef[1]:.2f}x + {coef[2]:.2f}",
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



def treinamento_por_weather_zone_cv(df, weekday=None, hora=None, k=5):
    zonas = ['coast', 'east', 'far_west', 'north', 'north_central',
             'south_central', 'southern', 'west']

    resultados = []

    for zona in zonas:
        temp_col = f"{zona}_tc"
        carga_col = f"{zona}_carga"

        # Filtrar dados
        dados = df.copy()
        if weekday is not None:
            dados = dados[dados['weekday'] == weekday]
        if hora is not None:
            dados = dados[dados['hour'] == hora]
        dados = dados[dados[temp_col].notna() & dados[carga_col].notna()]

        X = dados[[temp_col]].values
        y = dados[carga_col].values

        if len(X) < 10:
            print(f"Poucos dados para a zona {zona}. Pulando...")
            continue

        # Divisão dos dados
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Regressão quadrática na base de treino
        X_train_poly = np.hstack((X_train**2, X_train, np.ones_like(X_train)))
        coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

        # Treino
        y_pred_train = X_train_poly @ coef
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        r2_train = r2_score(y_train, y_pred_train)

        # Teste
        X_test_poly = np.hstack((X_test**2, X_test, np.ones_like(X_test)))
        y_pred_test = X_test_poly @ coef
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        r2_test = r2_score(y_test, y_pred_test)

        # Validação (sem cross)
        X_val_poly = np.hstack((X_val**2, X_val, np.ones_like(X_val)))
        y_pred_val = X_val_poly @ coef
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
            'equacao': f"y = {coef[0]:.2f}x² {coef[1]:.2f}x + {coef[2]:.2f}",
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


