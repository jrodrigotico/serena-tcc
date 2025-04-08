import pandas as pd
import numpy as np

def tabela_equacoes(df, n_weekday):
    # Listas de colunas (temperatura e carga)
    colunas_tc = ['avg_temp_celsius', 'coast_tc', 'east_tc', 'far_west_tc', 'north_tc', 
                  'north_central_tc', 'south_central_tc', 'southern_tc', 'west_tc']
    
    colunas_carga = ['sum_load', 'coast_carga', 'east_carga', 'far_west_carga', 'north_carga',
                     'north_central_carga', 'south_central_carga', 'southern_carga', 'west_carga']
    
    # Dicionário para armazenar as equações por hora
    resultados = {"Hora": list(range(24))}

    # Loop por região (temperatura x carga)
    for temp_col, carga_col in zip(colunas_tc, colunas_carga):
        print(temp_col, carga_col)
 
        equacoes = []

        for hora1 in range(24):
            # Filtro de dados para a hora e dia da semana desejado
            filtro = (df['hour'] == hora1) & (df[temp_col] < 105) & (df['weekday'] == n_weekday)
            dados_filtrados = df[filtro]

            # Verificar quantidade mínima de dados
            n = dados_filtrados.shape[0]
            if n < 3:
                equacoes.append(f"Insuficiente ({n} dados)")
                continue

            # Regressão quadrática
            try:
                coef = np.polyfit(dados_filtrados[temp_col], dados_filtrados[carga_col], 2)
                coef = [round(c, 2) for c in coef]
                equacao = f"y = {coef[0]:.2f}x² {coef[1]:.2f}x + {coef[2]:.2f}"
                equacoes.append(equacao)
            except Exception as e:
                equacoes.append("Erro")

        # Adicionar as equações da coluna atual ao dicionário final
        resultados[temp_col] = equacoes

    # Criar DataFrame com os resultados
    df_equacoes = pd.DataFrame(resultados)

    return df_equacoes
