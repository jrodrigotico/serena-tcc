import pandas as pd
import numpy as np

def tabela_equacoes1(df, n_weekday):
    
    # Listas de colunas (colocando total_tc em primeiro)
    colunas_tc = ['avg_temp_celsius']

    colunas_carga = ['sum_load']

    # Criar um dicionário para armazenar as equações
    resultados = {"Hora": list(range(24))}

    # Loop para cada coluna de temperatura e carga
    for temp_col, carga_col in zip(colunas_tc, colunas_carga):
        equacoes = []
        
        for hora1 in range(24):
            temp1 = df[(df['hour'] == hora1) & (df[temp_col] < 105) & (df['weekday'] ==n_weekday)]
            
            if len(temp1) < 3:  # Evitar erro com poucos pontos para regressão quadrática
                equacoes.append("Dados insuficientes")
                continue

            # Ajustar a regressão quadrática
            coef = np.polyfit(temp1[temp_col], temp1[carga_col], 2)

            # Arredondar coeficientes para 2 casas decimais
            coef = [round(c, 2) for c in coef]

            # Formatar a equação como string
            equacao = f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {coef[2]:.2f}"
            equacoes.append(equacao)

        # Adicionar coluna ao dicionário
        resultados[temp_col] = equacoes

    # Criar DataFrame final
    df_equacoes = pd.DataFrame(resultados)

    # # Garantir que total_tc seja a primeira coluna após "Hora"
    # colunas_ordenadas = ["Hora", "total_tc"] + [col for col in colunas_tc if col != "total_tc"]
    # df_equacoes = df_equacoes[colunas_ordenadas]

    # Exibir a tabela
    return df_equacoes



def tabela_equacoes3(df, n_weekday):
    
    # Listas de colunas (colocando total_tc em primeiro)
    colunas_tc = ['avg_temp_celsius', 'coast_tc', 'east_tc', 'far_west_tc', 'north_tc', 
                'north_central_tc', 'south_central_tc', 'southern_tc', 'west_tc']

    colunas_carga = ['sum_load', 'coast_carga', 'east_carga', 'far_west_carga', 'north_carga',
                    'north_central_carga', 'south_central_carga', 'southern_carga', 'west_carga']

    # Criar um dicionário para armazenar as equações
    resultados = {"Hora": list(range(24))}

    # Loop para cada coluna de temperatura e carga
    for temp_col, carga_col in zip(colunas_tc, colunas_carga):
        equacoes = []
        
        for hora1 in range(24):
            temp1 = df[(df['hour'] == hora1) & (df[temp_col] < 105) & (df['weekday'] ==n_weekday)]
            
            if len(temp1) < 3:  # Evitar erro com poucos pontos para regressão quadrática
                equacoes.append("Dados insuficientes")
                continue

            # Ajustar a regressão quadrática
            coef = np.polyfit(temp1[temp_col], temp1[carga_col], 2)

            # Arredondar coeficientes para 2 casas decimais
            coef = [round(c, 2) for c in coef]

            # Formatar a equação como string
            equacao = f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {coef[2]:.2f}"
            equacoes.append(equacao)

        # Adicionar coluna ao dicionário
        resultados[temp_col] = equacoes

    # Criar DataFrame final
    df_equacoes = pd.DataFrame(resultados)

    # # Garantir que total_tc seja a primeira coluna após "Hora"
    # colunas_ordenadas = ["Hora", "total_tc"] + [col for col in colunas_tc if col != "total_tc"]
    # df_equacoes = df_equacoes[colunas_ordenadas]

    # Exibir a tabela
    return df_equacoes




