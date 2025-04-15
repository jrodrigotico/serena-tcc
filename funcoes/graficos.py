import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plotar_graficos(df, hour, weekday, region):
    # Inferir a coluna de carga correspondente
    if region == "avg_temp_celsius":
        carga_col = "sum_load"
    else:
        carga_col = region.replace("_tc", "_carga")

    # Filtragem
    cenario_temp = df[(df['hour'] == hour) & 
                      (df['weekday'] == weekday) & 
                      (df[region] < 105)]

    plt.figure(figsize=(10, 5))

    # Scatter plot
    sns.scatterplot(data=cenario_temp, x=region, y=carga_col, label="Pontos", marker='o', color='blue')

    # Ajuste da regressão quadrática
    coef = np.polyfit(cenario_temp[region], cenario_temp[carga_col], 2)
    linha_regressao = np.poly1d(coef)

    x_vals = np.linspace(cenario_temp[region].min(), cenario_temp[region].max(), 1000)
    y_vals = linha_regressao(x_vals)

    plt.plot(x_vals, y_vals, color="red", 
             label=f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {coef[2]:.2f}")

    # Personalização
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Carga")
    plt.title(f"Temperatura vs Carga\nHora: {hour}, Weekday: {weekday}, Region: {region}")
    plt.legend()
    plt.grid(True)
    plt.show()


def grid_graficos(df, hour, weekday=1):
    # Lista das colunas de temperatura e o total
    regioes = ['avg_temp_celsius', 'coast_tc', 'east_tc', 'far_west_tc', 'north_tc', 
            'north_central_tc', 'south_central_tc', 'southern_tc', 'west_tc']

    # Criar o grid 3x3
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()

    for i, region in enumerate(regioes):
        ax = axs[i]

        # Inferir a coluna de carga correspondente
        carga_col = "sum_load" if region == "avg_temp_celsius" else region.replace("_tc", "_carga")

        # Filtrar os dados
        dados = df[(df['hour'] == hour) & 
                   (df['weekday'] == weekday) & 
                   (df[region] < 105)]

        if len(dados) < 3:
            ax.set_title(f"{region}\nDados insuficientes")
            ax.axis('off')
            continue

        # Scatter
        sns.scatterplot(data=dados, x=region, y=carga_col, ax=ax, color="blue", s=20, label="Pontos")

        # Regressão quadrática
        coef = np.polyfit(dados[region], dados[carga_col], 2)
        p = np.poly1d(coef)

        x_vals = np.linspace(dados[region].min(), dados[region].max(), 100)
        y_vals = p(x_vals)

        ax.plot(x_vals, y_vals, color='red',
                label=f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {coef[2]:.2f}")
        
        ax.set_title(f"{region}")
        ax.set_xlabel("Temp (°C)")
        ax.set_ylabel("Carga")
        ax.grid(True)
        ax.legend()

    # Ajustes finais
    plt.suptitle(f"Temperatura vs Carga - Hora {hour}, Weekday {weekday}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def grid_graficos_estacoes(df, estacao, hour=15, weekday=1):
    # Lista das colunas de temperatura
    regioes = ['avg_temp_celsius', 'coast_tc', 'east_tc', 'far_west_tc', 'north_tc', 
               'north_central_tc', 'south_central_tc', 'southern_tc', 'west_tc']

    # Criar o grid 3x3
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()

    for i, region in enumerate(regioes):
        ax = axs[i]

        # Inferir a coluna de carga correspondente
        carga_col = "sum_load" if region == "avg_temp_celsius" else region.replace("_tc", "_carga")

        # Filtrar dados da estação, hora e dia da semana
        dados = df[
            (df['estacao'] == estacao) &
            (df['hour'] == hour) &
            (df['weekday'] == weekday) &
            (df[region] < 105)
        ]

        if len(dados) < 3:
            ax.set_title(f"{region}\nDados insuficientes")
            ax.axis('off')
            continue

        # Scatterplot
        sns.scatterplot(data=dados, x=region, y=carga_col, ax=ax, color="blue", s=20, label="Pontos")

        # Regressão quadrática
        coef = np.polyfit(dados[region], dados[carga_col], 2)
        p = np.poly1d(coef)

        x_vals = np.linspace(dados[region].min(), dados[region].max(), 100)
        y_vals = p(x_vals)

        ax.plot(x_vals, y_vals, color='red',
                label=f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {coef[2]:.2f}")

        # Título e eixos
        nome_formatado = region.replace('_tc', '').replace('_', ' ').title()
        ax.set_title(nome_formatado)
        ax.set_xlabel("Temp (°C)")
        ax.set_ylabel("Carga")
        ax.grid(True)
        ax.legend(fontsize='small')

    # Ajustes finais
    plt.suptitle(f"Temperatura vs Carga - Estação: {estacao}, Hora: {hour}, Weekday: {weekday}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_metricas_por_hora(df_resultados):
    metricas = [
        'RMSE Treino', 'RMSE Teste', 'RMSE Validação',
        'R² Treino', 'R² Teste', 'R² Validação',
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for i, metrica in enumerate(metricas):
        ax = axes[i]
        sns.lineplot(data=df_resultados, x='hora', y=metrica, marker='o', ax=ax)
        ax.set_title('', fontsize=12)
        ax.set_xlabel('Hora')
        ax.set_ylabel(metrica)
        ax.grid(True)

        # Adiciona os valores nas linhas
        # for x, y in zip(df_resultados['hora'], df_resultados[metrica]):
        #     ax.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_heatmap(df_resultados, metrica):
    tabela = df_resultados.pivot(index='região', columns='hora', values=metrica)
    plt.figure(figsize=(20, 6))
    sns.heatmap(tabela, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f"Heatmap de {metrica} por região e hora")
    plt.xlabel("Hora")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_cv_rmse_percent(df_resultados):
    plt.figure(figsize=(12, 6))
    plt.plot(df_resultados['hora'], df_resultados['CV RMSE %'], marker='o', linestyle='-', color='darkorange')
    plt.title('Desvio Padrão Relativo do RMSE (CV RMSE %) por Hora')
    plt.xlabel('Hora do Dia')
    plt.ylabel('CV RMSE (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(df_resultados['hora'])  # Garante que todas as horas estejam no eixo x
    for i, val in enumerate(df_resultados['CV RMSE %']):
        plt.text(df_resultados['hora'][i], val + 0.5, f"{val:.1f}%", ha='center', fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cv_rmse_percent_todas_regioes(df_resultados):
    regioes = df_resultados['região'].unique()
    df_resultados = df_resultados.sort_values(by='hora')

    plt.figure(figsize=(14, 7))

    for regiao in regioes:
        df_regiao = df_resultados[df_resultados['região'] == regiao]
        plt.plot(
            df_regiao['hora'], df_regiao['CV RMSE %'],
            marker='o', linestyle='-', label=regiao.replace('_', ' ').capitalize()
        )

    plt.title('Desvio Padrão Relativo do RMSE (CV RMSE %) por Hora - Todas as Regiões')
    plt.xlabel('Hora do Dia')
    plt.ylabel('CV RMSE %')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(sorted(df_resultados['hora'].unique()))
    plt.legend(title='Região', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


