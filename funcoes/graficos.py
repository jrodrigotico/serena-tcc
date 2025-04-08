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


