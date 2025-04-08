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

