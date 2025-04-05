import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plotar_graficos(df, hour, weekday):
    
    cenario_temp = df[(df['hour']==hour) & (df['weekday']>=weekday) & (df['avg_temp_celsius']<105) ]

    # Criar a figura
    plt.figure(figsize=(10, 5))

    # Scatter plot (pontos)
    sns.scatterplot(data=cenario_temp, x="avg_temp_celsius", y="sum_load", label="Pontos", marker='o', color='blue')


    coef = np.polyfit(cenario_temp["avg_temp_celsius"], cenario_temp["sum_load"], 2)  # par[abola]
    linha_regressao = np.poly1d(coef)  

    # Gerar valores para plotar a linha
    x_vals = np.linspace(cenario_temp["avg_temp_celsius"].min(), cenario_temp["avg_temp_celsius"].max(), 1000)
    y_vals = linha_regressao(x_vals)

    # Plotar linha de regressão corretamente
    plt.plot(x_vals, y_vals, color="red", 
                label=f"y = {coef[0]:.2f}x² + {coef[1]:.2f}x + {coef[2]:.2f}")

    # Personalizar gráfico
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Carga")
    plt.title(f"Regressão entre Temperatura e Carga vs Hora {hour} e WeekDay {weekday}")
    plt.legend()
    plt.grid(True)

    # Exibir o gráfico
    plt.show()
