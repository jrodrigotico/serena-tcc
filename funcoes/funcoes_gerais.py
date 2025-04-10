def trocar_temp(df):
    colunas_fahrenheit = [
        'coast_tf',
        'east_tf',
        'far_west_tf',
        'north_tf',
        'north_central_tf',
        'south_central_tf',
        'southern_tf',
        'west_tf',
        'avg_temp_fahrenheit'
    ]

    for col in colunas_fahrenheit:
        if col == 'avg_temp_fahrenheit':
            nova_col = 'avg_temp_celsius'
        else:
            nova_col = col.replace('_tf', '_tc')

        df[nova_col] = (df[col] - 32) * (5 / 9)

    return df


def preparar_df_long_com_ponderada(df, df_pop, weekday=None, hour=None):
    zonas = ['coast', 'east', 'far_west', 'north', 'north_central',
             'south_central', 'southern', 'west']

    # Calcular os pesos com base na população
    pesos = {
        zona: df_pop[df_pop['LOAD_ZONE'].str.lower() == zona]['POPULATION'].sum()
        for zona in zonas
    }
    soma_total_pesos = sum(pesos.values())

    # Calcular temp_ponderada_pop
    df['temp_ponderada_pop'] = sum(df[f'{zona}_tc'] * pesos[zona] for zona in zonas) / soma_total_pesos

    # Aplicar filtros, se existirem
    if weekday is not None:
        df = df[df['weekday'] == weekday]
    if hour is not None:
        df = df[df['hour'] == hour]

    # Melt de temperatura
    df_temp = df.melt(
        id_vars=['interval_start_utc', 'hour', 'weekday', 'temp_ponderada_pop'],
        value_vars=[f"{zona}_tc" for zona in zonas],
        var_name='regiao',
        value_name='temperatura'
    )

    # Melt de carga
    df_carga = df.melt(
        id_vars=['interval_start_utc'],
        value_vars=[f"{zona}_carga" for zona in zonas],
        var_name='regiao',
        value_name='carga'
    )

    # Limpeza dos nomes das regiões
    df_temp['regiao'] = df_temp['regiao'].str.replace('_tc', '', regex=False)
    df_carga['regiao'] = df_carga['regiao'].str.replace('_carga', '', regex=False)

    # Merge final
    df_long = df_temp.merge(df_carga, on=['interval_start_utc', 'regiao'])

    return df_long

