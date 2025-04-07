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
