import pandas as pd

def estacao_do_ano(data):
    ano = data.year
    datas_estacoes = {
        'primavera': (pd.Timestamp(f'{ano}-03-20'), pd.Timestamp(f'{ano}-06-20')),
        'verao': (pd.Timestamp(f'{ano}-06-21'), pd.Timestamp(f'{ano}-09-22')),
        'outono': (pd.Timestamp(f'{ano}-09-23'), pd.Timestamp(f'{ano}-12-20')),
        'inverno': (
            (pd.Timestamp(f'{ano}-12-21'), pd.Timestamp(f'{ano+1}-03-19'))
            if data >= pd.Timestamp(f'{ano}-12-21') else
            (pd.Timestamp(f'{ano-1}-12-21'), pd.Timestamp(f'{ano}-03-19'))
        )
    }
    
    for estacao, (inicio, fim) in datas_estacoes.items():
        if inicio <= data <= fim:
            return estacao
    return 'inverno'