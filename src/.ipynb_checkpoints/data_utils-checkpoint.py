import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def carregar_csv(caminho, delimitador):
    return pd.read_csv(caminho, delimiter=delimitador)


def separar_variaveis(df: pd.DataFrame):
    """
    Retorna duas listas:
    - colunas_continuas: variáveis numéricas contínuas
    - colunas_binarias: variáveis binárias (apenas 2 valores distintos)
    """
    colunas_continuas = []
    colunas_binarias = []

    for col in df.columns:
        # Conta valores únicos
        n_unique = df[col].nunique()

        # Se for numérica e tiver muitos valores distintos → contínua
        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 2:
            colunas_continuas.append(col)

        # Se tiver exatamente 2 valores distintos → binária
        elif n_unique == 2:
            colunas_binarias.append(col)

    return colunas_continuas, colunas_binarias

def comparar_media_mediana(df):
    df_num = df.select_dtypes(include='number')
    resultado = pd.DataFrame({
        'média':     df_num.mean(),
        'mediana':   df_num.median(),
        'diferença': df_num.mean() - df_num.median()
    })
    return resultado

def comparar_75_max(df):
    df_num = df.select_dtypes(include='number')
    resultado = pd.DataFrame({
        '75%':       df_num.quantile(0.75),
        'max':       df_num.max(),
        'diferença': df_num.max() - df_num.quantile(0.75)
    })
    return resultado

def correlacao_com_target(df, target, threshold=0.0):
    """
    Retorna as correlações de cada variável numérica com o target,
    ordenadas da mais forte para a mais fraca.

    Parâmetros:
        df        : DataFrame
        target    : str   → nome da coluna target
        threshold : float → valor mínimo de correlação (em módulo) para exibir
                            default 0.0 retorna todas

    Retorno:
        DataFrame com as colunas 'variável' e 'correlação'
    """
    correlacoes = (
        df.select_dtypes(include=['number','bool'])
        .corr()[target]
        .drop(target)
        .reset_index()
        .rename(columns={'index': 'variável', target: 'correlação'})
    )

    resultado = (
        correlacoes[correlacoes['correlação'].abs() >= threshold]
        .sort_values('correlação', ascending=False)
        .reset_index(drop=True)
    )

    return resultado

def detectar_outliers_iqr(df, coluna, fator=1.5):
    """
    Retorna os outliers (inferiores e superiores) usando IQR.

    Parâmetros:
    - df: DataFrame
    - coluna: nome da coluna (str)
    - fator: multiplicador do IQR (default: 1.5)

    Retorno:
    - DataFrame contendo apenas os outliers
    """

    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - fator * IQR
    limite_superior = Q3 + fator * IQR

    outliers = df[
        (df[coluna] < limite_inferior) |
        (df[coluna] > limite_superior)
    ]

    return outliers

def maiores_correlacoes(df, threshold=0.7):
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    
    correlacoes = (
        correlation_matrix
        .unstack()
        .reset_index()
        .rename(columns={'level_0': 'variável_1', 'level_1': 'variável_2', 0: 'correlação'})
    )
    
    # remove autocorrelação e duplicatas
    correlacoes = correlacoes[correlacoes['variável_1'] < correlacoes['variável_2']]
    
    # filtra pelo threshold e ordena
    resultado = (
        correlacoes[correlacoes['correlação'].abs() >= threshold]
        .sort_values('correlação', ascending=False)
        .reset_index(drop=True)
    )
    
    return resultado


def listar_colunas(df: pd.DataFrame):
    """
    Retorna todas as colunas de um DataFrame como lista.
    """
    return df.columns.tolist()

def separar_x_y(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    return X, y


def separar_treino_teste(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def balancear_dados(X_train, y_train, random_state=42):
    print("Distribuição antes do balanceamento:")
    print(y_train.value_counts())
    
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print("\nDistribuição após o balanceamento:")
    print(y_balanced.value_counts())
    
    return X_balanced, y_balanced

def padronizar_colunas(df, colunas):
    """
    Padroniza colunas numéricas de um DataFrame usando StandardScaler.
    
    Parâmetros:
        df      : DataFrame original
        colunas : list → colunas a padronizar
    
    Retorna:
        df_padronizado : DataFrame com as colunas padronizadas
        scaler         : objeto StandardScaler treinado
    """
    scaler = StandardScaler()
    df_padronizado = df.copy()
    df_padronizado[colunas] = scaler.fit_transform(df[colunas])
    
    return df_padronizado, scaler

def impute_by_median(df, col):
    """Imputa valores ausentes pela mediana global da coluna."""
    df_result = df.copy()
    df_result[col] = df_result[col].fillna(df_result[col].median())
    return df_result


def impute_by_group_median(df, col, group_col):
    """Imputa valores ausentes pela mediana de um único agrupamento."""
    df_result = df.copy()
    df_result[col] = df_result[col].fillna(
        df_result.groupby(group_col)[col].transform('median')
    )
    return df_result


def impute_by_multigroup_median(df, col, group_cols):
    """Imputa valores ausentes pela mediana de múltiplos agrupamentos."""
    df_result = df.copy()
    df_result[col] = df_result[col].fillna(
        df_result.groupby(group_cols)[col].transform('median')
    )
    return df_result

def criar_faixas_quantil(
    df: pd.DataFrame,
    coluna: str,
    nova_coluna: str,
    n_faixas: int = 3,
    labels: list = None,
    retornar_cortes: bool = False
) -> pd.DataFrame:
    """
    Cria uma coluna categórica dividindo uma variável numérica em faixas
    de igual frequência (quantis), usando os próprios valores do dataset
    para definir os pontos de corte.
 
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    coluna : str
        Nome da coluna numérica a ser fatiada.
    nova_coluna : str
        Nome da nova coluna categórica a ser criada.
    n_faixas : int, default 3
        Quantidade de faixas (ex.: 3 → baixo / médio / alto).
    labels : list, default None
        Rótulos para as faixas. Se None, usa 'Faixa 1', 'Faixa 2', ...
        Deve ter exatamente n_faixas elementos.
    retornar_cortes : bool, default False
        Se True, retorna também os pontos de corte calculados.
 
    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna adicionada.
    list (opcional)
        Lista com os pontos de corte, se retornar_cortes=True.
 
    Exemplos
    --------
    >>> df = criar_faixas_quantil(df, 'Annual_Income', 'Faixa_Renda',
    ...                           labels=['Baixa', 'Média', 'Alta'])
    >>> df = criar_faixas_quantil(df, 'Spending_Score', 'Faixa_Score', n_faixas=4)
    """
    if labels is None:
        labels = [f'Faixa {i+1}' for i in range(n_faixas)]
 
    if len(labels) != n_faixas:
        raise ValueError(
            f"'labels' deve ter {n_faixas} elementos, mas foram fornecidos {len(labels)}."
        )
 
    percentis = np.linspace(0, 100, n_faixas + 1)
    cortes = np.percentile(df[coluna].dropna(), percentis)
    cortes[0] -= 1e-9
    cortes[-1] += 1e-9
 
    df[nova_coluna] = pd.cut(
        df[coluna],
        bins=cortes,
        labels=labels,
        include_lowest=True
    )
 
    limites = [f"{labels[i]}: [{cortes[i]:.2f}, {cortes[i+1]:.2f}]"
               for i in range(n_faixas)]
    print(f"[criar_faixas_quantil] '{coluna}' → '{nova_coluna}'")
    for l in limites:
        print(f"  {l}")
 
    if retornar_cortes:
        return df, cortes.tolist()
    return df

def criar_faixas_manuais(
    df: pd.DataFrame,
    coluna: str,
    nova_coluna: str,
    cortes: list,
    labels: list
) -> pd.DataFrame:
    """
    Cria uma coluna categórica com cortes manuais definidos pelo usuário.
    Útil quando os intervalos têm significado de negócio específico
    (ex.: faixas etárias padronizadas).
 
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    coluna : str
        Nome da coluna numérica a ser fatiada.
    nova_coluna : str
        Nome da nova coluna categórica a ser criada.
    cortes : list
        Bordas dos intervalos, incluindo mínimo e máximo.
        Ex.: [0, 25, 35, 45, 100] gera 4 faixas.
    labels : list
        Rótulos para cada intervalo. Deve ter len(cortes) - 1 elementos.
 
    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna adicionada.
 
    Exemplos
    --------
    >>> df = criar_faixas_manuais(
    ...     df, 'Age', 'Faixa_Etaria',
    ...     cortes=[0, 25, 35, 45, 60, 100],
    ...     labels=['18–25', '26–35', '36–45', '46–60', '60+']
    ... )
    """
    if len(labels) != len(cortes) - 1:
        raise ValueError(
            f"'labels' deve ter {len(cortes)-1} elementos para {len(cortes)} cortes."
        )
 
    df[nova_coluna] = pd.cut(
        df[coluna],
        bins=cortes,
        labels=labels,
        include_lowest=True,
        right=True
    )
 
    print(f"[criar_faixas_manuais] '{coluna}' → '{nova_coluna}'")
    for i, label in enumerate(labels):
        print(f"  {label}: ({cortes[i]}, {cortes[i+1]}]")
 
    return df
 
 
def criar_coluna_binaria(
    df: pd.DataFrame,
    coluna: str,
    nova_coluna: str,
    mapa: dict
) -> pd.DataFrame:
    """
    Converte uma coluna binária (0/1) em categorias legíveis,
    usando um dicionário de mapeamento fornecido pelo usuário.
 
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    coluna : str
        Nome da coluna binária existente.
    nova_coluna : str
        Nome da nova coluna categórica.
    mapa : dict
        Dicionário mapeando os valores existentes para os rótulos desejados.
        Ex.: {1: 'Masculino', 0: 'Feminino'}
 
    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna adicionada.
 
    Exemplos
    --------
    >>> df = criar_coluna_binaria(df, 'Male', 'Genero',
    ...                           mapa={1: 'Masculino', 0: 'Feminino'})
    """
    valores_unicos = set(df[coluna].dropna().unique())
    chaves_mapa = set(mapa.keys())
 
    if not valores_unicos.issubset(chaves_mapa):
        faltando = valores_unicos - chaves_mapa
        raise ValueError(
            f"Valores {faltando} presentes em '{coluna}' não têm mapeamento em 'mapa'."
        )
 
    df[nova_coluna] = df[coluna].map(mapa)
 
    print(f"[criar_coluna_binaria] '{coluna}' → '{nova_coluna}'")
    for k, v in mapa.items():
        contagem = (df[coluna] == k).sum()
        print(f"  {k} → {v}: {contagem} registros")
 
    return df

def resumo_clusters(df, colunas, cluster_col="cluster"):
    """
    Retorna a moda (categoria mais comum) por cluster.
    """

    resumo = {}

    for col in colunas:
        resumo[col] = df.groupby(cluster_col)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    return pd.DataFrame(resumo)


import pandas as pd

def filtrar_por_coluna_booleana(df, coluna_flag, coluna_valor=None):
    """
    Filtra um DataFrame com base em uma coluna booleana no formato 't'/'f'.
    
    - 'f' (false) → registro é EXIBIDO
    - 't' (true)  → registro é IGNORADO
    
    Parameters:
    -----------
    df           : pd.DataFrame — o dataset
    coluna_flag  : str — nome da coluna com 't' ou 'f'
    coluna_valor : str ou None — coluna dependente a exibir junto (opcional)
    
    Returns:
    --------
    pd.DataFrame filtrado
    """

    # Validações básicas
    if coluna_flag not in df.columns:
        raise ValueError(f"Coluna '{coluna_flag}' não encontrada no DataFrame.")
    
    if coluna_valor and coluna_valor not in df.columns:
        raise ValueError(f"Coluna '{coluna_valor}' não encontrada no DataFrame.")

    # Normaliza para minúsculo para evitar erros com 'T', 'F', ' t', etc.
    flag_normalizado = df[coluna_flag].astype(str).str.strip().str.lower()

    # Verifica valores inesperados
    valores_validos = {'t', 'f'}
    valores_encontrados = set(flag_normalizado.unique())
    valores_invalidos = valores_encontrados - valores_validos
    if valores_invalidos:
        print(f"⚠️  Atenção: valores inesperados encontrados em '{coluna_flag}': {valores_invalidos}")

    # Filtra apenas onde o flag é 'f' (false → exibe)
    mascara = flag_normalizado == 'f'
    df_filtrado = df[mascara].copy()

    # Se uma coluna de valor dependente foi passada, seleciona as colunas relevantes
    if coluna_valor:
        colunas_exibir = [coluna_flag, coluna_valor]
        df_filtrado = df_filtrado[colunas_exibir]

    print(f"✅ Total de registros: {len(df)}")
    print(f"   → Exibidos (flag='f'): {mascara.sum()}")
    print(f"   → Ignorados (flag='t'): {(~mascara).sum()}")

    return df_filtrado


def atualizar_valor_por_flag(df, coluna_flag, coluna_valor):
    """
    Onde coluna_flag == 'f', atualiza coluna_valor para '0'.

    Parameters:
    -----------
    df           : pd.DataFrame — o dataset
    coluna_flag  : str — coluna com valores 't' ou 'f'
    coluna_valor : str — coluna a ser atualizada

    Returns:
    --------
    pd.DataFrame atualizado
    """

    if coluna_flag not in df.columns:
        raise ValueError(f"Coluna '{coluna_flag}' não encontrada no DataFrame.")
    if coluna_valor not in df.columns:
        raise ValueError(f"Coluna '{coluna_valor}' não encontrada no DataFrame.")

    df = df.copy()

    # Máscara para registros com 'f'
    mascara = df[coluna_flag].astype(str).str.strip().str.lower() == 'f'

    # Exibe o antes
    print(f"✅ Total de registros         : {len(df)}")
    print(f"   → Com flag 'f' (alterar)  : {mascara.sum()}")
    print(f"   → Com flag 't' (manter)   : {(~mascara).sum()}")
    print(f"\n🔍 Antes da alteração:")
    print(df.loc[mascara, [coluna_flag, coluna_valor]])

    # Aplica a atualização
    df.loc[mascara, coluna_valor] = '0'

    # Exibe o depois
    print(f"\n✅ Depois da alteração:")
    print(df.loc[mascara, [coluna_flag, coluna_valor]])

    return df


def padronizar_colunas(df):
    """
    Padroniza o nome das colunas do DataFrame:
    - Substitui espaços por '_'
    - Converte para minúsculo
    - Remove espaços extras nas bordas

    Parameters:
    -----------
    df : pd.DataFrame — o dataset

    Returns:
    --------
    pd.DataFrame com colunas padronizadas
    """

    df = df.copy()

    colunas_antigas = df.columns.tolist()

    df.columns = (
        df.columns
        .str.strip()       # Remove espaços nas bordas
        .str.lower()       # Converte para minúsculo
        .str.replace(' ', '_', regex=False)  # Substitui espaços por '_'
    )

    colunas_novas = df.columns.tolist()

    # Exibe as alterações realizadas
    print("✅ Colunas padronizadas:\n")
    print(f"{'ANTES':<30} {'DEPOIS':<30}")
    print("-" * 60)
    for antiga, nova in zip(colunas_antigas, colunas_novas):
        alterado = "⚠️ " if antiga != nova else "   "
        print(f"{alterado}{antiga:<27} → {nova:<30}")

    return df
