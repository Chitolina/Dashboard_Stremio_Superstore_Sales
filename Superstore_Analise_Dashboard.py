import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(page_title="Dashboard Superstore" ) # layout="wide"
sns.set_style("whitegrid")

# Função para carregar dados com cache
@st.cache_data
def load_data():
    # URL do arquivo
    url = 'https://raw.githubusercontent.com/Chitolina/Dashboard_Stremio_Superstore_Sales/main/df/Sample_Superstore.csv'
    df = pd.read_csv(url, delimiter=',', encoding='latin1')
    
    # Função para corrigir datas
    def corrigir_data(data_str):
        for fmt in ['%d/%m/%Y', '%m/%d/%Y']:
            try: return pd.to_datetime(data_str, format=fmt)
            except ValueError: continue
        return pd.NaT
    
    # Tratamento de datas e criação de novas colunas
    df['Order Date'] = df['Order Date'].apply(corrigir_data)
    df = df.dropna(subset=['Order Date'])
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    df['Profit Margin'] = df['Profit'] / df['Sales']  # Nova coluna
    
    # Traduzindo valores das colunas 'Region', 'Category' e 'Sub-Category'
    region_translation = {
        'Northeast': 'Nordeste',
        'South': 'Sul',
        'Midwest': 'Centro-Oeste',
        'West': 'Oeste',
        'Central': 'Centro',
        'East': 'Leste'
    }

    category_translation = {
        'Furniture': 'Móveis',
        'Office Supplies': 'Suprimentos de Escritório',
        'Technology': 'Tecnologia'
    }

    subcategory_translation = {
        'Bookcases': 'Estantes',
        'Chairs': 'Cadeiras',
        'Labels': 'Etiquetas',
        'Tables': 'Mesas',
        'Storage': 'Armazenamento',
        'Furnishings': 'Mobiliário',
        'Art': 'Arte',
        'Phones': 'Telefones',
        'Binders': 'Pastas',
        'Appliances': 'Eletrodomésticos',
        'Paper': 'Papel',
        'Accessories': 'Acessórios',
        'Envelopes': 'Envelopes',
        'Fasteners': 'Fixadores',
        'Supplies': 'Suprimentos',
        'Machines': 'Máquinas',
        'Copiers': 'Copiadoras'
    }

    # Traduzindo os nomes das colunas
    df = df.rename(columns={
        'Region': 'Região',
        'Category': 'Categoria',
        'Sub-Category': 'Subcategoria',
        'Order ID': 'ID da Ordem',
        'Order Date': 'Data da Ordem',
        'Ship Date': 'Data de Envio',
        'Ship Mode': 'Modo de Envio',
        'Customer ID': 'ID do Cliente',
        'Customer Name': 'Nome do Cliente',
        'Segment': 'Segmento',
        'Country': 'País',
        'City': 'Cidade',
        'State': 'Estado',
        'Postal Code': 'Código Postal',
        'Product ID': 'ID do Produto',
        'Product Name': 'Nome do Produto',
        'Sales': 'Vendas',
        'Quantity': 'Quantidade',
        'Discount': 'Desconto',
        'Profit': 'Lucro',
        'Profit Margin': 'Margem de Lucro',
        'Month': 'Mês',
        'Year': 'Ano'
    })
    
    # Aplicando a tradução dos valores nas colunas 'Região', 'Categoria', 'Subcategoria'
    if 'Região' in df.columns:
        df['Região'] = df['Região'].map(region_translation).fillna(df['Região'])
    
    if 'Categoria' in df.columns:
        df['Categoria'] = df['Categoria'].map(category_translation).fillna(df['Categoria'])
    
    if 'Subcategoria' in df.columns:
        df['Subcategoria'] = df['Subcategoria'].map(subcategory_translation).fillna(df['Subcategoria'])

    return df

df = load_data()



# Título principal
st.title("Dashboard Analítico - Varejo")

tab0,tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Introdução ao Dataset","Visão Geral e Heatmap", "Rentabilidade", "Produtos", 
    "Impacto de Descontos", "Segmentação de Clientes", "Análise Geográfica"
])

# stilização em css
st.markdown("""
    <style>
        /* Garante que o conteúdo não seja cortado */
        .block-container {
            min-height: 100vh !important;
            height: auto !important;
            padding-top: 20px !important;
            padding-bottom: 20px !important;
            overflow-y: visible !important;
        }

        /* Corrige a altura máxima para permitir rolagem */
        html, body, [class*="css"] {
            height: auto !important;
            min-height: 100vh !important;
            overflow: visible !important;
        }

        /* Ajusta o posicionamento das abas */
        .stTabs [role="tablist"] {
            flex-wrap: wrap;
            justify-content: center;
            border-bottom: 2px solid #ccc;
            padding-bottom: 5px;
        }
        
        /* Estiliza as abas */
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            color: #333;
            border-radius: 10px 10px 0 0;
            background-color: #f5f5f5;
            margin: 5px;
            transition: all 0.3s ease-in-out;
        }

        /* Aba ativa */
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50 !important;
            color: white !important;
            border-bottom: 3px solid #388E3C;
        }

        /* Corrige a posição do título para evitar que suma */
        h1, h2, h3, h4, h5, h6 {
            margin-top: 10px !important;
        }

    </style>
""", unsafe_allow_html=True)


# Introdução ao DataFrame
with tab0:
    st.markdown("""
    ## Introdução ao Dataset  
    
    Este dashboard analisa dados de um varejo fictício (Superstore), incluindo vendas, lucro, descontos e categorias de produtos. Os dados são estruturados com base em pedidos realizados, contendo informações detalhadas sobre transações, clientes e regiões.  
    
    ## Foco nas colunas utilizadas na análise e métricas:  
    -  **Vendas**: em valor;
    -  **Lucro**: Resultado financeiro líquido;
    -  **Margem de Lucro**: Razão entre lucro e vendas;  
    -  **Descontos**: Impacto nas estratégias de preço;  
    -  **Categorias e Subcat.**: Segmentação dos produtos.  
    
    
    ## Valor de Negócio  
    
    Este dashboard fornece insights estratégicos para otimização das operações de varejo, tais como:  
    - **Identificação de padrões de compra** para melhorar a segmentação de clientes;  
    - **Análise da relação entre descontos e lucro**;  
    - **Entendimento do desempenho de produtos (nível cat e subcat.**);  
    - **Avaliação geográfica das vendas**.  
    
    """)

# Abas ===========================================================================



st.markdown(
    """
    <style>
        .big-font { font-size:20px !important; font-weight: bold; }
        .stMarkdown { margin-bottom: 10px; }
        .css-18e3th9 { padding-top: 1rem; } /* Ajuste no espaço superior */
    </style>
    """,
    unsafe_allow_html=True
)

# Primeira aba ===========================================================================
with tab1:
    st.markdown("## Mapa de Correlações")
    
    # Criando colunas para dividir a tela
    col1, col2 = st.columns([1, 1.5])  # Ajuste as proporções conforme necessário
    
    with col1:
        st.markdown("""
        ---
        ### Por que começar pela análise de correlações?
        - Entender a intensidade de influência entre variáveis e identificar padrões;  
        - Nortear para quais variáveis começar a destacar. 
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Criando o heatmap com valores maiores e mais escuros
        sns.heatmap(
            df[['Vendas', 'Lucro', 'Desconto', 'Quantidade']].corr(),
            annot=True, cmap='viridis', fmt='.2f',
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 12, "color": "black"}  # Tamanho maior e cor preta
        )
    
        # Ajustando tamanho dos ticks dos eixos
        ax.tick_params(axis='both', which='major', labelsize=12)
    
        plt.title('Relação entre Variáveis-Chave', fontsize=16)
        st.pyplot(fig)

    # Criando uma nova linha de layout
    st.markdown("---")
    st.markdown("##  Interpretação da Correlação de Pearson")

    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.info("###  Insight Crítico: Descontos vs Lucro")  
        st.markdown("""
        - **Correlação Negativa Moderada (-0.22)**  
        - De forma geral, sabemos que **descontos reduzem a margem de lucro**;  
        - Porém, se o desconto aumentar significativamente o volume de vendas, o lucro total pode **ainda crescer**, desde que a margem por unidade não fique muito baixa;  
        - O problema ocorre quando o desconto é tão alto que o aumento nas vendas não compensa a perda na margem.  
         """, unsafe_allow_html=True)

    with col4:
        st.success("###  Oportunidade: Vendas vs Lucro")  
        st.markdown("""
        - **Correlação Positiva Moderada (0.48)**  
        - Mais vendas tendem a aumentar o lucro;  
        - Produtos com margens mais altas podem gerar maior rentabilidade.  
        """, unsafe_allow_html=True)




# Segunda aba ===========================================================================
with tab2:
    st.markdown("## Análise de Rentabilidade")
    
    # Gráfico 1 - Scatterplot
    st.markdown("""
    ### Relação Vendas vs Lucro
    O gráfico de dispersão mostra a relação entre vendas e lucro para diferentes categorias de produtos. A linha tracejada mostra a tendência (regressão linear) geral esperada entre vendas e lucro.
    - A inclinação **positiva** significa que à medida que as vendas aumentam, o lucro também tende a aumentar.
    - O grau de dispersão dos pontos ao redor da linha mostra o quanto os dados seguem essa tendência. 
        - Pontos espalhados, correlação fraca, pontos próximos à linha, correlação forte.
        - Outra questão, quando as vendas são baixas, os **pontos estão concentrados**, indicando um **padrão estável de lucro**. 
         Assim que as vendas aumentam, os pontos se tornam mais **dispersos**, sugerindo **maior variabilidade no lucro**. 
         Isso tem influência de descontos, variação da margem (podendo ser negativa por prejuízo, ou um outlier não tratado) e diferentes categorias de produtos (tecnologia é a que mais sofre com esse impacto). Esse padrão indica que vendas maiores podem ser mais imprevisíveis. 
    """)
    
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    sns.scatterplot(data=df, x='Vendas', y='Lucro', hue='Categoria', 
                    palette='plasma', alpha=0.7, s=200, edgecolor='black', linewidth=1.5)
    sns.regplot(data=df, x='Vendas', y='Lucro', scatter=False, 
                color='red', line_kws={'linewidth':2, 'linestyle':'--'})
    
    formatter = mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}k')
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    
    # Ajustando tamanho dos rótulos dos eixos
    ax1.tick_params(axis='both', labelsize=14)  
    
    # Ajustando tamanho dos títulos
    # plt.title('Performance por Transação', fontsize=18, fontweight='bold')
    plt.xlabel('Vendas (Milhares)', fontsize=14, fontweight='bold')
    plt.ylabel('Lucro (Milhares)', fontsize=14, fontweight='bold')
    
    st.pyplot(fig1)
    st.markdown("---")

    # Gráfico 2 - Margem de Lucro
    st.markdown("""
    ### Margem de lucro por categoria
    Comparação da média de margem por cada categoria. Podemos perceber que tecnologia e escritório apresentam valores similares, enquanto móveis uma margem bem mais baixa.
    Margem de lucro seria suficiente para descartarmos um grupo, ou também precisamos ver o lucro absoluto?!
    """)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    barplot = sns.barplot(data=df, x='Categoria', y='Margem de Lucro', 
                          palette='plasma', alpha=0.7, estimator=np.mean)
    
    # Adiciona os valores médios dentro de cada barra
    for p in barplot.patches:
        height = p.get_height()
        ax2.annotate(f'{height:.2%}',  # Formata para porcentagem
                     (p.get_x() + p.get_width() / 2., height / 2),  # Posiciona no meio da barra
                     ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # plt.title('Margem de Lucro por Categoria', fontsize=16)
    plt.xlabel('Categoria', fontsize=12)
    plt.ylabel('Margem Média', fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
# Rodapé ===========================================================================

st.markdown("---")

st.write("**Fonte dos Dados:** [Kaggle - Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)")

st.write("**Desenvolvido por:** [Lucas Chitolina¹](https://github.com/Chitolina), [Lucas Chitolina²](https://chitolina.github.io/), & [DeepSeek](https://chat.deepseek.com/)")

# Terceira aba ===========================================================================
with tab3:
    st.markdown("## Análise Detalhada por Produto")
    
    # Seção 1 - Comparação de Vendas vs. Lucro por Categoria
    st.markdown("""
    ### Venda absoluta e lucro absoluto por categoria
    Ao trazer os valores absolutos depois de termos visto a margem de lucro, conseguimos visualizar qual categoria gera mais lucro em termos absolutos.
    Nesse caso a margem e lucro seguiram condizentes com as categorias, mas poderia haver o caso de uma categoria ter margem maior, porém não vender tanto e, assim,
    apresentar menos lucro que outra.
    """)
    
    # Calcular Vendas e Lucro por Categoria
    Vendas_Lucro_by_category = df.groupby('Categoria')[['Vendas', 'Lucro']].sum().reset_index()
    
    # Criar gráfico de barras lado a lado
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    categories = Vendas_Lucro_by_category['Categoria']
    x = np.arange(len(categories))  # Posições das barras
    
    # Definir cores mais suaves e harmoniosas
    colors = ['#4C72B0', '#55A868']  # Azul e verde
    
    # Criar barras lado a lado
    bars1 = ax1.bar(x - bar_width/2, Vendas_Lucro_by_category['Vendas'], width=bar_width, 
                    label='Vendas', color=colors[0], alpha=0.85)
    bars2 = ax1.bar(x + bar_width/2, Vendas_Lucro_by_category['Lucro'], width=bar_width, 
                    label='Lucro', color=colors[1], alpha=0.85)
    
    # Adicionar rótulos com melhor posicionamento
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000, 
                 f"{bar.get_height()/1000:.1f}k", ha='center', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000, 
                 f"{bar.get_height()/1000:.1f}k", ha='center', fontsize=11, fontweight='bold')
    
    # Formatação do eixo y
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    
    # Melhor organização dos eixos
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=30)
    
    # Ajustes finais
    # plt.title('Vendas totais e Lucro Absoluto por Categoria', fontsize=16)
    plt.ylabel('Valor em $', fontsize=12)
    plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legenda fora do gráfico
    sns.despine()
    st.pyplot(fig1)

    st.markdown("---")

    # Seção 2 - Comparação de Vendas e Lucros por Categoria
    st.markdown("""
    ###  Top 8 Subcategorias
    #### Maiores vendas e comparação com seus lucros:
    - Nem sempre os produtos que mais vendem são os mais lucrativos. Fatores como margem de lucro, descontos, promoções e custos operacionais podem impactar na rentabilidade.
    """)
    
    # Calcular vendas e lucros por subcategoria
    top8_Vendas = df.groupby(['Categoria', 'Subcategoria'])[['Vendas', 'Lucro']].sum().nlargest(8, 'Vendas').reset_index()
    
    # Criar rótulo combinando Subcategoria + Categoria
    top8_Vendas['Label'] = top8_Vendas['Subcategoria'] + " (" + top8_Vendas['Categoria'] + ")"
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(14, 8))  
    bar_width = 0.3  
    y_pos = np.arange(len(top8_Vendas))
    
    # Criar barras de vendas e lucros
    ax.barh(y_pos - bar_width/2, top8_Vendas['Vendas'], height=bar_width, label='Vendas', color='#4C72B0', alpha=0.85)
    ax.barh(y_pos + bar_width/2, top8_Vendas['Lucro'], height=bar_width, label='Lucro', color='#55A868', alpha=0.85)
    
    # Adicionar rótulos nas barras
    for bar in ax.patches:
        ax.text(bar.get_width() + 3000, bar.get_y() + bar.get_height()/2, f"${bar.get_width()/1000:.1f}k", va='center', fontsize=14)
    
    # Ajustar visualização
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top8_Vendas['Label'], fontsize=14)
    ax.set_xlabel('Valor em $', fontsize=16)
    # ax.set_title('Comparação de Vendas e Lucro - Top 8 Subcategorias', fontsize=18, pad=20)
    
    # Colocar a legenda fora do gráfico (à direita)
    ax.legend(title="Métrica", fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    
    # Exibir gráfico no Streamlit
    st.pyplot(fig)

        
    st.markdown("---")

    # Seção 3 - Produtos Problemáticos
    st.markdown("""
    ### Identificação de produtos pouco performáticos: 
    Podemos ver muitos produtos com alto volume de vendas, mas baixa lucratividade, alguns lucrando menos que 10% da venda (colocar gráfico em wide mode).
    """)
    
    # Calcular dados
    top_Vendas = df.groupby('Subcategoria').agg({'Vendas':'sum', 'Lucro':'sum'}).reset_index()
    
    # Calcular a porcentagem de lucro sobre as vendas
    top_Vendas['Margem de Lucro %'] = (top_Vendas['Lucro'] / top_Vendas['Vendas']) * 100
    
    # Plotar gráfico
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    scatter = sns.scatterplot(data=top_Vendas, x='Vendas', y='Lucro', 
                             hue='Subcategoria', size='Vendas', sizes=(50, 500),
                             palette='Paired', alpha=0.7, edgecolor='black')
    
    # Adicionar as porcentagens de lucro acima das bolinhas com tamanho menor e mais acima
    for i, row in top_Vendas.iterrows():
        ax3.text(row['Vendas'], row['Lucro'] + 4000,
                 f"{row['Margem de Lucro %']:.1f}%", 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=8, fontweight='bold', alpha=0.6, color='black') 
    
    # Linha de referência e formatação
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    
    # plt.title('Faturamento vs Lucratividade', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig3)
    
    # Nota técnica
    st.markdown("""
    #### Observações sobre lucros negativos:
    - Descontos agressivos aplicados;
    - Custos operacionais elevados;
    - Problemas na precificação;
    - Promoções mal planejadas.
    """)


# Quarta aba ===========================================================================
with tab4:
    st.markdown("""
                    #### Lucro médio e mediano por faixa de desconto
                    - Identificação do nível de desconto onde o lucro médio e mediano tornam-se negativos:
                    """)
    
    # Processar dados
    df['Faixa_Desconto'] = pd.cut(df['Desconto'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1], 
                                 labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
    
    # Calcular média e mediana por faixa de desconto
    discount_analysis = df.groupby('Faixa_Desconto')['Lucro'].agg(['mean', 'median']).reset_index()
    
    # Criar o gráfico
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Gráfico de média
    sns.lineplot(data=discount_analysis, x='Faixa_Desconto', y='mean', 
                 marker='o', markersize=10, color='#FF4500', linewidth=2.5, label='Média')
    
    # Gráfico de mediana
    sns.lineplot(data=discount_analysis, x='Faixa_Desconto', y='median', 
                 marker='o', markersize=10, color='#1f77b4', linewidth=2.5, label='Mediana')
    
    # Linha de referência destacada
    plt.axhline(0, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.7)
    
    # Ajustes visuais
    plt.xlabel('Faixa de Desconto', fontsize=12, weight='bold')
    plt.ylabel('Lucro (R$)', fontsize=12, weight='bold')
    plt.xticks(fontsize=12, color='#444')
    plt.yticks(fontsize=12, color='#444')
    plt.legend(title="Indicador", fontsize=12, title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Exibir no Streamlit
    st.pyplot(fig2)
    
    # Explicação do Primeiro Gráfico
    st.markdown("""
    #### Principais Insights:
    1. **Tendência Negativa**:  
       - Descontos acima de 20% mostram **maior variabilidade de prejuízos**, como vemos pela linha decrescendo;
    2. **Faixa Ideal**:  
       - Descontos entre 0-10% mantêm a maior concentração de resultados positivos, até 20% ainda se mostra vantajoso;
    3. **Outliers**:  
        - Embora a linha de lucro médio caia na faixa de 40-50%, há uma inversão inesperada em 50%+. 
        Isso pode ser resultado do tamanho reduzido da amostra nessa faixa, tornando-a mais sensível a outliers que distorcem a média. 
        Porém, mesmo ao usar a mediana esse comportamento persiste, embora de forma menos acentuada. Será que essa inversão é apenas efeito de outliers, ou há influência de outros fatores?. 
    """)
    
    st.markdown("---")
    
    # Contar a quantidade de transações por faixa
    discount_counts = df['Faixa_Desconto'].value_counts().reset_index()
    discount_counts.columns = ['Faixa de Desconto', 'Número de Transações']
    
    # Ordenar as faixas de desconto
    order = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
    discount_counts['Faixa de Desconto'] = pd.Categorical(discount_counts['Faixa de Desconto'], categories=order, ordered=True)
    discount_counts = discount_counts.sort_values('Faixa de Desconto')
    
    # Transformar as faixas de desconto em colunas
    discount_pivot = discount_counts.set_index('Faixa de Desconto').T
    
    # Exibir tabela no Streamlit
    st.markdown("""
                #### Distribuição das Transações por Faixa de Desconto
                - Ao trazermos as amostras de transações por cada faixa, percebemos que elas não têm tamanhos similares.
    """)
    st.table(discount_pivot)
    
    st.markdown("---")
    
    st.markdown("""
    #### Removendo outliers, como se comporta a linha?
    - Mesmo sem outliers a reta mantém o padrão pela influência de outros possíveis fatores:
    
        - Distribuição assimétrica: A tendência de queda e recuperação já está presente nos dados;
        - Amostras pequenas: Faixas com poucas transações ainda sofrem variações naturais;
        - Efeito real do desconto: Pode ser que em alguns casos descontos altos podem **impulsionar o lucro via volume de vendas**;
        - Outliers não eram o problema: Se não distorciam tanto a média, sua remoção não impactava muito o gráfico.
    """)
    
    # Remover outliers (por exemplo, valores acima de 3 desvios padrão)
    cleaned_data = df[(df['Lucro'] < df['Lucro'].mean() + 3 * df['Lucro'].std()) &
                      (df['Lucro'] > df['Lucro'].mean() - 3 * df['Lucro'].std())]
    
    # Processar dados
    cleaned_data['Faixa_Desconto'] = pd.cut(cleaned_data['Desconto'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1], 
                                           labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
    
    # Calcular média e mediana por faixa de desconto
    discount_analysis = cleaned_data.groupby('Faixa_Desconto')['Lucro'].agg(['mean', 'median']).reset_index()
    
    # Criar o gráfico
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Gráfico de média
    sns.lineplot(data=discount_analysis, x='Faixa_Desconto', y='mean', 
                 marker='o', markersize=10, color='#FF4500', linewidth=2.5, label='Média')
    
    # Gráfico de mediana
    sns.lineplot(data=discount_analysis, x='Faixa_Desconto', y='median', 
                 marker='o', markersize=10, color='#1f77b4', linewidth=2.5, label='Mediana')
    
    # Linha de referência destacada
    plt.axhline(0, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.7)
    
    # Ajustes visuais
    plt.xlabel('Faixa de Desconto', fontsize=12, weight='bold')
    plt.ylabel('Lucro (R$)', fontsize=12, weight='bold')
    plt.xticks(fontsize=12, color='#444')
    plt.yticks(fontsize=12, color='#444')
    plt.legend(title="Indicador", fontsize=12, title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Exibir no Streamlit
    st.pyplot(fig2)
    
    st.markdown("---")

    # Boxplot - Identificação de Outliers e Dispersão
    st.markdown("#### Boxplot: Distribuição do Lucro por Faixa de Desconto")
    fig1, ax1 = plt.subplots(figsize=(12, 6))  # Ajustando o tamanho para ser igual aos outros gráficos
    
    # Customizando o boxplot
    sns.boxplot(x='Faixa_Desconto', y='Lucro', data=cleaned_data, ax=ax1, 
                palette='coolwarm', width=0.5, linewidth=1.5)
    
    # Customizando os rótulos do eixo X
    ax1.set_xlabel('Faixa de Desconto', fontsize=12, weight='bold')
    ax1.set_ylabel('Lucro (R$)', fontsize=12, weight='bold')
    
    # Ajustando os rótulos do eixo X para ficarem mais legíveis
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=12, color='#444')
    
    # Ajustando os rótulos do eixo Y para ficarem consistentes
    ax1.tick_params(axis='y', labelsize=12, labelcolor='#444')
    
    # Melhorando o estilo das linhas e adicionando uma linha de média
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    
    # Melhorando a aparência geral
    sns.despine(top=True, right=True)
    
    st.pyplot(fig1)




    
# Quinta aba ===========================================================================

with tab5:
    st.markdown("## Análise Estratégica de Clientes")
    
    # Seção 1 - Scatterplot de Rentabilidade
    st.markdown("""
        ### Identificação de Clientes Rentáveis vs. Problemáticos
        
        O scatterplot abaixo ilustra a relação entre o volume total de vendas e o lucro gerado por cada cliente. 
        Destaca-se os **3 clientes mais rentáveis** e os **3 menos rentáveis**, permitindo uma análise comparativa.
    """)
    
    # Processar dados (ajustado para novos nomes de colunas)
    cliente_lucro = df.groupby('Nome do Cliente').agg({'Vendas':'sum', 'Lucro':'sum'}).reset_index()
    top_3 = cliente_lucro.nlargest(3, 'Lucro')
    bottom_3 = cliente_lucro.nsmallest(3, 'Lucro')
    
    # Plotar gráfico
    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = sns.scatterplot(
        data=cliente_lucro, x='Vendas', y='Lucro', 
        hue='Lucro', size='Lucro', palette='rocket_r',
        sizes=(50, 500), alpha=0.7, edgecolor='black'
    )
    
    # Destacar top/bottom
    for i in range(3):
        ax.annotate(top_3['Nome do Cliente'].iloc[i], 
                   (top_3['Vendas'].iloc[i], top_3['Lucro'].iloc[i]),
                   textcoords="offset points", xytext=(0,15), 
                   ha='center', fontsize=12, weight='bold', color='darkgreen')
        
        ax.annotate(bottom_3['Nome do Cliente'].iloc[i], 
                   (bottom_3['Vendas'].iloc[i], bottom_3['Lucro'].iloc[i]),
                   textcoords="offset points", xytext=(0,-20), 
                   ha='center', fontsize=12, weight='bold', color='darkred')

    # Formatação profissional
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y/1000:.0f}K'))
    plt.title('Mapa de Rentabilidade por Cliente', fontsize=16)
    plt.xlabel('Gasto Total do Cliente', fontsize=12)
    plt.ylabel('Lucro Gerado', fontsize=12)
    st.pyplot(fig)

    # Seção 2 - Lista Estratégica
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ####  Clientes Premium (Top 3)
        Principais geradores de valor e % de consumo em cada categoria:
        """)
        
        # Adicionando a porcentagem de consumo por categoria (ajuste para novas colunas)
        categoria_lucro_cliente = df.groupby(['Nome do Cliente', 'Categoria']).agg({'Vendas':'sum'}).reset_index()
        categoria_lucro_cliente['Porcentagem'] = categoria_lucro_cliente.groupby('Nome do Cliente')['Vendas'].transform(lambda x: (x / x.sum()) * 100)
        top_3_categorias = categoria_lucro_cliente[categoria_lucro_cliente['Nome do Cliente'].isin(top_3['Nome do Cliente'])]
        
        # Mostrar a tabela com categorias e porcentagens
        top_3_display = top_3[['Nome do Cliente', 'Vendas', 'Lucro']].rename(columns={'Nome do Cliente':'Cliente', 'Vendas':'Gasto Total', 'Lucro':'Lucro'})
        
        # Adicionar as porcentagens por categoria à tabela
        for cliente in top_3['Nome do Cliente']:
            categorias = top_3_categorias[top_3_categorias['Nome do Cliente'] == cliente]
            categoria_percentages = ', '.join([f"{row['Categoria']}: {row['Porcentagem']:.1f}%" for _, row in categorias.iterrows()])
            top_3_display.loc[top_3_display['Cliente'] == cliente, 'Categorias (%)'] = categoria_percentages

        st.dataframe(
            top_3_display.style.format({'Gasto Total':'${:.0f}', 'Lucro':'${:.0f}'}), 
            hide_index=True, 
            use_container_width=True
        )
    
    with col2:
        st.markdown("""
        ####  Clientes de Risco (Bottom 3)
        Principais geradores de prejuízos e % de consumo em cada categoria:
        """)
        
        # Adicionando a porcentagem de consumo por categoria
        bottom_3_categorias = categoria_lucro_cliente[categoria_lucro_cliente['Nome do Cliente'].isin(bottom_3['Nome do Cliente'])]
        
        # Mostrar a tabela com categorias e porcentagens
        bottom_3_display = bottom_3[['Nome do Cliente', 'Vendas', 'Lucro']].rename(columns={'Nome do Cliente':'Cliente', 'Vendas':'Gasto Total', 'Lucro':'Lucro'})
        
        # Adicionar as porcentagens por categoria à tabela
        for cliente in bottom_3['Nome do Cliente']:
            categorias = bottom_3_categorias[bottom_3_categorias['Nome do Cliente'] == cliente]
            categoria_percentages = ', '.join([f"{row['Categoria']}: {row['Porcentagem']:.1f}%" for _, row in categorias.iterrows()])
            bottom_3_display.loc[bottom_3_display['Cliente'] == cliente, 'Categorias (%)'] = categoria_percentages

        st.dataframe(
            bottom_3_display.style.format({'Gasto Total':'${:.0f}', 'Lucro':'${:.0f}'}), 
            hide_index=True, 
            use_container_width=True
        )

    # Mapeamento de Região:
    Estado_to_Região = {
        'Alabama': 'South', 'Alaska': 'West', 'Arizona': 'West', 'Arkansas': 'South',
        'California': 'West', 'Colorado': 'West', 'Connecticut': 'Northeast', 'Delaware': 'South',
        'Florida': 'South', 'Georgia': 'South', 'Hawaii': 'West', 'Idaho': 'West',
        'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest',
        'Kentucky': 'South', 'Louisiana': 'South', 'Maine': 'Northeast', 'Maryland': 'South',
        'Massachusetts': 'Northeast', 'Michigan': 'Midwest', 'Minnesota': 'Midwest', 'Mississippi': 'South',
        'Missouri': 'Midwest', 'Montana': 'West', 'Nebraska': 'Midwest', 'Nevada': 'West',
        'New Hampshire': 'Northeast', 'New Jersey': 'Northeast', 'New Mexico': 'West', 'New York': 'Northeast',
        'North Carolina': 'South', 'North Dakota': 'Midwest', 'Ohio': 'Midwest', 'Oklahoma': 'South',
        'Oregon': 'West', 'Pennsylvania': 'Northeast', 'Rhode Island': 'Northeast', 'South Carolina': 'South',
        'South Dakota': 'Midwest', 'Tennessee': 'South', 'Texas': 'South', 'Utah': 'West',
        'Vermont': 'Northeast', 'Virginia': 'South', 'Washington': 'West', 'West Virginia': 'South',
        'Wisconsin': 'Midwest', 'Wyoming': 'West'
    }
    if 'Região' not in df.columns:
        df['Região'] = df['Estado'].map(Estado_to_Região)
    
    # Define as cores para as regiões (mesmo que nem todas sejam usadas)
    Região_colors = {
            'Nordeste': '#1f77b4',  # Azul
            'Sul': '#ff7f0e',       # Laranja
            'Centro-Oeste': '#2ca02c',  # Verde
            'Oeste': '#d62728',     # Vermelho
            'Centro': '#9467bd',    # Roxo
            'Leste': '#8c564b'      # Marrom
        }
        
    alpha_val = 0.8  # Transparência para as barras
    
with tab6:
    st.markdown("## Análise Regiãoal Estratégica")
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # ====================================================
    # Gráfico 1: Top 10 Estados com Maior Prejuízo
    st.markdown("### Top 10 Estados com Maior Prejuízo")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Agrupa os dados por estado e região e soma os lucros
    Estado_Lucro = df.groupby(['Estado', 'Região'])['Lucro'].sum().reset_index()  # Ajustando nomes das colunas
    top_Estados = Estado_Lucro.nsmallest(10, 'Lucro')  # Alterando para 'Lucro', conforme renomeação
    
    # Estilização do gráfico
    sns.set_style("whitegrid")
    barplot1 = sns.barplot(
        data=top_Estados, 
        x='Lucro',  # Ajustando para a nova coluna
        y='Estado',  # Ajustando para a nova coluna
        hue='Região',  # Ajustando para a nova coluna
        palette=Região_colors, 
        ax=ax1,
        edgecolor='black',
        linewidth=1.2
    )
    
    for patch in ax1.patches:
        patch.set_alpha(0.85)
    
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1)
    for p in ax1.patches:
        x = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax1.text(x, y, f'{x:,.0f}', ha='left', va='center', fontsize=10, color='black')
    
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax1.set_xlabel('Lucro Total (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('')
    sns.despine()
    fig1.tight_layout()
    st.pyplot(fig1)
    
    # Insights para o gráfico 1
    st.markdown("#### Insights:")
    st.markdown("""
    - Os 10 estados com maior prejuízo mostram que existem áreas mais difíceis para o negócio alavancar, como influência de custos operacionais ou baixa demanda;
    - A correlação entre as perdas de lucro e as regiões ajuda a identificar áreas com maior necessidade de intervenção.
    """)
    
    # ====================================================
    # Gráfico 2: Desempenho de Vendas por Região
    st.markdown("### Desempenho de Vendas por Região")
    Vendas_by_Região = df.groupby('Região')[['Vendas', 'Lucro']].sum().reset_index().sort_values(by='Vendas', ascending=False)  # Ajustando nomes das colunas
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    barplot2 = sns.barplot(data=Vendas_by_Região, x='Região', y='Vendas', palette=Região_colors, ax=ax2)  # Alterando para 'Vendas'
    for patch in ax2.patches:
        patch.set_alpha(alpha_val)
    # Adiciona os valores sobre as barras
    for p in ax2.patches:
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width() / 2., height, f'{height:,.0f}', ha='center', va='bottom', fontsize=12, color='black')
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y/1000:.0f}K'))
    ax2.set_xlabel('Região', fontsize=14)
    ax2.set_ylabel('Vendas Totais (K)', fontsize=14)
    sns.despine()
    fig2.tight_layout()
    st.pyplot(fig2)

    # Insights para o gráfico 2
    st.markdown("#### Insights:")
    st.markdown("""
    - Uma análise mais detalhada sobre o lucro pode revelar margens de venda mais ou menos eficientes em diferentes regiões, não adianta olhar apenas a venda sem ver o lucro.
    """)
    
    # Adicionando legenda para os estados
    st.markdown("#### Clusterização dos Estados por Desempenho")

    # ====================================================
    # Gráfico 3: Clusterização dos Estados por Desempenho
    # Agrupar os dados por estado e região
    cluster_data = df.groupby(['Estado', 'Região']).agg({'Vendas': 'sum', 'Lucro': 'sum'}).reset_index()

    # Calculando a margem
    cluster_data['Margin'] = (cluster_data['Lucro'] / cluster_data['Vendas']) * 100
    
    # Normalizando os dados para clusterização
    features = cluster_data[['Vendas', 'Lucro', 'Margin']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Aplicando KMeans para clusterização
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Gerando o gráfico de clusterização
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # Scatter plot com base no cluster
    scatter = sns.scatterplot(data=cluster_data, x='Vendas', y='Lucro', hue='Cluster', palette="Set1", s=150, ax=ax3, alpha=0.7)
    
    # Identificando os centroids
    centroids = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids)
    
    # Identificando os estados exemplares para cada cluster
    cluster_0_Estados = cluster_data[cluster_data['Cluster'] == 0].head(2)
    cluster_1_Estados = cluster_data[cluster_data['Cluster'] == 1].head(2)
    cluster_2_Estados = cluster_data[cluster_data['Cluster'] == 2].head(2)
    
    # Função para ajustar os textos e bolinhas
    def adjust_text_position(Estado_data, ax, cluster_color):
        for _, row in Estado_data.iterrows():
            ax.scatter(row['Vendas'], row['Lucro'], s=150, color=cluster_color, alpha=0.7)  # Bolinhas com a cor do cluster
            ax.text(row['Vendas'], row['Lucro'] + 150, row['Estado'], fontsize=10, ha='center', color='black', fontweight='light')
    
    # Adicionando as bolinhas maiores com nomes dos estados
    adjust_text_position(cluster_0_Estados, ax3, 'red')
    adjust_text_position(cluster_1_Estados, ax3, 'blue')
    adjust_text_position(cluster_2_Estados, ax3, 'green')
    
    # Ajustando o gráfico
    ax3.set_xlabel("Vendas Totais (K)", fontsize=14)
    ax3.set_ylabel("Lucro Total (K)", fontsize=14)
    
    # Formatação dos eixos para exibir valores em K
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1000:.0f}K'))
    
    # Removendo a legenda de clusters do gráfico
    ax3.get_legend().remove()
    
    sns.despine()
    fig3.tight_layout()
    st.pyplot(fig3)
    
    # Insights para o gráfico 3
    st.markdown("#### Insights:")
    st.markdown("""
    - Foram utilizadas as variáveis de margem de lucro, lucro e venda para a clusterização;
    - A clusterização dos estados permite identificar padrões de desempenho semelhantes entre eles, o que ajuda no uso de estratégias direcionadas a cada grupo;
    ##### Cluster 0 (Azul):
    Cluster com vendas moderadas e lucros razoáveis. Os estados que estão nesse cluster não estão nem entre os mais baixos nem entre os mais altos em vendas e lucro. Isso poderia ser indicado por uma combinação de vendas médias e margem razoável.
    
    ##### Cluster 1 (Vermelho):
    Se a venda for razoável, mas o lucro negativo, então este cluster pode indicar problemas de margem. Margens de lucro são pequenas ou até negativas, sugerindo um custo mais alto em relação à receita.
    
    ##### Cluster 2 (Verde):
    Este cluster parece ter altas vendas e lucros elevados, o que significa que são os estados mais bem-sucedidos, onde tanto as receitas quanto o lucro estão em alta.
    Todavia, a clusterização colocou apenas dois estados nesse cluster, quase como se fossem muito excepcionais em relação aos outros.
    """)
    
    # ====================================================
    # Gráfico 4: Heatmap de Correlações por Região (em gráficos pequenos lado a lado)
    
    st.markdown("### Heatmap de Correlações por Região")

    # Criando uma lista de regiões para iterar
    Regiãos = df['Região'].unique()
    
    # Definindo o número de colunas de subgráficos (para exibir lado a lado)
    n_cols = len(Regiãos)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))  # Ajuste de tamanho para o número de regiões
    
    # Iterando por cada região para calcular as correlações e gerar o heatmap
    for i, Região in enumerate(Regiãos):
        Região_data = df[df['Região'] == Região]
    
        # Calculando a matriz de correlação para 'Vendas' e 'Lucro'
        corr_matrix = Região_data[['Vendas', 'Lucro']].corr()
    
        # Plotando o heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=2, vmin=-1, vmax=1, ax=axes[i])
        axes[i].set_title(f'Relação {Região}', fontsize=12)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=45, va='top')
    
    # Ajustando o layout para melhor visualização
    plt.tight_layout()
    
    # Exibindo o gráfico
    st.pyplot(fig)
    
    # Insights para o gráfico 4
    st.markdown("#### Insights:")
    st.markdown("""
    - **Correlação forte**: Nas três regiões com correlação forte (todas exceto sul), existe uma relação clara entre as vendas e o lucro. Isso indica que as mudanças em uma variável (vendas) afetam diretamente o lucro. Uma alta correlação positiva sugere que vendas mais altas geram mais lucro, enquanto uma correlação negativa indica que aumentos em vendas podem diminuir o lucro (talvez devido a custos elevados).
      
    - **Correlação fraca**: Na região sul, a correlação entre vendas e lucro é fraca ou inexistente. Isso pode indicar que, embora as vendas aumentem, elas não estão impactando de maneira significativa o lucro, sugerindo um problema de eficiência nos custos ou margens muito baixas. Em alguns casos, isso pode sinalizar a necessidade de revisar a estratégia de vendas ou os custos operacionais nessa região.
    
    - **Impacto Estratégico**: A análise dessas correlações pode ajudar a identificar onde os esforços de vendas são mais eficazes, bem como medidas necessárias para melhorar a rentabilidade nas áreas de baixo desempenho.
    
    - **O que fazer?**: 
      - Para as regiões com correlação forte, é recomendável continuar incentivando o aumento de vendas, já que isso tende a impactar positivamente o lucro;
      - Para a região com correlação fraca, seria interessante investigar mais a fundo as razões para o baixo impacto das vendas sobre o lucro, como custos elevados ou margens pequenas, e avaliar possíveis estratégias de melhoria.
    """)
