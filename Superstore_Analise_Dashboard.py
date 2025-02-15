import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

st.set_page_config(page_title="Dashboard Superstore" ) # layout="wide"
sns.set_style("whitegrid")

# Função para carregar dados com cache
@st.cache_data
def load_data():
    # Arquivo local
    #path = r"C:\Python_projetos\Superstore_df\df\Sample_Superstore.csv"
    #df = pd.read_csv(path, delimiter=',', encoding='latin1')
    # Arquivo no git
    url = 'https://raw.githubusercontent.com/Chitolina/Dashboard_Stremio_Superstore_Sales/main/df/Sample_Superstore.csv'
    df = pd.read_csv(url, delimiter=',', encoding='latin1')
    
    # Função para corrigir datas
    def corrigir_data(data_str):
        try: return pd.to_datetime(data_str, format='%d/%m/%Y')
        except ValueError:
            try: return pd.to_datetime(data_str, format='%m/%d/%Y')
            except ValueError: return pd.NaT
    
    df['Order Date'] = df['Order Date'].apply(corrigir_data)
    df = df.dropna(subset=['Order Date'])
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    df['Profit Margin'] = df['Profit'] / df['Sales']  # Nova coluna
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
    col1, col2 = st.columns([1,1.5])  # Ajuste as proporções conforme necessário
    
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
            df[['Sales', 'Profit', 'Discount', 'Quantity']].corr(),
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
    sns.scatterplot(data=df, x='Sales', y='Profit', hue='Category', 
                    palette='plasma', alpha=0.7, s=200, edgecolor='black', linewidth=1.5)
    sns.regplot(data=df, x='Sales', y='Profit', scatter=False, 
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
    barplot = sns.barplot(data=df, x='Category', y='Profit Margin', 
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
    Nesse caso o a margem e lucro seguiram condizentes com as categorias, mas poderia haver o caso de uma categoria ter margem maior, porém não vender tanto e, assim,
    apresentar menos lucro que outra.
    """)
    
    # Calcular Vendas e Lucro por Categoria
    sales_profit_by_category = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
    
    # Criar gráfico de barras lado a lado
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    categories = sales_profit_by_category['Category']
    x = np.arange(len(categories))  # Posições das barras
    
    # Definir cores mais suaves e harmoniosas
    colors = ['#4C72B0', '#55A868']  # Azul e verde
    
    # Criar barras lado a lado
    bars1 = ax1.bar(x - bar_width/2, sales_profit_by_category['Sales'], width=bar_width, 
                    label='Vendas', color=colors[0], alpha=0.85)
    bars2 = ax1.bar(x + bar_width/2, sales_profit_by_category['Profit'], width=bar_width, 
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
    top8_sales = df.groupby(['Category', 'Sub-Category'])[['Sales', 'Profit']].sum().nlargest(8, 'Sales').reset_index()
    
    # Criar rótulo combinando Subcategoria + Categoria
    top8_sales['Label'] = top8_sales['Sub-Category'] + " (" + top8_sales['Category'] + ")"
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(14, 8))  
    bar_width = 0.3  
    y_pos = np.arange(len(top8_sales))
    
    # Criar barras de vendas e lucros
    ax.barh(y_pos - bar_width/2, top8_sales['Sales'], height=bar_width, label='Vendas', color='#4C72B0', alpha=0.85)
    ax.barh(y_pos + bar_width/2, top8_sales['Profit'], height=bar_width, label='Lucro', color='#55A868', alpha=0.85)
    
    # Adicionar rótulos nas barras
    for bar in ax.patches:
        ax.text(bar.get_width() + 3000, bar.get_y() + bar.get_height()/2, f"${bar.get_width()/1000:.1f}k", va='center', fontsize=14)
    
    # Ajustar visualização
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top8_sales['Label'], fontsize=14)
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
    top_sales = df.groupby('Sub-Category').agg({'Sales':'sum', 'Profit':'sum'}).reset_index()
    
    # Calcular a porcentagem de lucro sobre as vendas
    top_sales['Profit Margin %'] = (top_sales['Profit'] / top_sales['Sales']) * 100
    
    # Plotar gráfico
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    scatter = sns.scatterplot(data=top_sales, x='Sales', y='Profit', 
                             hue='Sub-Category', size='Sales', sizes=(50, 500),
                             palette='Paired', alpha=0.7, edgecolor='black')
    
    # Adicionar as porcentagens de lucro acima das bolinhas com tamanho menor e mais acima
    for i, row in top_sales.iterrows():
        ax3.text(row['Sales'], row['Profit'] + 4000,
                 f"{row['Profit Margin %']:.1f}%", 
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
    df['Discount_bins'] = pd.cut(df['Discount'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1], 
                                 labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
    
    # Calcular média e mediana por faixa de desconto
    discount_analysis = df.groupby('Discount_bins')['Profit'].agg(['mean', 'median']).reset_index()
    
    # Criar o gráfico
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Gráfico de média
    sns.lineplot(data=discount_analysis, x='Discount_bins', y='mean', 
                 marker='o', markersize=10, color='#FF4500', linewidth=2.5, label='Média')
    
    # Gráfico de mediana
    sns.lineplot(data=discount_analysis, x='Discount_bins', y='median', 
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
    discount_counts = df['Discount_bins'].value_counts().reset_index()
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
    cleaned_data = df[(df['Profit'] < df['Profit'].mean() + 3 * df['Profit'].std()) &
                      (df['Profit'] > df['Profit'].mean() - 3 * df['Profit'].std())]
    
    # Processar dados
    cleaned_data['Discount_bins'] = pd.cut(cleaned_data['Discount'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1], 
                                           labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
    
    # Calcular média e mediana por faixa de desconto
    discount_analysis = cleaned_data.groupby('Discount_bins')['Profit'].agg(['mean', 'median']).reset_index()
    
    # Criar o gráfico
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Gráfico de média
    sns.lineplot(data=discount_analysis, x='Discount_bins', y='mean', 
                 marker='o', markersize=10, color='#FF4500', linewidth=2.5, label='Média')
    
    # Gráfico de mediana
    sns.lineplot(data=discount_analysis, x='Discount_bins', y='median', 
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
    st.markdown("####  Boxplot: Distribuição do Lucro por Faixa de Desconto")
    fig1, ax1 = plt.subplots(figsize=(12, 6))  # Ajustando o tamanho para ser igual aos outros gráficos
    
    # Customizando o boxplot
    sns.boxplot(x='Discount_bins', y='Profit', data=cleaned_data, ax=ax1, 
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
        Destaca-se os **3 clientes mais rentáveis** e os **3 menos rentáveis**, permitindo uma análise comparativa (será que existe um padrão de categorias preferíveis por eles?)
        
        Observa-se que a maioria dos clientes se concentra em uma faixa de lucro de até **2 mil**, associada a gastos de até **10 mil**. 
        No entanto, há casos de clientes com gastos moderados que geram lucros baixos, assim como clientes com gastos elevados que contribuem significativamente para o lucro. 
    """)
    
    # Processar dados
    customer_profit = df.groupby('Customer Name').agg({'Sales':'sum', 'Profit':'sum'}).reset_index()
    top_3 = customer_profit.nlargest(3, 'Profit')
    bottom_3 = customer_profit.nsmallest(3, 'Profit')
    
    # Plotar gráfico
    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = sns.scatterplot(
        data=customer_profit, x='Sales', y='Profit', 
        hue='Profit', size='Profit', palette='rocket_r',
        sizes=(50, 500), alpha=0.7, edgecolor='black'
    )
    
    # Destacar top/bottom
    for i in range(3):
        ax.annotate(top_3['Customer Name'].iloc[i], 
                   (top_3['Sales'].iloc[i], top_3['Profit'].iloc[i]),
                   textcoords="offset points", xytext=(0,15), 
                   ha='center', fontsize=12, weight='bold', color='darkgreen')
        
        ax.annotate(bottom_3['Customer Name'].iloc[i], 
                   (bottom_3['Sales'].iloc[i], bottom_3['Profit'].iloc[i]),
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
        
        # Adicionando a porcentagem de consumo por categoria
        customer_category_profit = df.groupby(['Customer Name', 'Category']).agg({'Sales':'sum'}).reset_index()
        customer_category_profit['Percentage'] = customer_category_profit.groupby('Customer Name')['Sales'].transform(lambda x: (x / x.sum()) * 100)
        top_3_categories = customer_category_profit[customer_category_profit['Customer Name'].isin(top_3['Customer Name'])]
        
        # Mostrar a tabela com categorias e porcentagens
        top_3_display = top_3[['Customer Name', 'Sales', 'Profit']].rename(columns={'Customer Name':'Cliente', 'Sales':'Gasto Total', 'Profit':'Lucro'})
        
        # Adicionar as porcentagens por categoria à tabela
        for customer in top_3['Customer Name']:
            categories = top_3_categories[top_3_categories['Customer Name'] == customer]
            category_percentages = ', '.join([f"{row['Category']}: {row['Percentage']:.1f}%" for _, row in categories.iterrows()])
            top_3_display.loc[top_3_display['Cliente'] == customer, 'Categorias (%)'] = category_percentages

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
        bottom_3_categories = customer_category_profit[customer_category_profit['Customer Name'].isin(bottom_3['Customer Name'])]
        
        # Mostrar a tabela com categorias e porcentagens
        bottom_3_display = bottom_3[['Customer Name', 'Sales', 'Profit']].rename(columns={'Customer Name':'Cliente', 'Sales':'Gasto Total', 'Profit':'Lucro'})
        
        # Adicionar as porcentagens por categoria à tabela
        for customer in bottom_3['Customer Name']:
            categories = bottom_3_categories[bottom_3_categories['Customer Name'] == customer]
            category_percentages = ', '.join([f"{row['Category']}: {row['Percentage']:.1f}%" for _, row in categories.iterrows()])
            bottom_3_display.loc[bottom_3_display['Cliente'] == customer, 'Categorias (%)'] = category_percentages

        st.dataframe(
            bottom_3_display.style.format({'Gasto Total':'${:.0f}', 'Lucro':'${:.0f}'}), 
            hide_index=True, 
            use_container_width=True
        )

    # Análise Estratégica
    st.markdown("""
    #### Insights:
    1. **Clientes de Alto Risco**:  
       - Mesmo com gasto significativo, geram prejuízo.
    2. **Oportunidades de nos clientes medianos**:  
       - Clientes rentáveis com gasto médio-alto podem ser melhor estudados.
    """)


    

with tab6:
    st.markdown("## Análise Regional Estratégica")
    
    # Gráfico 1 - Estados com maior prejuízo
    st.markdown("### Top 10 Estados com Maior Prejuízo")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    state_profit = df.groupby(['State', 'Region'])['Profit'].sum().reset_index().sort_values(by='Profit')
    sns.barplot(data=state_profit.head(10), x='Profit', y='State', hue='Region', palette='plasma', ax=ax1)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.xlabel('Lucro Total (K)')
    plt.ylabel('')
    st.pyplot(fig1)

    # Gráfico 2 - Vendas por região
    st.markdown("### Desempenho de Vendas por Região")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sales_by_region = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index().sort_values(by='Sales', ascending=False)
    sns.barplot(data=sales_by_region, x='Region', y='Sales', palette='plasma', ax=ax2)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y/1000:.0f}K'))
    plt.xlabel('Região')
    plt.ylabel('Vendas Totais (K)')
    st.pyplot(fig2)

    # Explicação dos resultados
    st.markdown("""
    ####  Principais Insights:
    1. **Foco em Correções Regionais**:  
       - Estados com prejuízos persistentes requerem revisão de estratégias locais
    2. **Otimização de Recursos**:  
       - Alocar equipes comerciais conforme potencial de cada região
    3. **Benchmarking Interno**:  
       - Replicar práticas de regiões bem-sucedidas para áreas problemáticas
    """)
