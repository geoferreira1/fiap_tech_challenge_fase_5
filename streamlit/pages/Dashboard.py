# ==========================================================================
# Import de bibliotecas
# ==========================================================================

# Bibliotecas do Sistema e Utilitários
import io            # Manipulação de fluxos de dados (entrada/saída) em memória
import requests      # Realização de requisições HTTP para buscar arquivos externos
import time          # Funções relacionadas a tempo e controle de execução
import unicodedata   # Manipulação de caracteres Unicode e normalização de strings

# Processamento e Manipulação de Dados
import numpy as np   # Suporte a arrays multidimensionais e funções matemáticas
import pandas as pd  # Manipulação e análise de dados estruturados (DataFrames)

# Visualização de Dados
import matplotlib.pyplot as plt # Criação de gráficos estáticos e customização de figuras
import seaborn as sns           # Visualização de dados estatísticos baseada em Matplotlib
import streamlit as st          # Framework para criação de dashboards e aplicações web

# ==========================================================================
# Config página
# ==========================================================================

st.set_page_config( # Define as configurações globais da página web
    page_title="Passos Mágicos | Jornada de Transformação", # Título exibido na aba do navegador
    page_icon="✨", # Ícone (favicon) exibido na aba do navegador
    layout="wide" # Configura o layout para utilizar toda a largura da tela
) # Encerra a configuração da página

# Estilo global Set2 para harmonia visual
sns.set_theme(style="whitegrid") # Define o tema visual do Seaborn com fundo branco e grade
PALETA = sns.color_palette("Set2") # Cria uma paleta de cores fixa baseada no esquema Set2

# ==========================================================================
# Funções de Dados (ETL)
# ==========================================================================

@st.cache_data # Decorador para armazenar os dados em cache e otimizar a performance
def load_data(): # Inicia a definição da função de carga e limpeza
    """Carrega dados via URL e prepara indicadores para a narrativa.""" # Docstring da função
    url = "https://raw.githubusercontent.com/geoferreira1/fiap_tech_challenge_fase_5/main/data_processed/df_unificado.csv" # URL do dataset
    df = pd.read_csv(url) # Lê o arquivo CSV remoto e converte em objeto DataFrame
    
    # Saneamento e Tipagem
    df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce').fillna(0).astype(int) # Converte idade para inteiro tratando erros
    df['GENERO'] = df['GENERO'].astype(str).str.capitalize() # Padroniza gênero com a primeira letra em maiúscula
    df['PEDRA'] = df['PEDRA'].fillna('NÃO CLASSIFICADO') # Garante preenchimento de nulos para não quebrar filtros
    df['PONTO_VIRADA'] = df['PONTO_VIRADA'].fillna('Não Inf.') # Padroniza nulos do ponto de virada como informação inexistente
    df['ANO'] = df['ANO'].astype(int) # Certifica que o ano é tratado como número inteiro
    
    return df # Retorna o DataFrame processado

df = load_data() # Executa a função de carga e armazena o resultado na variável df

def classificar_indicador(valor, nome_indicador, manual=None): # Função para rotular indicadores com base em regras
    """Mapeia valores numéricos para categorias qualitativas utilizando um dicionário de regras.""" # Docstring
    if pd.notna(manual): return manual # Retorna a marcação manual imediatamente caso ela exista
    if pd.isna(valor): return "N/A" # Retorna "N/A" caso o valor de entrada seja nulo
    
    # Dicionário central contendo todas as regras de negócio do projeto Passos Mágicos
    mapa_regras = { # Inicia a estrutura de catálogo que agrupa os critérios de todos os indicadores
        'IAN': {10.0: 'Adequado', 5.0: 'Mod. Defasado', 0.0: 'Sev. Defasado'}, # Regras de adequação escolar
        'IEG': {8.5: 'Alto', 6.0: 'Médio', 0.0: 'Baixo'}, # Regras de engajamento do aluno
        'IDA': {7.5: 'Alto (>=7.5)', 5.0: 'Médio (5-7.5)', 0.0: 'Baixo (<5)'}, # Regras de desempenho acadêmico
        'IPS': {7.5: 'Adequado', 5.0: 'Em Alerta', 0.0: 'Crítico'}, # Regras de índice psicossocial
        'IPP': {8.0: 'Excelente', 7.0: 'Adequado', 0.0: 'Insuficiente'}, # Regras de potencial psicopedagógico
        'IAA': {8.5: 'Alta', 6.0: 'Média', 0.0: 'Baixa'}, # Regras de autoavaliação do aluno
        'IPV': {7.0: 'Sim', 0.0: 'Não'} # Regras para o indicador de Ponto de Virada
    } # Finaliza o dicionário de regras
    
    nome_indicador = nome_indicador.upper() # Normaliza o nome do indicador para maiúsculas
    regras = mapa_regras.get(nome_indicador) # Recupera o conjunto específico de limites solicitado
    
    if not regras: return "Indicador Inválido" # Retorna erro caso o nome do indicador não exista
        
    for limite, rotulo in regras.items(): # Percorre os limites definidos, do maior para o menor
        if valor >= limite: return rotulo # Retorna o primeiro rótulo que satisfaça a condição
        
    return "N/A" # Retorna padrão caso não atinja nenhuma das faixas

# ==========================================================================
# Barra Lateral (Filtros Estratégicos)
# ==========================================================================

with st.sidebar: # Inicia o bloco de componentes da barra lateral esquerda
    st.title("🚀 Insights") # Adiciona título Markdown na sidebar
    st.info("Este painel narra como a Passos Mágicos resgata o potencial de crianças e jovens.") # Card informativo
    st.title("🔍 Filtros da Jornada") # Exibe o título principal da sidebar
    st.markdown("Ajuste os filtros para focar em grupos específicos de alunos.") # Adiciona texto explicativo

    anos = sorted(df['ANO'].unique()) # Obtém e ordena os anos únicos presentes na base
    ano_sel = st.multiselect("Ciclos Anuais", anos, default=anos) # Cria seleção múltipla para ciclos anuais
    
    # Retornando o Filtro de Pedras
    lista_pedras = [p for p in df['PEDRA'].unique() if p != 'NÃO CLASSIFICADO'] # Gera lista de pedras excluindo nulos
    pedra_sel = st.multiselect("Nível de Evolução (Pedra)", sorted(lista_pedras), default=lista_pedras) # Seleção de pedras

    generos = sorted(df['GENERO'].unique()) # Obtém e ordena os gêneros únicos presentes
    gen_sel = st.multiselect("Gênero", generos, default=generos) # Cria seleção múltipla para gêneros
    
    # Filtro dinâmico
    df_f = df[ # Inicia a aplicação dos filtros no DataFrame principal
        (df['ANO'].isin(ano_sel)) & # Filtra as linhas que correspondem aos anos selecionados
        (df['PEDRA'].isin(pedra_sel if pedra_sel else df['PEDRA'].unique())) & # Filtra as pedras selecionadas
        (df['GENERO'].isin(gen_sel)) # Filtra os gêneros selecionados
    ].copy() # Cria uma cópia independente do DataFrame resultante

# ==========================================================================
# Dashboard - A Jornada de Transformação (Storytelling)
# ==========================================================================

st.caption("✨ PEDE Analytics | Ong Passos Mágicos <sup>1</sup>", unsafe_allow_html=True) # Exibe legenda superior estilizada
st.title("✨ Passos Mágicos: A Jornada da Transformação") # Exibe o título principal do dashboard
st.markdown("""
    *Toda criança possui um talento escondido. Nossa missão é lapidar esse potencial. 
    Abaixo, narramos como os indicadores do PEDE revelam o impacto real na vida dos nossos alunos.*
""") # Adiciona texto de introdução do storytelling
st.divider() # Adiciona uma linha divisória horizontal

if df_f.empty: # Verifica se o resultado dos filtros é um conjunto vazio
    st.warning("Selecione os filtros para iniciar a narrativa.") # Exibe aviso caso não existam dados selecionados
else: # Inicia a renderização caso existam dados
    # --- ORGANIZAÇÃO EM ATOS NARRATIVOS ---
    ato1, ato2, ato3, ato4, ato5 = st.tabs([ # Cria as abas de navegação para os atos narrativos
        "📍 A Chegada", 
        "📈 O Desenvolvimento", 
        "🧠 A Virada de Chave", 
        "🏆 A Consolidação",
        "🌟 Síntese final"
    ]) # Encerra a criação das abas

    # --------------------------------------------------------------------------
    # A chegada (Q1 e Q6)
    # --------------------------------------------------------------------------
    with ato1: # Define o conteúdo da primeira aba
        st.header("Identificando a Vulnerabilidade") # Cabeçalho do Ato I
        st.write("""
            Nossa história começa no acolhimento. O primeiro desafio é a **defasagem**. 
            Muitos chegam com anos de atraso escolar, mas será que essa barreira é apenas acadêmica?
        """) # Descrição do contexto narrativo do Ato I
        
        col1, col2 = st.columns(2) # Divide a interface em duas colunas verticais
        with col1: # Inicia a primeira coluna
            st.subheader("1. Adequação do nível (IAN)") # Subtítulo do indicador IAN
            st.markdown("Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao longo do ano?") # Pergunta analítica
            
            # 1. REMOVEMOS O DROPNA: Para os números baterem com o Excel, não podemos deletar linhas nulas.
            df_ian = df_f.copy() # Cria cópia local para análise de IAN
            
            # Garantimos que o ANO seja tratado como texto para evitar o erro de decimais (2022.0, 2022.5)
            df_ian['ANO'] = df_ian['ANO'].astype(str) # Converte ano para string

            # 2. Aplica a função: Alunos sem nota agora viram "N/A" em vez de sumirem
            df_ian['IAN_Descricao'] = df_ian['IAN'].apply(lambda x: classificar_indicador(x, 'IAN')) # Classifica scores

            # 3. Define a ordem: Adicionei o 'N/A' para você enxergar onde estão os alunos que faltavam
            ordem = ['Sev. Defasado', 'Mod. Defasado', 'Adequado', 'N/A'] # Define ordem categórica

            # --- GRÁFICO ÚNICO ---
            fig, ax = plt.subplots(figsize=(8, 5)) # Cria a figura e o eixo do Matplotlib
            
            sns.histplot( # Gera o gráfico de barras empilhadas
                data=df_ian, # Dados utilizados
                x='ANO', # Eixo X baseado no ano
                hue='IAN_Descricao', # Cores baseadas na classificação
                multiple='stack', # Empilha as categorias
                palette='Set2', # Aplica a paleta visual
                shrink=0.7, # Ajusta largura das barras
                linewidth=0, # Remove bordas das barras
                discrete=True, # Trata eixo X como discreto
                ax=ax # Vincula ao eixo criado
            ) # Encerra plotagem

            # --- REMOVE AS LINHAS DE GRADE ---
            ax.grid(False) # Desativa as linhas de grade do gráfico

            # 5. Adiciona o rótulo de dados
            for container in ax.containers: # Itera sobre os containers de barras
                ax.bar_label(container, label_type='center', fontsize=10, fontweight='bold') # Insere valores centrais

            # 6. Personalização
            ax.set_title('Distribuição de Alunos por Nível de Adequação (IAN)', fontsize=14, fontweight='bold') # Define título
            ax.set_xlabel('Ano letivo') # Define rótulo do eixo X
            ax.set_ylabel('Quantidade de Alunos') # Define rótulo do eixo Y

            # 7. Ajustando a legenda (Interna ou Externa)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title='Nível IAN') # Posiciona legenda lateralmente

            # 8. FINALIZAÇÃO
            plt.tight_layout() # Ajusta automaticamente o layout da figura
            st.pyplot(fig) # Renderiza o gráfico na aplicação Streamlit
            
            st.markdown("""
                ### 🎬 O Início da Jornada

                Este gráfico revela o ponto de partida do aluno dentro do programa.

                📌 A concentração nos níveis **Severamente Defasado** e **Moderadamente Defasado** mostra o tamanho do desafio assumido pela ONG.

                💡 Quando observamos crescimento na categoria **Adequado** ao longo dos anos, temos evidência concreta de transformação educacional.

                🎯 Estratégia: quanto maior a vulnerabilidade inicial, maior deve ser a intensidade do reforço pedagógico nas fases iniciais da jornada.
            """)
        with col2: # Inicia a segunda coluna
            st.subheader("6. Aspectos psicopedagógicos (IPP)") # Subtítulo do indicador IPP
            st.markdown("As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?") # Pergunta analítica

            # 1. Preparação: Filtramos e classificamos
            df_ipp = df_f.dropna(subset=['IPP', 'IAN']).copy() # Remove nulos apenas para análise de médias
            df_ipp['IAN_Descricao'] = df_ipp['IAN'].apply(lambda x: classificar_indicador(x, 'IAN')) # Categoriza conforme IAN
            ordem_ian = ['Sev. Defasado', 'Mod. Defasado', 'Adequado'] # Define ordem do eixo X

            # 2. Criação da Figura (Média do IPP por Nível de IAN)
            fig, ax = plt.subplots(figsize=(8, 5)) # Inicia figura de barras
            
            # Cálculo da média para o gráfico de barras
            ipp_por_ian = df_ipp.groupby('IAN_Descricao')['IPP'].mean().reindex(ordem_ian).reset_index() # Calcula médias agrupadas
            
            # 3. Gráfico de Barras (Opção 2 do seu material)
            sns.barplot(data=ipp_por_ian, x='IAN_Descricao', y='IPP', palette='Set2', ax=ax) # Gera barras de médias
            
            # Remove grades e contornos conforme seu padrão
            ax.grid(False) # Desativa grade visual
            for patch in ax.patches: # Itera sobre as barras
                patch.set_edgecolor('none') # Remove contorno individual

            # 4. Rótulos de dados (Médias em cima das barras)
            for container in ax.containers: # Itera sobre containers
                ax.bar_label(container, fmt='%.2f', padding=3, fontweight='bold') # Exibe média com 2 casas decimais

            # 5. Personalização de títulos e eixos
            ax.set_title('Média do IPP por Nível de IAN', fontsize=14, fontweight='bold') # Define título
            ax.set_xlabel('Nível de Adequação Escolar (IAN)') # Rótulo X
            ax.set_ylabel('Média do IPP') # Rótulo Y
            
            plt.tight_layout() # Ajusta layout final
            st.pyplot(fig) # Renderiza no Streamlit
            
            st.markdown("""
                ### 🧠 Potencial Além da Defasagem

                Mesmo alunos com defasagem podem apresentar alto potencial psicopedagógico.

                Isso significa que o problema não é incapacidade — é falta de oportunidade estruturada.

                🎯 Estratégia: investir no desenvolvimento emocional e cognitivo pode acelerar a recuperação acadêmica.
            """)
    # --------------------------------------------------------------------------
    # O DESENVOLVIMENTO (Q2, Q3 E Q4)
    # --------------------------------------------------------------------------
    with ato2: # Define o bloco de conteúdo da segunda aba
        st.header("Lapidando o Conhecimento") # Cabeçalho do Ato II
        st.write("""
            Com o apoio da ONG, o aluno começa a evoluir. Monitoramos não apenas as notas (IDA), 
            mas o brilho nos olhos: o **Engajamento**.
        """) # Descrição do contexto do Ato II
        
    # --- PERGUNTA 2: IDA POR FASE E ANO ---
        st.subheader("2. Desempenho acadêmico (IDA)") # Título da Pergunta 2
        st.markdown("O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo das fases e anos?") # Pergunta analítica
        
        # 1. Preparação dos dados
        df_ida = df_f.dropna(subset=['IDA']).copy() # Filtra apenas alunos com nota IDA registrada
        df_ida['ANO'] = df_ida['ANO'].astype(str) # Padroniza ano como texto
        
        # 2. Classificação
        df_ida['IDA_Categoria'] = df_ida['IDA'].apply(lambda x: classificar_indicador(x, 'IDA')) # Classifica scores IDA
        ordem_ida = ['Baixo (<5)', 'Médio (5-7.5)', 'Alto (>=7.5)'] # Define categorias ordinais

        # 3. Execução do Gráfico
        fig, ax = plt.subplots(figsize=(12, 5)) # Cria moldura larga para distribuição
        
        sns.histplot( # Gera o histograma de desempenho
            data=df_ida, # Dados filtrados
            x='ANO', # Eixo X temporal
            hue='IDA_Categoria', # Cores por nível
            hue_order=ordem_ida, # Segue ordem de categorias
            multiple='stack', # Empilha barras
            palette='Set2', # Aplica paleta
            shrink=0.7, # Ajusta largura
            linewidth=0, # Remove contornos
            discrete=True, # Eixo X discreto
            ax=ax # Vincula ao eixo
        ) # Encerra plot
        
        # 4. Rótulos de dados
        for container in ax.containers: # Itera containers
            ax.bar_label(container, label_type='center', fontsize=10, fontweight='bold') # Insere contagens
        
        # 5. Personalização
        ax.set_title('Distribuição de Alunos por Nível de IDA', fontsize=14, fontweight='bold') # Define título
        ax.set_xlabel('Ano Letivo') # Rótulo X
        ax.set_ylabel('Quantidade de Alunos') # Rótulo Y
        ax.grid(False) # Mantém o fundo limpo
        
        # Legenda externa para não poluir
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title='Nível IDA') # Move legenda
        
        plt.tight_layout() # Ajusta layout
        st.pyplot(fig) # Renderiza no Streamlit
        
        st.markdown("""
        ### 📈 Crescimento Mensurável

        Aqui avaliamos se o esforço virou resultado concreto.

        O aumento da categoria **Alto (>=7.5)** ao longo dos anos indica que a metodologia aplicada está funcionando.

        📌 Se houver concentração persistente na faixa "Baixo", isso sinaliza necessidade de intervenção direcionada.

        🎯 Estratégia: identificar quais práticas pedagógicas foram aplicadas nos ciclos de melhor desempenho e replicá-las.
        """)

        st.divider() # Linha de separação

        col3, col4 = st.columns(2) # Divide em duas colunas para engajamento e autoavaliação
        
        with col3: # Terceira coluna
            # --- PERGUNTA 3: ENGAJAMENTO (APENAS SIM E NÃO) ---
            st.subheader("3. Engajamento nas atividades (IEG)") # Título da Pergunta 3
            st.markdown("O grau de engajamento dos alunos (IEG) tem relação direta com seus indicadores de desempenho (IDA) e do ponto de virada (IPV)?") # Pergunta analítica
            
            # Filtro rigoroso para exibir apenas Sim e Não (removemos 'Não Inf.' e nulos)
            df_pv = df_f[df_f['PONTO_VIRADA'].isin(['Sim', 'Não'])].dropna(subset=['IEG']).copy() # Filtra sim/não
            ieg_pv_media = df_pv.groupby('PONTO_VIRADA')['IEG'].mean().reindex(['Não', 'Sim']).reset_index() # Média por virada
            
            fig, ax = plt.subplots(figsize=(8, 6)) # Cria figura de comparação
            ax_bar = sns.barplot(data=ieg_pv_media, x='PONTO_VIRADA', y='IEG', palette='Set2', ax=ax) # Plot de barras comparativo
            
            # Rótulos nas barras
            for container in ax_bar.containers: # Itera containers
                ax_bar.bar_label(container, fmt='%.2f', padding=3, fontweight='bold') # Rótulos das médias
            
            ax.set_title('Média de Engajamento: Sim vs Não', fontweight='bold') # Título gráfico
            ax.set_xlabel('Atingiu Ponto de Virada?') # Rótulo X
            ax.set_ylabel('Média do IEG') # Rótulo Y
            ax.grid(False) # Remove grade
            
            for patch in ax_bar.patches: # Itera barras individuais
                patch.set_edgecolor('none') # Remove contornos
            
            st.pyplot(fig) # Renderiza

            st.markdown("""
            ### 🚀 O Motor da Transformação

            Alunos que atingem o ponto de virada apresentam engajamento significativamente maior.

            Isso reforça que o sucesso acadêmico começa na atitude, não apenas na técnica.

            🎯 Estratégia: programas de mentoria e incentivo comportamental são fundamentais para acelerar a virada.
            """)

        with col4: # Quarta coluna
            # --- PERGUNTA 4: AUTOAVALIAÇÃO VS REALIDADE ---
            st.subheader("4. Autoavaliação (IAA)") # Título da Pergunta 4
            st.markdown("As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho real (IDA) e engajamento (IEG)?") # Pergunta analítica
            
            fig, ax = plt.subplots(figsize=(8, 6)) # Cria figura para análise de densidade
            sns.kdeplot(df_f['IAA'], label='Autoavaliação (IAA)', fill=True, color=PALETA[0], ax=ax) # Curva de densidade subjetiva
            sns.kdeplot(df_f['IDA'], label='Nota Real (IDA)', fill=True, color=PALETA[1], ax=ax) # Curva de densidade objetiva
            
            ax.grid(False) # Remove linhas de fundo
            ax.set_title("Subjetivo (IAA) vs Objetivo (IDA)", fontweight='bold') # Título gráfico
            ax.set_xlabel("Nota") # Rótulo X
            ax.set_ylabel("Densidade") # Rótulo Y
            ax.legend() # Ativa legenda explicativa
            
            st.pyplot(fig) # Renderiza no Streamlit

            st.markdown("""
            ### 🧠 Percepção vs Realidade

            Quando a autoavaliação (IAA) está alinhada com a nota real (IDA), temos maturidade emocional.

            📌 Desalinhamentos indicam:
            - IAA maior que IDA → excesso de confiança
            - IAA menor que IDA → baixa autoestima

            🎯 Estratégia: trabalhar inteligência emocional para alinhar percepção e desempenho.
            """)

    # --------------------------------------------------------------------------
    # O PONTO DE VIRADA (Q5 e Q7)
    # --------------------------------------------------------------------------
    with ato3: # Define o bloco de conteúdo da terceira aba
        st.header("O Ponto de Virada") # Cabeçalho do Ato III
        st.write("""
            Chegamos ao momento mais crítico: a mudança de mentalidade. 
            O apoio **psicossocial** é o que garante que o aluno não desista no meio do caminho.
        """) # Descrição do contexto do Ato III
        
        col5, col6 = st.columns(2) # Divide em colunas para IPS e Correlação
        with col5: # Quinta coluna
            st.subheader("5. Aspectos psicossociais (IPS)") # Título da Pergunta 5
            st.markdown("Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento?") # Pergunta analítica
            df_ips = df_f.dropna(subset=['IPS', 'IDA', 'IEG']).copy() # Filtra dados psicossociais válidos

            # 2. Aplica a função para criar a coluna de descrição
            df_ips['ANO'] = df_ips['ANO'].astype(str) # Converte ano para texto
            df_ips['IPS_Nivel'] = df_ips['IPS'].apply(lambda x: classificar_indicador(x, 'IPS')) # Classifica níveis IPS
            ordem_ips = ['Crítico', 'Em Alerta', 'Adequado'] # Define escala qualitativa

            # 3. Início da Figura
            fig, ax = plt.subplots(figsize=(8, 6)) # Inicia figura

            # --- GRÁFICO 1: DISTRIBUIÇÃO CATEGÓRICA (Histplot) ---
            ax = sns.histplot(data=df_ips, x='ANO', hue='IPS_Nivel', hue_order=ordem_ips,
                              multiple='stack', palette='Set2', shrink=0.7, linewidth=0,
                              discrete=True, stat='percent', common_norm=False, ax=ax) # Plota distribuição percentual
            for container in ax.containers: # Itera containers
                ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, fontweight='bold') # Rótulos em %
            ax.set_title('Distribuição Psicossocial (IPS) por Ano (%)', fontweight='bold') # Título gráfico
            ax.set_xlabel('Ano Letivo') # Rótulo X
            ax.set_ylabel('Percentual de Alunos (%)') # Rótulo Y
            ax.grid(False) # Remove grade
            st.pyplot(fig) # Renderiza

            st.markdown("""
            ### ⚠️ O Pilar Invisível da Jornada

            Sem estabilidade emocional, o aprendizado não se sustenta.

            A redução do percentual na categoria **Crítico** ao longo do tempo é um indicador silencioso de sucesso estrutural.

            🎯 Estratégia: fortalecer acompanhamento psicossocial nos ciclos iniciais.
            """)

        with col6: # Sexta coluna
            st.subheader("7. Ponto de virada (IPV)") # Título da Pergunta 7
            st.markdown("Quais comportamentos - acadêmicos, emocionais ou de engajamento - mais influenciam o IPV ao longo do tempo?") # Pergunta analítica
            
            # 1. Preparação: Cálculo da correlação
            colunas_analise = ['IDA', 'IEG', 'IPS', 'IAA', 'IPP', 'IPV'] # Seleciona métricas numéricas
            correl_pv = df_f[colunas_analise].corrwith(df_f['INDE']).sort_values(ascending=False) # Calcula correlação com INDE
            
            # 2. Execução do Gráfico
            fig, ax = plt.subplots(figsize=(7.6, 6)) # Figura para barras de força
            
            # Criamos o gráfico de barras horizontais usando a paleta Set2
            sns.barplot(
                x=correl_pv.values, # Valores da correlação
                y=correl_pv.index, # Nomes dos indicadores
                hue=correl_pv.index, # Cores por indicador
                palette='Set2', # Paleta visual
                ax=ax, # Vincula ao eixo
                legend=False # Oculta legenda redundante
            ) # Encerra plot
            
            # 3. Rótulos de dados
            for i, v in enumerate(correl_pv.values): # Itera sobre valores
                ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontweight='bold', fontsize=10) # Rótulos de força lateral
            
            # 4. Personalização
            ax.set_title("Drivers do Sucesso (Correlação com INDE)", fontsize=14, fontweight='bold') # Título gráfico
            ax.set_xlabel("Força da Correlação") # Rótulo X
            ax.set_ylabel("Indicadores") # Rótulo Y
            
            # Remove grades e bordas das barras
            ax.grid(False) # Remove grade
            for patch in ax.patches: # Itera barras
                patch.set_edgecolor('none') # Suaviza barras

            # Ajusta o limite do eixo X para dar espaço aos rótulos
            ax.set_xlim(0, 1.1) # Define escala do eixo X

            plt.tight_layout() # Ajusta layout
            st.pyplot(fig) # Renderiza no Streamlit

            st.markdown("""
            ### 🏆 O Que Realmente Move o Sucesso

            Este gráfico revela quais indicadores possuem maior influência sobre o INDE.

            Quanto maior a correlação, maior o impacto estratégico daquele indicador no resultado final.

            🎯 Estratégia: priorizar investimentos e esforços nos pilares com maior força de correlação.
            """)

    # --------------------------------------------------------------------------
    # O IMPACTO REAL (Q8 e Q10)
    # --------------------------------------------------------------------------
    with ato4: # Define o conteúdo da quarta aba
        st.header("Colhendo Frutos") # Cabeçalho do Ato IV
        st.write("""
            Ao final do ciclo, provamos que o sucesso é **multidimensional**. 
            Não é apenas uma nota, é a união de mente, atitude e esforço.
        """) # Descrição do contexto do Ato IV
        
        col7, col8 = st.columns(2) # Cria colunas finais de performance e evolução
        with col7: # Sétima coluna
            st.subheader("8. Multidimensionalidade dos indicadores") # Título da Pergunta 8
            st.markdown("Quais combinações de indicadores (IDA + IEG + IPS + IPP) melhor explicam o desempenho global do aluno (INDE)?") # Pergunta analítica

            # 1. Preparação dos dados
            indicadores = ['IDA', 'IEG', 'IPS', 'IPP'] # Define pilares
            df_8 = df_f.dropna(subset=indicadores + ['INDE']).copy() # Filtra nulos essenciais

            if df_8.empty: # Caso não existam dados
                st.warning("Dados insuficientes para gerar a análise de combinações com os filtros atuais.") # Exibe aviso
            else: # Caso existam dados
                # Calculamos a média do grupo geral
                media_geral = df_8[indicadores].mean().to_frame().reset_index() # Média geral
                media_geral.columns = ['Indicador', 'Nota'] # Renomeia colunas
                media_geral['Grupo'] = 'Média Geral' # Identifica grupo

                # Calculamos a média dos alunos que estão no Top 20% do INDE (Elite)
                threshold = df_8['INDE'].quantile(0.8) # Define nota de corte dos melhores
                media_elite = df_8[df_8['INDE'] >= threshold][indicadores].mean().to_frame().reset_index() # Média elite
                media_elite.columns = ['Indicador', 'Nota'] # Renomeia colunas
                media_elite['Grupo'] = 'Alunos Alta Performance (Top 20%)' # Identifica elite

                # Unimos os dois para comparação
                df_plot_8 = pd.concat([media_geral, media_elite]) # Concatena para gráfico

                # 2. Criação da Figura
                fig, ax = plt.subplots(figsize=(10, 6)) # Inicia figura

                # Gráfico de barras comparativo
                sns.barplot(
                    data=df_plot_8, # Dados concatenados
                    x='Indicador', # Categorias X
                    y='Nota', # Valores Y
                    hue='Grupo', # Cores por grupo
                    palette='Set2', # Aplica paleta
                    ax=ax # Vincula ao eixo
                ) # Encerra plot

                # 3. Rótulos e Estética
                for container in ax.containers: # Itera containers
                    ax.bar_label(container, fmt='%.2f', padding=3, fontweight='bold') # Rótulos médias

                ax.set_title('Perfil Comparativo: Média Geral vs Elite', fontsize=14, fontweight='bold') # Título gráfico
                ax.set_ylabel('Nota Média') # Rótulo Y
                ax.set_xlabel('Indicadores') # Rótulo X
                ax.set_ylim(0, 11) # Limite escala Y
                ax.grid(False) # Mantém padrão sem grades

                # Remove bordas das barras
                for patch in ax.patches: # Itera barras
                    patch.set_edgecolor('none') # Remove contorno

                # Legenda interna
                ax.legend(title='Grupo', loc='upper left', frameon=True) # Configura legenda

                plt.tight_layout() # Ajusta layout

                # COMANDO CRÍTICO: Exibe no Streamlit
                st.pyplot(fig) # Renderiza

                st.markdown("""
                ### 🌟 O DNA da Alta Performance

                Comparar a média geral com os alunos Top 20% revela o diferencial competitivo.

                Os maiores saltos geralmente aparecem em:
                - Engajamento (IEG)
                - Desempenho Acadêmico (IDA)

                🎯 Estratégia: mapear práticas e comportamentos da elite para replicar nos demais alunos.
                """)
    

        with col8: # Oitava coluna
            st.subheader("10. Efetividade do programa") # Título da Pergunta 10
            st.markdown("Os indicadores mostram melhora consistente ao longo do ciclo nas diferentes fases (Quartzo, Ágata, Ametista e Topázio), confirmando o impacto real do programa?") # Pergunta analítica

            # 1. Preparação: Médias por Pedra
            indicadores_pedras = ['INDE', 'IDA', 'IEG', 'IPS', 'IPP'] # Métricas para jornada
            ordem_pedras = ['QUARTZO', 'AGATA', 'AMETISTA', 'TOPAZIO'] # Jornada evolutiva

            df_pedras = df_f.groupby('PEDRA')[indicadores_pedras].mean().reindex(ordem_pedras).reset_index() # Agrupa médias

            # Transformamos para o formato longo para o Seaborn
            df_plot_10 = df_pedras.melt(id_vars='PEDRA', var_name='Indicador', value_name='Média') # Transpõe dados

            # 2. Execução do Gráfico (Barras Agrupadas)
            fig, ax = plt.subplots(figsize=(10, 6)) # Inicia figura final

            sns.barplot(
                data=df_plot_10, # Dados transpostos
                x='PEDRA', # Eixo X por estágio
                y='Média', # Nota média Y
                hue='Indicador', # Cores por métrica
                palette='Set2', # Aplica paleta
                ax=ax # Vincula eixo
            ) # Encerra plot

            # 3. Rótulos de dados
            for container in ax.containers: # Itera containers
                ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8, fontweight='bold') # Notas médias no topo

            # 4. Personalização
            ax.set_title('Comparativo de Indicadores por Nível de Pedra', fontsize=14, fontweight='bold') # Título gráfico
            ax.set_xlabel('Ciclo de Evolução (Pedra)') # Rótulo X
            ax.set_ylabel('Nota Média') # Rótulo Y
            ax.set_ylim(0, 12) # Ajusta escala Y
            ax.grid(False) # Limpa grade

            # Legenda lateral para não atrapalhar
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title='Indicadores') # Move legenda

            # Remove bordas das barras
            for patch in ax.patches: # Itera barras
                patch.set_edgecolor('none') # Suaviza barras

            plt.tight_layout() # Ajusta layout
            st.pyplot(fig) # Renderiza no Streamlit

            st.markdown("""
            ### 📈 A Jornada Estruturada Funciona

            Cada Pedra representa um estágio de desenvolvimento.

            A progressão consistente dos indicadores valida a metodologia da ONG como estruturada e escalável.

            🎯 Estratégia: utilizar essa evidência para captação de recursos e fortalecimento institucional.
            """)


    # --------------------------------------------------------------------------
    # Síntese Final
    # --------------------------------------------------------------------------
    with ato5:# Define o conteúdo da quinta aba (Síntese Final)
        st.header("Insights Adicionais e Síntese Final")# Cabeçalho principal da seção
        st.write("""
        A seguir vemos alguns insights adicionais para além dos indicadores propostos, bem como a síntese final passando por todos os pontos que foram abordados nas análises.
        """)# Texto introdutório para contextualizar a síntese

        col9, col10 = st.columns(2) # Cria colunas finais de performance e evolução
        with col9: # Nona coluna

            # ================================================================
            # 1. PREPARAÇÃO DOS DADOS
            # ================================================================
            # Define a lista de indicadores para a análise de correlação final
            indicadores = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
            # Define a ordem hierárquica das pedras para ordenação dos dados
            ordem_pedras = ['QUARTZO', 'AGATA', 'AMETISTA', 'TOPAZIO']

            # Cria uma cópia limpa do DataFrame filtrado, removendo nulos em colunas críticas
            df_ins = df_f.dropna(subset=indicadores + ['PEDRA', 'ANO']).copy()

            # ================================================================
            # 2. PROCESSAMENTO DOS INSIGHTS
            # ================================================================
            # Calcula a correlação de Pearson entre os indicadores e o INDE (Nota Global)
            correl_inde = df_ins[indicadores + ['INDE']].corr()['INDE'].drop(['INDE']).sort_values(ascending=False)

            # Calcula a diferença entre Potencial (IPP) e Desempenho (IDA) para medir o "Gap de Oportunidade"
            df_ins['Gap_Potencial'] = df_ins['IPP'] - df_ins['IDA']
            # Agrupa a média desse gap por Pedra para entender a evolução do aproveitamento
            gap_potencial = df_ins.groupby('PEDRA')['Gap_Potencial'].mean().reindex(ordem_pedras).reset_index()

            # ================================================================
            # 3. CRIAÇÃO DOS GRÁFICOS
            # ================================================================
            # Inicializa a figura Matplotlib para o primeiro gráfico de síntese
            fig, ax = plt.subplots(figsize=(8,6))

            # Plota as correlações para identificar quais indicadores mais sustentam o INDE
            sns.barplot(
                x=correl_inde.index, # Nomes dos indicadores no eixo X
                y=correl_inde.values, # Valores de correlação no eixo Y
                palette='Set2', # Aplica a paleta de cores padronizada
                ax=ax # Vincula ao eixo criado
            ) # Encerra a plotagem de barras

            # Itera sobre as barras para adicionar os rótulos de correlação com 3 casas decimais
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3, fontweight='bold')

            # Configurações estéticas e de rotulagem do gráfico
            ax.set_title('Âncoras Estratégicas do INDE', fontweight='bold') # Define título
            ax.set_xlabel('Indicadores') # Define rótulo X
            ax.set_ylabel('Força de Correlação') # Define rótulo Y
            ax.grid(False) # Remove as grades de fundo
            # Remove o contorno das barras para manter o visual limpo
            for patch in ax.patches:
                patch.set_edgecolor('none')

            plt.tight_layout() # Ajusta o layout para evitar cortes de texto
            st.pyplot(fig) # Renderiza o gráfico final no Streamlit

            st.markdown("""
            ### 🔎 Priorizar o que realmente move o sucesso
            O ranking de correlação revela que nem todos os indicadores possuem o mesmo impacto sobre o INDE.
            """) # Adiciona comentário estratégico abaixo do gráfico

        with col10: # Décima coluna
            # Inicializa a figura Matplotlib para o segundo gráfico de síntese
            fig, ax = plt.subplots(figsize=(8, 6))

            # ---------------- GRÁFICO 2: EVOLUÇÃO EMOCIONAL (IPS) ----------------
            # Analisa se a saúde psicossocial acompanha a evolução das Pedras
            sns.barplot(
                data=df_ins, # Utiliza os dados processados na coluna anterior
                x='PEDRA', # Eixo X com os estágios de pedra
                y='IPS', # Eixo Y com a nota psicossocial
                palette='Set2', # Paleta Set2 para consistência visual
                order=ordem_pedras, # Garante a ordem Quartzo -> Topázio
                ax=ax # Vincula ao eixo
            ) # Encerra plotagem

            # Adiciona rótulos de média no topo de cada barra para leitura precisa
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontweight='bold')

            # Personalização do gráfico de saúde emocional
            ax.set_title('Saúde Psicossocial por Fase', fontweight='bold') # Define título
            ax.set_xlabel('Fase (Pedra)') # Rótulo X
            ax.set_ylabel('Média do IPS') # Rótulo Y
            ax.grid(False) # Desativa grades
            # Remove bordas das barras
            for patch in ax.patches:
                patch.set_edgecolor('none')
            
            plt.tight_layout() # Ajusta layout
            st.pyplot(fig) # Renderiza no Streamlit

            st.markdown("""
            ##### 💎 A Jornada por Pedra Valida a Metodologia

            A progressão consistente dos indicadores ao longo das **Pedras (Quartzo → Ágata → Ametista → Topázio)** comprova que a evolução não é aleatória.

            Ela é estruturada, é replicável e metodológica.
            """)# Adiciona comentário estratégico abaixo do gráfico


        st.divider()

        st.markdown("""
        ### 🏆 A Jornada Completa da Transformação

        Ao percorrer cada etapa desta análise, observamos que a jornada do aluno não é linear — ela é estruturada:

        ##### 📍 1. O Ponto de Partida Não Define o Destino

        Os dados de **Adequação Escolar (IAN)** mostram que muitos alunos iniciam sua trajetória com defasagem significativa.  
        Entretanto, ao cruzarmos com o **Potencial Psicopedagógico (IPP)**, percebemos algo fundamental:

        > A vulnerabilidade inicial não representa ausência de talento — representa ausência de oportunidade.

        A ONG entra exatamente nesse ponto crítico.

        ---

        ##### 📈 2. O Crescimento é Mensurável

        A evolução do **Desempenho Acadêmico (IDA)** ao longo dos anos demonstra que o reforço educacional gera impacto real.

        Mas o dado mais revelador surge quando analisamos o **Engajamento (IEG)**:

        > Alunos que atingem o ponto de virada apresentam níveis significativamente maiores de engajamento.

        Isso indica que o sucesso acadêmico não começa na nota — começa na atitude.

        ---

        ##### 🧠 3. O Pilar Invisível Sustenta a Jornada

        A análise do **Indicador Psicossocial (IPS)** evidencia que estabilidade emocional é pré-condição para aprendizado sustentável.

        Sem segurança emocional, não há progresso consistente.

        Além disso, o alinhamento entre **Autoavaliação (IAA)** e desempenho real mostra que maturidade emocional acompanha evolução acadêmica.

        ---

        ##### 🏆 4. O Que Realmente Move o Sucesso

        Ao analisarmos a correlação com o **INDE**, identificamos que os maiores drivers de sucesso são:

        - Engajamento (IEG)
        - Desempenho Acadêmico (IDA)

        Ou seja:

        > Alta performance é resultado da combinação entre comportamento e competência.

        Quando comparamos a média geral com os alunos Top 20%, essa diferença se torna ainda mais evidente.

        ---

        ##### 🎯 Síntese

        Esta análise demonstra que:

        ✔ A defasagem inicial não determina o futuro  
        ✔ O engajamento é o principal motor de transformação  
        ✔ O apoio psicossocial sustenta o crescimento  
        ✔ Alta performance pode ser desenvolvida  
        ✔ A metodologia da ONG é validada por dados

        ✨ A Passos Mágicos não apenas melhora indicadores, mas também transforma trajetórias de vida de forma estruturada e mensurável.
        """)# Adiciona comentário estratégico final


# ==========================================================================
# Rodapé
# ==========================================================================

st.divider() # Adiciona linha divisória final
st.caption("Projeto do curso de Pós Graduação de Data Analytics da FIAP.") # Crédito acadêmico
st.caption("* PEDE analytics | Ong Passos Mágicos é um nome fictício utilizado para fins acadêmicos.") # Disclaimer