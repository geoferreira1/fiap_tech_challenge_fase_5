# ==========================================================================
# Import de bibliotecas
# ==========================================================================

# Bibliotecas do Sistema e Utilitários
import io            # Gerencia fluxos de dados (entrada/saída) em memória binária
import requests      # Permite realizar requisições HTTP para buscar o modelo no GitHub
import time          # Fornece funções de controle de tempo para pausas e animações
import unicodedata   # Utilizado para normalizar textos e remover acentos de strings

# Processamento e Manipulação de Dados
import joblib        # Carrega e salva objetos serializados, como modelos de IA (.joblib)
import numpy as np   # Biblioteca para cálculos matemáticos e operações com arrays
import pandas as pd  # Ferramenta principal para criação e manipulação de DataFrames

# Visualização de Dados
import matplotlib.pyplot as plt # Interface base para geração de gráficos estáticos
import seaborn as sns           # Biblioteca de visualização estatística refinada
import streamlit as st          # Framework para converter o script em aplicação web interativa

# ==========================================================================
# Config página
# ==========================================================================
st.set_page_config( # Define as configurações globais da interface do usuário
    page_title="Modelo de Predição sobre o risco de defasagem dos alunos da ong Passos Mágicos", # Título da aba
    page_icon="🎯", # Emoji ícone da aba do navegador
    layout="wide" # Configura o uso de toda a largura da tela disponível
) # Encerra a configuração da página

# ==========================================================================
# Funções de Suporte
# ==========================================================================

def setup_options(lista): # Define função para padronizar e ordenar listas de seleção
    """Ordena as opções de respostas em ordem crescente.""" 
    def chave_interna(texto): # Função interna para critério de ordenação sem acentos
        if not isinstance(texto, str): # Verifica se o dado não é uma string
            texto = str(texto) if texto is not None else "" # Converte nulo para string vazia
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ascii').lower() # Normaliza texto
    
    return sorted(lista, key=chave_interna) # Retorna a lista devidamente ordenada


@st.cache_resource # Mantém o modelo carregado na memória para evitar reprocessamento constante
def load_model(): # Define função para carregamento do arquivo do modelo
    """Carrega o modelo treinado (.joblib) com fallback para GitHub."""
    local_path = 'models/modelo_final_gradient_boosting.joblib' # Define caminho do arquivo no ambiente local
    github_url = "https://raw.githubusercontent.com/geoferreira1/fiap_tech_challenge_fase_5/main/models/modelo_final_gradient_boosting.joblib" # URL do repositório remoto

    # 1. Tentativa de carregamento a partir do diretório local
    try: # Inicia bloco de captura de erros
        return joblib.load(local_path) # Tenta carregar o modelo localmente
    except (FileNotFoundError, Exception) as e: # Captura erro se o arquivo não existir
        print(f"Aviso: Modelo local não encontrado ou erro no carregamento: {e}") # Exibe aviso no console

    # 2. Tentativa Remota (GitHub) como alternativa de segurança
    try: # Inicia bloco de tentativa remota
        response = requests.get(github_url, timeout=15) # Realiza o download do modelo via HTTP
        response.raise_for_status() # Lança erro se a requisição não for bem-sucedida
        return joblib.load(io.BytesIO(response.content)) # Carrega o modelo a partir dos bytes baixados
    except Exception as e: # Captura qualquer falha no processo remoto
        print(f"Erro crítico: Não foi possível carregar o modelo remotamente: {e}") # Exibe erro fatal no console
    
    return None # Retorna nulo caso todas as tentativas falhem

def config_page(): # Define função para construir a barra lateral (sidebar)
    """Desenha os elementos na barra lateral esquerda."""
    with st.sidebar: # Inicia o contexto da barra lateral do Streamlit
        st.title("🎯 Desafio") # Adiciona título Markdown na sidebar
        st.info("Modelo preditivo e análise de insights desenvolvivos para a pós graduação de **Data Analytics da FIAP.**") # Exibe caixa azul de info
        st.markdown("---") # Adiciona linha horizontal de separação
        st.title("👩🏽‍💻 Aluno(a):") # Rótulo para identificação
        st.write("[Geovana dos Santos ferreira](https://www.linkedin.com/in/geovanaferreira/)") # Link para o LinkedIn
        st.markdown("---") # Adiciona linha separadora
        st.title("🔗 Repositório:") # Rótulo para link do GitHub
        st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
            <style>.github-icon { font-size: 35px; color: #24292e; text-decoration: none; transition: 0.3s; }
            .github-icon:hover { color: #6e5494; }</style>
            <a href="https://github.com/geoferreira1/fiap_tech_challenge_fase_5" target="_blank" class="github-icon">
            <i class="fa-brands fa-github"></i></a>""", unsafe_allow_html=True) # Insere ícone do GitHub via HTML/CSS

def classificar_nivel_risco(prob): # Função auxiliar para rotular o risco (lógica de apoio)
    """Classifica o nível de risco baseado na probabilidade"""
    if prob < 0.30: return 'Sem Risco', '✅', 'risk-low' # Risco baixo
    elif prob < 0.60: return 'Atenção', '⚡', 'risk-attention' # Atenção
    elif prob < 0.85: return 'Risco Moderado', '⚠️', 'risk-moderate' # Moderado
    else: return 'Risco Alto', '🚨', 'risk-high' # Risco alto

# ==========================================================================
# Coleta de Dados (Formulário)
# ==========================================================================

def get_clinic_input(): # Define função para construir o formulário de entrada de dados
    """Coleta os dados do aluno na página e retorna um DataFrame"""
    st.header("1. Informações Pessoais") # Título da primeira seção do formulário
    st.markdown("Preencha os campos abaixo para verificar o **nível de defasagem do aluno**. (Obrigatório)") # Texto instrutivo
    
    col1, col2 = st.columns(2) # Divide o formulário em duas colunas verticais
    
    with col1: # Elementos da primeira coluna
        idade = st.number_input("Idade", min_value=7, max_value=27, value=15) # Coleta idade do aluno
        fase = st.number_input("Fase Atual", min_value=0, max_value=9, value=5) # Coleta fase pedagógica atual
    
    with col2: # Elementos da segunda coluna
        genero = st.selectbox("Gênero", setup_options(["Masculino", "Feminino"])) # Coleta gênero
        fase_ideal = st.number_input("Fase Ideal", min_value=0, max_value=8, value=5) # Coleta fase ideal teórica

    instituicao_opcoes = { # Dicionário para mapear nomes amigáveis para valores técnicos do dataset
            "Pública": "Pública", "Privada": "Privada", "Privada - Programa de Apadrinhamento": "Privada - Programa de Apadrinhamento",
            "Privada com Bolsa 100%": "Privada *Parcerias com Bolsa 100%", "Privada - Empresa Parceira": "Privada - Pagamento por *Empresa Parceira",
            "Escola JP II": "Escola JP II", "Rede Decisão": "Rede Decisão", "Bolsista Universitário (Formado)": "Bolsista Universitário *Formado (a)",
            "Concluiu o 3º EM": "Concluiu o 3º EM", "Desconhecido": "Desconhecido", "Nenhuma das opções acima": "Nenhuma das opções acima"
    } # Encerra mapeamento de instituições
    instituicao_display = st.selectbox("Instituição de Ensino", list(instituicao_opcoes.keys())) # Widget de seleção
    instituicao = instituicao_opcoes[instituicao_display] # Armazena o valor técnico selecionado
 
    defasagem = int(np.ceil(fase - fase_ideal)) # Calcula a defasagem numérica (arredondada para cima)

    if defasagem < -2: base_defasagem = 'Severo' # Classifica nível de defasagem severo
    elif defasagem >= -2 and defasagem <= 0: base_defasagem = 'Moderado' # Moderado
    else: base_defasagem = 'Em Fase' # Aluno no nível correto

    st.markdown("---") # Linha divisória entre seções
    st.header("2. Indicadores PEDE") # Título da segunda seção do formulário
    st.markdown("Preencha os campos abaixo para que seja realizada a previsão. (Obrigatório)") # Texto instrutivo
    
    col_h1, col_h2 = st.columns(2) # Cria colunas para os sliders de indicadores
    
    with col_h1: # Indicadores qualitativos e acadêmicos
        ipv_escrito = st.selectbox("IPV (Ponto de Virada)", setup_options(["Sim", "Não"])) # Status do Ponto de Virada
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.0, 0.1) # Nota numérica do IPV
        ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 6.5, 0.1) # Nota do indicador acadêmico
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 7.0, 0.1) # Nota do indicador de engajamento
    
    with col_h2: # Indicadores psicopedagógicos e sociais
        pedra = st.selectbox("Pedra", setup_options(['QUARTZO', 'AGATA', 'AMETISTA', 'TOPAZIO'])) # Classificação de pedra
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1) # Nota do indicador social
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1) # Nota da autoavaliação
        ipp = st.slider("IPP (Potencial Psicopedagógico)", 0.0, 10.0, 7.0, 0.1) # Nota do potencial pedagógico
   
    st.markdown("---") # Linha divisória final do formulário

    data = { # Cria dicionário com os dados coletados respeitando as chaves do modelo treinado
        'IDADE': idade, 'GENERO': genero, 'IDA': ida, 'IEG': ieg, 'IAA': iaa, 'IPS': ips,
        'PONTO_VIRADA': ipv_escrito, 'PEDRA': pedra, 'DEFASAGEM': defasagem, 'FASE': fase,
        'FASE_IDEAL': fase_ideal, 'IPP': ipp, 'IPV': ipv, 'INSTITUICAO_ENSINO': instituicao
    } # Encerra estruturação do dicionário de dados
    
    return pd.DataFrame(data, index=[0]) # Retorna os dados convertidos em um DataFrame do Pandas

# ==========================================================================
# 6. Execução Principal (Main)
# ==========================================================================

def main(): # Define função principal que coordena o app
    config_page() # Inicializa a barra lateral
    model = load_model() # Tenta carregar o modelo de Machine Learning

    st.caption("✨ PEDE Analytics | Ong Passos Mágicos <sup>1</sup>", unsafe_allow_html=True) # Exibe legenda superior
    st.title("🎯 Modelo de Predição | Risco de Defasagem") # Exibe título principal da página
    st.markdown("Preencha o formulário a seguir para que o modelo calcule a probabilidade do risco de defasagem dos alunos.") # Texto
    st.markdown("---") # Divisor

    input_df = get_clinic_input() # Chama a função de formulário e armazena os dados do usuário
    st.markdown("###") # Espaçamento vertical

    # Botão para disparar o cálculo da inteligência artificial
    if st.button("🎯 Clique aqui para fazer a previsão", type="primary", use_container_width=True): # Inicia se clicado
        if model is not None: # Verifica se o modelo está pronto para uso
            try: # Bloco de execução da predição
                progress_text = "Analisando dados do aluno. Por favor, aguarde..." # Texto da barra de progresso
                my_bar = st.progress(0, text=progress_text) # Inicializa barra de progresso em 0%
                for percent_complete in range(100): # Loop para simular o tempo de processamento
                    time.sleep(0.01) # Pausa curta para animação
                    my_bar.progress(percent_complete + 1, text=progress_text) # Atualiza progresso da barra
                time.sleep(0.5) # Pausa final
                my_bar.empty() # Remove a barra da tela

                prediction = model.predict(input_df) # Realiza a classificação (Risco vs Não Risco)
                probability = model.predict_proba(input_df) # Extrai as probabilidades de cada classe
                prob_risco = probability[0][1]*100 # Converte probabilidade da classe de risco para porcentagem

                st.markdown("---") # Divisor
                st.header("Resultado da Análise") # Título da seção de resultados

                # Lógica de diagnóstico baseada no resultado da probabilidade
                if prob_risco >= 51: # Regra de Alto Risco
                    st.error(f"🚨 **ALTO RISCO DE DEFASAGEM**") # Mensagem de erro (vermelha)
                    st.metric(label="A probabilidade do aluno ficar defasado futuramente é de:", value=f"{prob_risco:.1f}%") # Exibe métrica
                    st.warning("💭 **Recomendação:** Aluno necessita de plano de recuperação imediato e reunião com responsáveis.") # Aviso

                elif prob_risco == 50: # Regra de Médio Risco
                    st.warning(f"⚠️ **MÉDIO RISCO**") # Mensagem de atenção (amarela)
                    st.metric(label="A probabilidade do aluno ficar defasado futuramente é de:", value=f"{prob_risco:.1f}%") # Métrica
                    st.info("💭 **Recomendação:** Sugere-se monitoramento semanal e oferta de aulas de reforço em contraturno.") # Info

                else: # Regra de Baixo Risco
                    st.success(f"🥳 **BAIXO RISCO DE DEFASAGEM**") # Mensagem de sucesso (verde)
                    st.metric(label="A probabilidade do aluno ficar defasado futuramente é de:", value=f"{prob_risco:.1f}%") # Métrica
                    st.info("💭 **Recomendação:** O aluno demonstra forte engajamento e resultados sólidos. Manter acompanhamento regular.") # Info

            except Exception as e: # Captura erros durante o cálculo
                st.error(f"Ocorreu um erro técnico ao realizar a predição: {e}") # Exibe erro técnico
        else: # Se o modelo falhou no carregamento
            st.error("📣 O modelo de predição não foi carregado corretamente.") # Alerta de erro de carregamento

    st.markdown("---") # Divisor final
    st.caption("Projeto do curso de Pós Graduação de Data Analytics da FIAP.") # Crédito acadêmico
    st.caption("* PEDE analytics | Ong Passos Mágicos é um nome fictício utilizado para fins acadêmicos.") # Disclaimer

if __name__ == "__main__": # Ponto de entrada padrão do script Python
    main() # Executa a função principal