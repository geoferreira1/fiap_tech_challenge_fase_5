# ==========================================================================
# Import de bibliotecas
# ==========================================================================

# Bibliotecas do Sistema e Utilitários
import io            # Manipulação de fluxos de dados (entrada/saída)
import requests      # Realização de requisições HTTP
import time          # Funções relacionadas a tempo e execução
import unicodedata   # Manipulação de caracteres Unicode e normalização de texto

# Processamento e Manipulação de Dados
import joblib        # Persistência de objetos Python e modelos de Machine Learning
import numpy as np   # Suporte a arrays multidimensionais e funções matemáticas
import pandas as pd  # Manipulação e análise de dados estruturados (DataFrames)

# Visualização de Dados
import matplotlib.pyplot as plt # Criação de gráficos estáticos e customização
import seaborn as sns           # Visualização de dados estatísticos baseada em Matplotlib
import streamlit as st          # Framework para criação de dashboards e apps web

# ==========================================================================
# Config página
# ==========================================================================
st.set_page_config(
    page_title="Modelo de Predição sobre o risco de defasagem dos alunos da ong Passos Mágicos", # Define o nome na aba do navegador.
    page_icon="🎯", # Define o emoji que aparece na aba.
    layout="wide" # Define que o conteúdo do site ficará centralizado na tela.
)

# ==========================================================================
# Funções
# ==========================================================================

def setup_options(lista):
    """
    Ordena as opções de respostas em ordem crescente.
    """ 
    def chave_interna(texto):
        if not isinstance(texto, str):
            texto = str(texto) if texto is not None else ""
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ascii').lower()
    
    return sorted(lista, key=chave_interna)


@st.cache_resource # Mantém o modelo na memória após o primeiro carregamento
def load_model(): 
    """
    Carrega o modelo treinado (.joblib) com fallback para GitHub.
    """
    local_path = 'models/modelo_final_gradient_boosting.joblib'
    github_url = "https://raw.githubusercontent.com/geoferreira1/fiap_tech_challenge_fase_5/main/models/modelo_final_gradient_boosting.joblib"

    # 1. Tentativa Local
    try:
        return joblib.load(local_path)
    except (FileNotFoundError, Exception) as e:
        print(f"Aviso: Modelo local não encontrado ou erro no carregamento: {e}")

    # 2. Tentativa Remota (GitHub)
    try:
        response = requests.get(github_url, timeout=15)
        response.raise_for_status() # Levanta erro se o status não for 200
        
        return joblib.load(io.BytesIO(response.content))
    except Exception as e:
        print(f"Erro crítico: Não foi possível carregar o modelo remotamente: {e}")
    
    return None

def config_page(): # Configurar menu lateral
    """
    Desenha os elementos na barra lateral esquerda.
    """
    with st.sidebar: # Inicia o contexto da barra lateral.
        st.markdown("🎯 Desafio") # Título da seção.
        st.info("Modelo preditivo e análise de insights desenvolvivos para a pós graduação de **Data Analytics da FIAP.**") # Quadro informativo.
        st.markdown("---") # Linha horizontal divisória.
        st.markdown("👩🏽‍💻 Aluno(a):")
        st.write("""
        [Geovana dos Santos ferreira](https://www.linkedin.com/in/geovanaferreira/) 
        """)
        st.markdown("---")
        st.markdown("🔗 Repositório:")
        st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
            <style>
                .github-icon {
                    font-size: 35px;
                    color: #24292e; /* Cor padrão do GitHub */
                    text-decoration: none;
                    transition: 0.3s;
                }
                .github-icon:hover {
                    color: #6e5494; /* Cor roxa ao passar o mouse */
                }
            </style>
    
            <a href="https://github.com/geoferreira1/fiap_tech_challenge_fase_5" target="_blank" class="github-icon">
                <i class="fa-brands fa-github"></i>
            </a>
        """, unsafe_allow_html=True)

def classificar_nivel_risco(prob):
    """Classifica o nível de risco baseado na probabilidade"""
    if prob < 0.30:
        return 'Sem Risco', '✅', 'risk-low'
    elif prob < 0.60:
        return 'Atenção', '⚡', 'risk-attention'
    elif prob < 0.85:
        return 'Risco Moderado', '⚠️', 'risk-moderate'
    else:
        return 'Risco Alto', '🚨', 'risk-high'

def get_clinic_input(): # Coletar os dados do questionario
    """
    Coleta os dados do aluno na página e retorna um DataFrame
    """
    # DADOS PESSOAIS
    st.header("1. Informações Pessoais")
    st.markdown("Preencha os campos abaixo para verificar o **nível de defasagem do aluno**. (Obrigatório)" )
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.number_input("Idade", min_value=7, max_value=27, value=15)
        fase = st.number_input("Fase Atual", min_value=0, max_value=9, value=5)
    
    with col2:
        genero = st.selectbox("Gênero", setup_options(["Masculino", "Feminino"]))
        fase_ideal = st.number_input("Fase Ideal", min_value=0, max_value=8, value=5)


    instituicao_opcoes = {
            "Pública": "Pública",
            "Privada": "Privada",
            "Privada - Programa de Apadrinhamento": "Privada - Programa de Apadrinhamento",
            "Privada com Bolsa 100%": "Privada *Parcerias com Bolsa 100%",
               "Privada - Empresa Parceira": "Privada - Pagamento por *Empresa Parceira",
               "Escola JP II": "Escola JP II",
               "Rede Decisão": "Rede Decisão",
               "Bolsista Universitário (Formado)": "Bolsista Universitário *Formado (a)",
               "Concluiu o 3º EM": "Concluiu o 3º EM",
               "Desconhecido": "Desconhecido",
               "Nenhuma das opções acima": "Nenhuma das opções acima"
           }
    instituicao_display = st.selectbox("Instituição de Ensino", list(instituicao_opcoes.keys()))
    instituicao = instituicao_opcoes[instituicao_display]
 
    # Cálculo de defasagem
    defasagem = int(np.ceil(fase - fase_ideal))

    if defasagem < -2:
        base_defasagem = 'Severo'

    elif defasagem >= -2 and defasagem <= 0:
        base_defasagem = 'Moderado'

    else:
        base_defasagem = 'Em Fase'

    st.markdown("---")

    # HISTÓRICO E HÁBITOS
    st.header("2. Indicadores PEDE")
    st.markdown("Preencha os campos abaixo para que seja realizada a previsão. (Obrigatório)")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        ipv_escrito = st.selectbox("IPV (Ponto de Virada)", setup_options(["Sim", "Não"]))
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.0, 0.1)
        ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 6.5, 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 7.0, 0.1)

    
    with col_h2:
        pedra = st.selectbox("Pedra", setup_options(['QUARTZO', 'AGATA', 'AMETISTA', 'TOPAZIO']))
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1)
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1)
        ipp = st.slider("IPP (Potencial Psicopedagógico)", 0.0, 10.0, 7.0, 0.1)
   
    st.markdown("---")

    data = {
        'IDADE': idade,
        'GENERO': genero,
        'IDA': ida,
        'IEG': ieg,
        'IAA': iaa,
        'IPS': ips,
        'PONTO_VIRADA': ipv_escrito,
        'PEDRA': pedra,
        'DEFASAGEM': defasagem,
        'FASE': fase,
        'FASE_IDEAL': fase_ideal,
        'IPP': ipp,
        'IPV': ipv,
        'INSTITUICAO_ENSINO': instituicao
    }
    
    return pd.DataFrame(data, index=[0])


def main(): # Função princial
    # 1. Configura a Barra Lateral
    config_page()

    # 2. Carrega o Modelo
    model = load_model()

    # 3. Página do cálculo predição
    st.caption("🏥 PEDE Analytics | Ong Passos Mágicos <sup>1</sup>", unsafe_allow_html=True)
    st.title("🎯 Modelo de Predição | Risco de Defasagem")
    st.markdown("""
    Preencha o formulário a seguir para que o modelo calcule a probabilidade do risco de defasagem dos alunos.
    """)
    st.markdown("---")

    # 4. Formulário
    input_df = get_clinic_input()

    # 5. Botão e Predição
    st.markdown("###")
    
    if st.button("🎯 Clique aqui para fazer a previsão", type="primary", use_container_width=True):
        if model is not None:
            try:
                    # --- INÍCIO DA BARRA DE PROGRESSO ---
                progress_text = "Analisando dados do aluno. Por favor, aguarde..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)  # Simula o tempo de processamento
                    my_bar.progress(percent_complete + 1, text=progress_text)

                time.sleep(0.5) # Pequena pausa para o usuário ver os 100%
                my_bar.empty()  # Limpa a barra após concluir
                # --- FIM DA BARRA DE PROGRESSO ---

                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)
                prob_risco = probability[0][1]*100

                st.markdown("---")
                st.header("Resultado da Análise")

                if prob_risco >= 51:
                    st.error(f"🚨 **ALTO RISCO DE DEFASAGEM**")
                    st.metric(label="A probabilidade do aluno ficar defasado futuramente é de:", value=f"{probability[0][1] * 100:.1f}%")
                    st.warning("💭 **Recomendação:** Aluno necessita de plano de recuperação imediato e reunião com responsáveis.")

                elif prob_risco == 50:
                    st.warning(f"⚠️ **MÉDIO RISCO**")
                    st.metric(label="A probabilidade do aluno ficar defasado futuramente é de:", value=f"{probability[0][1] * 100:.1f}%")
                    st.info("💭 **Recomendação:** Sugere-se monitoramento semanal e oferta de aulas de reforço em contraturno.")

                else:
                    st.success(f"🥳 **BAIXO RISCO DE DEFASAGEM**")
                    st.metric(label="A probabilidade do aluno ficar defasado futuramente é de:", value=f"{probability[0][1] * 100:.1f}%")
                    st.info("💭 **Recomendação:** O aluno demonstra forte engajamento e resultados sólidos. Manter acompanhamento regular.")

            except Exception as e:
                st.error(f"Ocorreu um erro técnico ao realizar a predição: {e}")
        else:
            st.error("📣 O modelo de predição retornou um erro, por gentileza verifique se os dados foram selecionados corretamente.")

    st.markdown("---")

    # Adiciona o crédito final da aplicação centralizado no rodapé
    st.caption("Projeto do curso de Pós Graduação de Data Analytics da FIAP.")
    st.caption("* PEDE analytics | Ong Passos Mágicos é um nome fictício utilizado para fins estritamente acadêmicos.")
if __name__ == "__main__":
    main()