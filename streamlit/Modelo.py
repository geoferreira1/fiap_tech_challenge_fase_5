# ==========================================================================
# Importe de bibliotecas
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
    page_title="Modelo de Predição sobre o risto de defasagem dos alunos da ong Passos Mágicos", # Define o nome na aba do navegador.
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


def get_clinic_input(): # Coletar os dados do questionario
    """
    Coleta os dados do aluno na página e retorna um DataFrame
    """
    # DADOS PESSOAIS
    st.header("1. Informações Pessoais")
    st.markdown("Preencha os campos abaixo para verificar o **nível de defasagem do aluno**.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.number_input("Idade", min_value=7, max_value=27, value=15)
        fase = st.number_input("Fase", min_value=0, max_value=9, value=5)
    
    with col2:
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)
        sexo = st.selectbox("Gênero", setup_options(["Masculino", "Feminino"]))

    # Normalização da informação de gênero
    GENERO = 1 if sexo == "Feminino" else 0

    # Cálculo de IMC
    imc = int(np.ceil(peso / (altura ** 2)))

    if imc < 18.5:
        base_imc = 'Abaixo do peso'

    elif imc >= 18.5 and imc <= 24.9:
        base_imc = 'Peso normal'

    elif imc >= 25.0 and imc <= 29.9:
        base_imc = 'Sobrepeso'

    elif imc >= 30.0 and imc <= 34.9:
        base_imc = 'Obesidade grau I'

    elif imc >= 35.0 and imc <= 39.9:
        base_imc = 'Obesidade grau II'

    else:
        base_imc = 'Obesidade grau III'

    st.info(f"🎛️ **IMC do paciênte é de:** {imc} kg/m² ({base_imc})")
    st.markdown("---")

    # HISTÓRICO E HÁBITOS
    st.header("2.  Estilo de vida e hábitos alimentares")
    st.markdown("Preencha os campos abaixo para que seja realizada a previsão.")
    
    option_map = {
        'Sim': "Sim",
        'Não': "Não"
    }

    mapa_refeicoes = {
        '1': 'uma_refeicao_por_dia',
        '2': 'duas_refeicoes_por_dia',
        '3': 'tres_refeicoes_por_dia',
        '4+': 'maior_que_tres_refeicoes_por_dia'
    }

    mapa_vegetais = {
        'Raramente': 'raramente', 
        'Às vezes': 'as_vezes', 
        'Sempre': 'sempre'
    }

    mapa_agua = {
        '< 1 Litro': 'baixa', 
        '1-2 Litros': 'moderada', 
        '> 2 Litros': 'alta'
    }

    mapa_entre_refeicoes = {
        'Nunca': 'nunca', 
        'Às vezes': 'baixa', 
        'Frequentemente': 'moderada', 
        'Sempre': 'alta'
    }

    mapa_alcool = {
        'Nunca': 'nunca', 
        'Às vezes': 'baixa', 
        'Frequentemente': 'moderada', 
        'Sempre': 'alta'
    }

    mapa_ativdade = {
        'Sedentário': 'sedentario', 
        'Baixa': 'baixa', 
        'Moderada': 'moderada', 
        'Alta': 'alta'
    }
    mapa_internet = {
        'Baixa': 'baixa', 
        'Moderada': 'moderada', 
        'Alta': 'alta'
    }
    mapa_transporte = {
        'Transporte Público': 'transporte_publico', 
        'Caminhada': 'caminhada', 
        'Carro': 'carro', 
        'Bicicleta': 'bicicleta', 
        'Moto': 'moto'
    }

    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        
        historico_familiar = st.pills(
        "Possui histórico familiar de sobrepeso?",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default='Sim' 
        )
        
        fuma = st.pills(
        "Você é fumante ou ex-fumante?",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default='Sim' 
        )
        
        consumo_alimentos_altamente_caloricos = st.pills(
        "Consome alimentos calóricos frequentemente?",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default='Sim' 
        )
    
        
        monitoramento_calorias = st.pills(
        "Costuma contabilizar as calorias ingeridas?",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default='Não' 
        )
        
        refeicao_selecionada = st.pills(
        "Quantas refeições principais faz por dia?",
        options=list(mapa_refeicoes.keys()), 
        selection_mode="single",
        default='1'
        )

        vegetal_selecionada = st.pills(
        "Costuma comer vegetais?",
        options=list(mapa_vegetais.keys()), 
        selection_mode="single",
        default='Raramente'
        )
    
    with col_h2:
        
        agua_selecionada = st.pills(
        "Consumo diário de água?",
        options=list(mapa_agua.keys()), 
        selection_mode="single",
        default='< 1 Litro'
        )
        
        alimentacao_entre_refeicoes_selecionada = st.pills(
        "Costuma comer entre as refeições?",
        options=list(mapa_entre_refeicoes.keys()), 
        selection_mode="single",
        default='Nunca' 
        )
    
        alcool_selecionada = st.pills(
        "Costuma beber bebidas alcoólicas?",
        options=list(mapa_alcool.keys()), 
        selection_mode="single",
        default='Nunca' 
        )

        atividade_fisica_selecionada = st.pills(
        "Pratica atividade física?",
        options=list(mapa_ativdade.keys()), 
        selection_mode="single",
        default='Sedentário'
        )

        tecnologia_selecionada = st.pills(
        "Tempo diário em dispositivos eletrônicos?",
        options=list(mapa_internet.keys()), 
        selection_mode="single",
        default='Baixa'
        )

        meio_de_transporte_selecionada = st.pills(
        "Meio de transporte principal?",
        options=list(mapa_transporte.keys()), 
        selection_mode="single",
        default='Transporte Público'
        )
    
    # Normalização das respostas
    historico_familiar = 1 if historico_familiar == "Sim" else 0
    fuma = 1 if fuma == "Sim" else 0
    consumo_alimentos_altamente_caloricos = 1 if consumo_alimentos_altamente_caloricos == "Sim" else 0
    monitoramento_calorias = 1 if monitoramento_calorias == "Sim" else 0
    consumo_refeicoes_principais = mapa_refeicoes[refeicao_selecionada]
    consumo_vegetais = mapa_vegetais[vegetal_selecionada]
    consumo_agua = mapa_agua[agua_selecionada]
    consumo_lanches_entre_refeicoes = mapa_entre_refeicoes[alimentacao_entre_refeicoes_selecionada]
    consumo_alcool = mapa_alcool[alcool_selecionada]
    frequencia_atividade_fisica = mapa_ativdade[atividade_fisica_selecionada]
    tempo_uso_tecnologia = mapa_internet[tecnologia_selecionada]
    meio_de_transporte = mapa_transporte[meio_de_transporte_selecionada]

    
    st.markdown("---")

    data = {
        'idade': idade,
        'genero': genero,
        'consumo_refeicoes_principais': consumo_refeicoes_principais,
        'consumo_vegetais': consumo_vegetais,
        'consumo_agua': consumo_agua,
        'frequencia_atividade_fisica': frequencia_atividade_fisica,
        'tempo_uso_tecnologia': tempo_uso_tecnologia,
        'fuma': fuma,
        'consumo_alimentos_altamente_caloricos': consumo_alimentos_altamente_caloricos,
        'monitoramento_calorias': monitoramento_calorias,
        'historico_familiar': historico_familiar,
        'consumo_lanches_entre_refeicoes': consumo_lanches_entre_refeicoes,
        'consumo_alcool': consumo_alcool,
        'meio_de_transporte': meio_de_transporte,
        'imc': imc
    }
    
    return pd.DataFrame(data, index=[0])


def main(): # Função princial
    # 1. Configura a Barra Lateral
    config_page()

    # 2. Carrega o Modelo
    model = load_model()

    # 3. Página do cálculo predição
    st.caption("🏥 MedAnalytics | Gestão de Saúde <sup>1</sup>", unsafe_allow_html=True)
    st.title("🎯 Modelo de Predição | Risco de Obesidade")
    st.markdown("""
    Preencha o formulário a seguir para que o modelo calcule a probabilidade do risco de obesidade do paciente.
    """)
    st.markdown("---")

    # 4. Formulário
    input_df = get_clinic_input()

    # 5. Botão e Predição
    st.markdown("###")
    
    if st.button("🎯 Clique aqui para saber a previsão", type="primary", use_container_width=True):
        if model is not None:
            try:
                    # --- INÍCIO DA BARRA DE PROGRESSO ---
                progress_text = "Analisando dados do paciente. Por favor, aguarde..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)  # Simula o tempo de processamento
                    my_bar.progress(percent_complete + 1, text=progress_text)

                time.sleep(0.5) # Pequena pausa para o usuário ver os 100%
                my_bar.empty()  # Limpa a barra após concluir
                # --- FIM DA BARRA DE PROGRESSO ---

                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)

                st.markdown("---")
                st.header("Resultado da Análise")

                if prediction[0] == 1:
                    st.error("🚨 **ALTO RISCO DE OBESIDADE**")
                    st.metric(label="A probabilidade do paciente se tornar obeso futuramente é de:", value=f"{probability[0][1] * 100:.1f}%")
                    st.warning("💭 **Recomendação:** Sugere-se encaminhamento para orientação médica e nutricional além de realizar ajustes no estilo de vida.")
                else:
                    st.success("🥳 **BAIXO RISCO DE OBESIDADE**")
                    st.metric(label="Probabilidade de Risco", value=f"{probability[0][1] * 100:.1f}%")
                    st.info("💭 **Recomendação:** Continar mantendo hábitos saudáveis e realizar acompanhamento médico periódico.")
            
            except Exception as e:
                st.error(f"Ocorreu um erro técnico ao realizar a predição: {e}")
        else:
            st.error("📣 O modelo de predição retornou um erro, por gentileza verifique se os dados foram selecionados corretamente.")

    st.markdown("---")

    # Adiciona o crédito final da aplicação centralizado no rodapé
    st.caption("Projeto do curso de Pós Graduação de Data Analytics da FIAP.")
    st.caption("* MedAnalytics | Gestão de Saúde é um nome fictício utilizado para fins estritamente acadêmicos.")
if __name__ == "__main__":
    main()