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