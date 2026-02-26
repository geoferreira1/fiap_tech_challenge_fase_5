# 🪄 Passos Mágicos | Impacto Educacional - Datathon - Tech Challenge Fase 5 (FIAP)

> Este projeto é referente ao **Datathon - Tech Challenge da 5ª fase da Pós-Tech FIAP (Data Analytics)**. 

O objetivo é analisar o impacto social e educacional da **Associação Passos Mágicos**, utilizando **análise de dados** para medir a evolução dos alunos e fornecer uma visão estratégica sobre o desenvolvimento acadêmico e psicossocial. Além da prática de análise de dados e storytelling,
esse desafio também traz o desafio do desenvolvimento de um modelo preditivo.

---

## 🎯 O Desafio

O projeto visa transformar os dados históricos da associação em insights acionáveis para coordenadores e investidores, focando em:

1.  **Visão Analítica:** Um dashboard interativo para monitorar a evolução dos alunos (PED, IAN, IDA, IEG) ao longo dos anos.
2.  **Métricas de Desempenho:** Identificação de alunos que precisam de maior suporte e análise de correlação entre o engajamento (IEG) e o desempenho acadêmico (IDA).
3.  **Construção de um modelo preditivo:** Identificação da probabilidade do aluno ou aluna entrar em risco de defasagem.

---

## 🏗️ Arquitetura do Projeto
<p align="center">
  <img src="https://github.com/user-attachments/assets/79c985a1-ff10-4afd-be47-6bf70d0dd01b">
</p>

### Pipeline de Desenvolvimento
Todas as etapas do projeto foram disponibilizadas no arquivo `fiap_tech_challenge_fase_5.ipynb`, abrangendo:

* **ETL & Data Cleaning:** Tratamento de ruídos em variáveis categóricas e numéricas, além tradução completa dos labels para Português (PT-BR).
* **Feature Engineering:** Criação das features de **Defasagem** (Target binário).
* **Modelagem:** Testes comparativos entre os modelos Regressão Logística, GradientBoostingClassifier e Random Forest de Machine Learning (ML).
* **Seleção de Modelo:** O **GradientBoostingClassifier** foi o escolhido devido à sua superioridade no *Recall* e *F1-Score*, fundamentais para evitar falsos negativos na área da saúde.

---

## 📈 Performance do Modelo

O modelo **GradientBoostingClassifier** foi selecionado visando o foco na segurança do paciente e assertividade diagnóstica:

| Métrica | Valor | Importância para o Negócio |
| :--- | :--- | :--- |
| **Acurácia** | ~98% | Assertividade geral do sistema. |
| **Recall (Sensibilidade)** | 97% | Garante que pacientes em risco real sejam identificados. |
| **F1-Score** | 98% | Equilíbrio ideal entre precisão e sensibilidade. |

---

## 📊 Insights de Negócio (Visão Dashboard)

Extraímos padrões fundamentais para a estratégia da associação, como:
* **Curva de Aprendizado:** Alunos com mais tempo de projeto tendem a apresentar estabilidade no crescimento do IDA.
* **Fator Engajamento:** O IEG demonstrou ser um dos principais preditores de sucesso na transição entre níveis de ensino.
* **Análise de Unidades:** Identificação de unidades com maior necessidade de reforço pedagógico específico através da média do IPP.

---

### 🖥️ Streamlit
O dashboard interativo com as análises e indicadores pode ser acessado através do link abaixo:

🪄 [Painel de Impacto Passos Mágicos](https://ong-pmagicos-fiaptechchallengefase5-datathon.streamlit.app/)

---

## 📂 Estrutura do Repositório

---

## 👨‍💻 Autor: 
  - [Geovane Ferreira](https://www.linkedin.com/in/geovaneferreira/)
