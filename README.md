# 🪄 Passos Mágicos | Impacto Educacional - Datathon - Tech Challenge Fase 5 (FIAP)

> Este projeto é referente ao **Datathon - Tech Challenge da 5ª fase da Pós-Tech FIAP (Data Analytics)**. 

O objetivo é analisar o impacto social e educacional da **Associação Passos Mágicos**, utilizando **análise de dados** para medir a evolução dos alunos e fornecer uma visão estratégica sobre o desenvolvimento acadêmico e psicossocial. Além da prática de análise de dados e storytelling,
esse desafio também traz o desafio do desenvolvimento de um modelo preditivo.

---

## 🎯 O Desafio

O projeto visa transformar os dados históricos da associação em insights acionáveis para coordenadores, focando em:

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

O modelo **GradientBoostingClassifier** foi selecionado visando o foco na assertividade diagnóstica:

| Métrica | Valor | Importância para o Negócio |
| :--- | :--- | :--- |
| **Acurácia** | ~98% | Assertividade geral do sistema. |
| **Recall (Sensibilidade)** | 97% | Garante que alunos em risco real sejam identificados. |
| **F1-Score** | 98% | Equilíbrio ideal entre precisão e sensibilidade. |

---

## 📊 Insights de Negócio (Visão Dashboard)

🏆 A Jornada Completa da Transformação
Ao percorrer cada etapa desta análise, observamos que a jornada do aluno não é linear — ela é estruturada:

📍 1. O Ponto de Partida Não Define o Destino
Os dados de Adequação Escolar (IAN) mostram que muitos alunos iniciam sua trajetória com defasagem significativa.
Entretanto, ao cruzarmos com o Potencial Psicopedagógico (IPP), percebemos algo fundamental:

> A vulnerabilidade inicial não representa ausência de talento — representa ausência de oportunidade.

A ONG entra exatamente nesse ponto crítico.

📈 2. O Crescimento é Mensurável
A evolução do Desempenho Acadêmico (IDA) ao longo dos anos demonstra que o reforço educacional gera impacto real.

Mas o dado mais revelador surge quando analisamos o Engajamento (IEG):

> Alunos que atingem o ponto de virada apresentam níveis significativamente maiores de engajamento.

Isso indica que o sucesso acadêmico não começa na nota — começa na atitude.

🧠 3. O Pilar Invisível Sustenta a Jornada
A análise do Indicador Psicossocial (IPS) evidencia que estabilidade emocional é pré-condição para aprendizado sustentável.

Sem segurança emocional, não há progresso consistente.

Além disso, o alinhamento entre Autoavaliação (IAA) e desempenho real mostra que maturidade emocional acompanha evolução acadêmica.

🏆 4. O Que Realmente Move o Sucesso
Ao analisarmos a correlação com o INDE, identificamos que os maiores drivers de sucesso são:

- Engajamento (IEG)
- Desempenho Acadêmico (IDA)
  
Ou seja:

> Alta performance é resultado da combinação entre comportamento e competência.

Quando comparamos a média geral com os alunos Top 20%, essa diferença se torna ainda mais evidente.

🎯 Síntese
Esta análise demonstra que:

✔ A defasagem inicial não determina o futuro

✔ O engajamento é o principal motor de transformação

✔ O apoio psicossocial sustenta o crescimento

✔ Alta performance pode ser desenvolvida

✔ A metodologia da ONG é validada por dados

✨ A Passos Mágicos não apenas melhora indicadores, mas também transforma trajetórias de vida de forma estruturada e mensurável.
---

### 🖥️ Streamlit
O dashboard interativo com as análises e indicadores pode ser acessado através do link abaixo:

🪄 [Painel de Impacto Passos Mágicos](https://ong-pmagicos-fiaptechchallengefase5-datathon.streamlit.app/)

---

## 📂 Estrutura do Repositório

```
├── data_raw/
│   ├── base_passos_magicos.xls                # Base bruta original
│   └── desvendando_passos.pdf                 # Referência técnica das variáveis
│   └── desvendando_passos.pdf                 # Referência técnica das variáveis
│   └── Dicionário Dados Datathon.pdf          # Referência técnica das variáveis
│   └── Links adicionais da passos.docx        # Referência técnica das variáveis
│   └── PEDE_ Pontos importantes.docx          # Referência técnica das variáveis
│   └── Relatório PEDE2020.pdf                 # Referência técnica das variáveis
│   └── Relatório PEDE2021.pdf                 # Referência técnica das variáveis
│   └── Relatório PEDE2022.pdf                 # Referência técnica das variáveis
├── data_processed/
│   └── df_unificado.csv                       # Base tratada após ETL
├── models/
│   └── modelo_final_gradient_boosting.joblib  # Pipeline de ML pronto para produção
├── notebook/
│   └── fiap_tech_challenge_fase_5.ipynb       # Documentação do experimento (Notebook)
├── streamlit/
│   ├── pages/
│   │   └── Dashboard.py                       # Dashboard fo projeto / Visão Analítica (Streamlit)
│   └── Modelo.py                              # Interface de Predição (Streamlit)
├── requirements.txt                           # Dependências do ecossistema
└── README.md                                  # Documentação do projeto
```

---

## 👨‍💻 Autor: 
  - [Geovane Ferreira](https://www.linkedin.com/in/geovaneferreira/)
