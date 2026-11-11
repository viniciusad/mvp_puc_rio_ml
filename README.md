# MVP — Sistema Inteligente de Recomendação de Notebooks e Tendências por Perfil de Uso (Sistemas Inteligentes, Qualidade e Segurança)

## 1. Título e Descrição do Projeto

**MVP — Sistema Inteligente de Recomendação de Notebooks e Tendências de Mercado por Perfil de Uso (Engenharia de Sistemas de Software Inteligentes, Qualidade de Software e Desenvolvimento Seguro)**

Este projeto implementa um **Produto Mínimo Viável (MVP)** que combina **Classificação Supervisionada Multiclasse** e **Filtragem Baseada em Conteúdo (Similaridade de Cosseno / KNN)** para recomendar as melhores configurações de notebooks do mercado brasileiro e apresentar tendências do segmento com base no perfil do usuário (`0 = Uso Básico / Estudos`, `1 = Trabalho / Programação / Multitarefa`, `2 = Jogos / Edição de Vídeo / 3D`). A solução contempla todo o ciclo exigido pela pós-graduação da PUC-Rio:
- **Pipeline de Ciência de Dados e Machine Learning (`notebook/notebook_MVP.ipynb`)**: Carga automatizada via URL Raw do GitHub (100% reprodutível no Google Colab sem API Keys), demonstração de limpeza de anúncios da API do Mercado Livre via expressões regulares (`regex`), análise exploratória (EDA), separação *Holdout* ($80\%/20\%$), prevenção de *Data Leakage* via `Pipeline`, comparação entre dados originais, padronizados (`StandardScaler`) e normalizados (`MinMaxScaler`) nos 4 algoritmos obrigatórios (**KNN**, **Árvore de Classificação C4.5/CART**, **Naive Bayes** e **SVM**), otimização com `GridSearchCV` + validação cruzada estratificada ($K=5$), motor de recomendação `NearestNeighbors(metric='cosine')` e simulador interativo.
- **Aplicação Full Stack Modular (`backend/` e `frontend/`)**: API REST em Flask estruturada segundo os princípios **SOLID** e **DRY** (`ModelService` Singleton desacoplado das rotas HTTP) integrada a um Front-end guiado em HTML5, CSS3 e JavaScript ES6+ que exibe a classificação do perfil, painel de tendências de mercado e os Top-5 notebooks mais compatíveis.
- **Quality Gate Duplo Automatizado (`tests/test_model.py`)**: Suíte **PyTest** que valida tanto as métricas de classificação ($\text{Acurácia} \ge 0{,}80$ e $\text{F1-Score Macro} \ge 0{,}80$) quanto a coerência de negócio das recomendações KNN (garantindo que 100% do Top-5 para o perfil *Jogos / Edição 3D* possua GPU dedicada, pelo menos 16 GB de RAM e compatibilidade média $\ge 0{,}80$).

---

## 2. Arquitetura e Tecnologias Utilizadas

| Camada / Componente | Tecnologia / Biblioteca | Versão Recomendada | Papel na Arquitetura |
| :--- | :--- | :---: | :--- |
| **Linguagem Base** | Python | 3.10+ | Linguagem principal de Ciência de Dados, Back-end e Testes |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Regex (`re`) | $\ge 1.4$ / $\ge 2.2$ | Limpeza de hardware, Pipelines, classificação e recomendação por cosseno |
| **Visualização & Relatório** | Jupyter Notebook, Matplotlib, Seaborn | $\ge 3.8$ | Relatório executável no Google Colab, EDA e simulador interativo |
| **Back-end (API REST)** | Flask, Flask-CORS | $\ge 3.0$ | Servidor HTTP (`/api/predict`, `/api/dataset`, `/api/health`, `/train`) |
| **Front-end** | HTML5, CSS3, JavaScript (ES6+ Fetch API) | Nativo | Interface guiada com presets de perfil, dicas para leigos e 3 blocos de resultado |
| **Testes & Quality Gate** | PyTest | $\ge 8.0$ | Quality Gate Duplo ($\ge 0{,}80$ + regras de negócio de recomendação) e contrato HTTP |

---

## 3. Engenharia de Atributos e Perfis de Uso

Cada registro em `data/notebooks.csv` contém metadados do anúncio (`titulo`, `marca`) e **5 atributos numéricos preditivos**:

| Atributo | Descrição Técnica | Regra de Padronização (Data Cleaning) |
| :--- | :--- | :--- |
| `preco` | Orçamento / Preço atual em R$ | Filtro de *outliers* removendo peças avulsas e valores abaixo de R$ 1.200 |
| `ram_gb` | Memória RAM em GB (`4`, `8`, `16`, `32`, `64`) | Extração via Regex (ex: `"16 GB"` $\rightarrow$ `16`) |
| `ssd_gb` | Armazenamento SSD em GB (`128` a `2048`) | Extração via Regex convertendo `"512 GB"` $\rightarrow$ `512` e `"1 TB"` $\rightarrow$ `1024` |
| `score_cpu` | Nível do processador (`1` a `5`) | `1`: Celeron/Athlon \| `2`: i3/Ryzen 3 \| `3`: i5/Ryzen 5 \| `4`: i7/Ryzen 7/Apple M1-M2 \| `5`: i9/Ryzen 9/M3 Pro |
| `tem_gpu_dedicada` | Presença de placa de vídeo dedicada (`0` ou `1`) | Busca por termos `"RTX"`, `"GTX"`, `"Radeon RX"` ou `"Dedicada"` |
| `target` | Perfil de Uso Recomendado (`0`, `1` ou `2`) | `0`: **Uso Básico / Estudos** \| `1`: **Trabalho / Programação** \| `2`: **Jogos / Edição 3D** |

---

## 4. Estrutura de Diretórios do Repositório

```text
mvp_puc_rio_final/
├── notebook/
│   └── notebook_MVP.ipynb         # Relatório executável no Google Colab com EDA e Simulador (4,0 pts)
├── backend/
│   ├── __init__.py                # Inicializador do pacote back-end
│   ├── app.py                     # Controlador HTTP (Rotas Flask e entrega do Front-end)
│   ├── model_service.py           # Camada de Serviço Singleton: Classificação + Top-5 KNN + Tendências (SOLID)
│   ├── train_model.py             # Pipeline reprodutível para limpeza Regex, treino e geração do model.pkl
│   ├── model.pkl                  # Artefato binário unificado (Classificador + Scaler + KNN Cosseno + Catálogo)
│   └── requirements.txt           # Dependências de execução do projeto
├── frontend/
│   ├── index.html                 # Interface guiada com Presets de Perfil, Dicas ("?") e Painel de Tendências
│   ├── app.js                     # Controlador de eventos e renderização dos 3 blocos de saída
│   └── train.html                 # Console de execução e logs de treinamento em tempo real (SSE)
├── tests/
│   └── test_model.py              # Suíte PyTest: Quality Gate Duplo, Serviço e Rotas HTTP
├── data/
│   └── notebooks.csv              # Base limpa de notebooks do mercado BR para leitura via GitHub Raw
├── Docs/
│   └── plano-implementacao-mvp.md # Plano tático de implementação do MVP
├── .gitignore                     # Exclusão de arquivos temporários e ambientes virtuais
└── README.md                      # Documentação técnica completa do repositório
```

---

## 5. Instruções de Instalação e Execução Local (Passo a Passo)

### 5.1. Clonar o Repositório
```bash
git clone https://github.com/viniciusad/mvp_puc_rio_final.git
cd mvp_puc_rio_final
```

### 5.2. Criar e Ativar o Ambiente Virtual
```bash
# Opção A: Usando uv
uv venv .venv
.venv\Scripts\activate

# Opção B: Usando python venv padrão (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5.3. Instalar as Dependências
```bash
pip install -r backend/requirements.txt
# Ou com uv:
uv pip install -r backend/requirements.txt
```

### 5.4. (Opcional) Regerar a Base e o Modelo Localmente
O repositório já inclui `data/notebooks.csv` e `backend/model.pkl`. Para regerar o artefato:
```bash
python backend/train_model.py
```

### 5.5. Executar o Servidor Back-end e Acessar o Front-end
```bash
python backend/app.py
```
- Acesse **`http://localhost:5000`** no navegador para utilizar a aplicação completa.
- Acesse **`http://localhost:5000/train`** para disparar e acompanhar o retreinamento ao vivo pelo navegador.
- Endpoint de Health Check: `GET http://localhost:5000/api/health`
- Endpoint de Recomendação e Tendências: `POST http://localhost:5000/api/predict` com payload JSON:
  ```json
  {
    "features": [6800, 16, 512, 4, 1]
  }
  ```

---

## 6. Instruções para Execução dos Testes Automatizados (PyTest)

Para executar a suíte de testes automatizados e validar o **Quality Gate Duplo**:

```bash
pytest tests/ -v -s
```

1. **`test_desempenho_modelo_quality_gate`**: Verifica no conjunto de teste retido (*Holdout 20%*) se **Acurácia $\ge 0{,}80$** e **F1-Score Macro $\ge 0{,}80$**.
2. **`test_coerencia_recomendacao_knn_regras_negocio`**: Garante que 100% das recomendações Top-5 para o perfil *Jogos / Edição 3D* possuem `tem_gpu_dedicada == 1`, `ram_gb >= 16` e compatibilidade média por cosseno $\ge 0{,}80$.
3. **`test_contrato_inferencia_model_service` & `test_api_endpoints_health_e_predict`**: Validam o contrato dos 3 blocos de saída e todas as rotas da API Flask.

---

## 7. Link para o Notebook Executável no Google Colab

- **Arquivo no Repositório**: [`notebook/notebook_MVP.ipynb`](notebook/notebook_MVP.ipynb)
- O notebook carrega automaticamente `data/notebooks.csv` via URL Raw pública do GitHub (com fallback resiliente local/em memória), sem exigir chaves de API privadas, executando do início ao fim (*Run All*) e disponibilizando um simulador interativo de recomendação.

---

## 8. Diretrizes de DevSecOps, Privacidade e LGPD

1. **Anonimização de Dados de Vendedores (*Privacy by Design*)**: Identificadores de vendedores, apelidos de contas e localizações foram removidos da base `data/notebooks.csv`, preservando apenas marca, título técnico, especificações de hardware e preço de mercado.
2. **Minimização de Dados do Usuário**: O formulário solicita apenas os 5 parâmetros técnicos/orçamentários necessários para o cálculo de similaridade e classificação, sem coletar Nome, CPF, E-mail ou IP.
3. **Sanitização de Logs**: O servidor Flask valida rigorosamente os tipos numéricos de entrada e não armazena payloads de orçamento de usuários em logs de erro.
