# FIAP Tech Challenge - Previsão de Preços de Aluguel

## 📋 Descrição do Projeto

Este projeto foi desenvolvido como parte do Tech Challenge da FIAP, com o objetivo de criar um modelo de Machine Learning para previsão de preços de aluguel de imóveis no Canadá. O projeto engloba todas as etapas do ciclo de vida de um projeto de ML, desde a coleta e análise de dados até a construção de uma aplicação produtiva.

## 🎯 Objetivo

Desenvolver um modelo de Machine Learning capaz de prever com precisão os preços de aluguel de imóveis com base em características como localização geográfica, tamanho, número de quartos e banheiros, tipo de construção, e outras features relevantes.

## 📊 Dataset

- **Fonte**: Listings de aluguel publicados em setembro de 2025
- **Tamanho Inicial**: 5.857 registros com 24 features
- **Tamanho Final (após limpeza)**: 4.552 registros
- **Escopo Geográfico**: Províncias canadenses

### Features Principais:
- **Geográficas**: longitude, latitude, state, city
- **Características do Imóvel**: size (sqft), total_rooms, total_bathrooms, building_type
- **Comodidades**: allow_pets, allow_smoking, furnished, count_private_parking
- **Target**: price (preço mensal de aluguel)

## 🔄 Etapas do Processo

### 1. **Importação e Configuração**
- Carregamento de bibliotecas necessárias (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Configuração do ambiente de análise
- Importação do dataset

### 2. **Exploração Inicial dos Dados (EDA)**
- Análise estatística descritiva
- Identificação de valores ausentes
- Análise de tipos de dados
- Distribuição das variáveis

### 3. **Limpeza de Dados**
- Remoção de duplicatas (58 registros removidos)
- Filtragem por tipo de aluguel (mantidos apenas contratos fixos de longo prazo)
- Remoção de colunas com >90% de valores ausentes (count_balconies, transit_distance)
- Eliminação de features problemáticas (year_built, construction_type)
- Filtragem geográfica (mantidos apenas imóveis em províncias canadenses)

### 4. **Análise da Variável Alvo (Price)**
- Distribuição de preços
- Transformação logarítmica
- Análise de box plots
- Estatísticas descritivas (média: $2.457, mediana: $2.195)

### 5. **Detecção e Remoção de Outliers**
- Método IQR (Interquartile Range) com multiplicador de 3
- Remoção de outliers de preço (100-10.000 CAD)
- Filtragem de outliers geográficos
- Limitação de tamanho (50-5.000 sqft)
- Limitação de quartos (0-10) e banheiros (0-6)
- Total de 1.305 outliers removidos

### 6. **Engenharia de Features**
- `price_per_sqft`: preço por metro quadrado
- `rooms_per_bathroom`: razão quartos/banheiros
- `has_parking`: indicador binário de estacionamento
- `is_large`: indicador de imóvel grande (acima da mediana)
- `size_category`: categorização por tamanho (tiny, small, medium, large, xlarge, huge)
- `price_category`: categorização por preço (budget, economy, standard, premium, luxury)

### 7. **Análise Exploratória Avançada**
- Distribuição de preços por tipo de construção
- Análise geográfica com mapas de dispersão
- Correlação entre preço e tamanho
- Impacto de features categóricas (pets, fumantes, mobiliado)
- Matriz de correlação entre variáveis numéricas

### 8. **Análise de Correlação**
- Matriz de correlação completa
- Identificação das features mais correlacionadas com o preço
- Top features: total_bathrooms, size, longitude

### 9. **Preparação para Modelagem**
- **Decisão Estratégica**: Remoção da feature 'city' para evitar high cardinality (151 cidades diferentes)
- **Solução**: Uso de longitude/latitude para capturar padrões geográficos
- Features finais: 12 variáveis + 1 target
- Tratamento de valores ausentes (mediana para numéricas, 0 para booleanas, moda para categóricas)

### 10. **Divisão Treino-Teste**
- Estratificação por: size_category × total_rooms × total_bathrooms
- Split: 80% treino (3.641 amostras) / 20% teste (911 amostras)
- Uso de StratifiedShuffleSplit para balanceamento

### 11. **Pipeline de Pré-processamento**
- **Features Numéricas** (10): StandardScaler para normalização
- **Features Categóricas** (2): OneHotEncoder para state e building_type
- Total de 21 features após pré-processamento

### 12. **Treinamento e Comparação de Modelos**

Três modelos foram avaliados:

| Modelo | Test MAE | Test RMSE | Test R² | Overfitting |
|--------|----------|-----------|---------|-------------|
| **Linear Regression** | $394.24 | $611.16 | 0.6678 | 0.014 |
| **Decision Tree** | $307.64 | $513.80 | 0.7652 | 0.182 |
| **Random Forest** | **$243.53** | **$406.23** | **0.8532** | 0.126 |

### 13. **Validação Cruzada**
- 5-fold cross-validation no Random Forest
- MAE médio: $265.77 ± $17.43
- Confirma a robustez do modelo

### 14. **Análise de Importância de Features**

Top 5 features mais importantes:
1. **total_bathrooms** (39.61%)
2. **size** (20.31%)
3. **longitude** (17.20%)
4. **latitude** (10.39%)
5. **total_rooms** (6.87%)

### 15. **Modelo Final Selecionado**

**Random Forest Regressor**
- **Performance**:
  - Test MAE: $243.53
  - Test RMSE: $406.23
  - Test R²: 0.8532
- **Vantagens**:
  - Melhor performance entre os modelos testados
  - Bom equilíbrio entre viés e variância
  - Capaz de capturar relações não-lineares
  - Robusto a outliers
  - Generalização eficaz para cidades não vistas no treino

## 🎯 Principais Melhorias Implementadas

- ✅ **Análise Exploratória Completa**: Visualizações detalhadas e insights sobre os dados
- ✅ **Tratamento de High Cardinality**: Substituição de city names por coordenadas geográficas
- ✅ **Detecção de Outliers**: Método IQR robusto
- ✅ **Engenharia de Features**: Criação de features derivadas relevantes
- ✅ **Validação Estratificada**: Divisão balanceada por múltiplas dimensões
- ✅ **Comparação de Modelos**: Avaliação objetiva de múltiplos algoritmos
- ✅ **Generalização**: Modelo funciona com qualquer cidade, mesmo não vistas no treino

## 📦 Arquivos do Projeto

- `Price_prediction_1_35_PT-BR.ipynb`: Notebook completo com análise e treinamento
- `random_forest_rental_price_model_v1_35.pkl`: Pipeline completo (preprocessor + modelo)
- `model_features_v1_35.json`: Metadados e informações das features

## 🚀 Como Usar o Modelo

```python
import joblib
import pandas as pd

# Carregar o modelo
model = joblib.load('random_forest_rental_price_model_v1_35.pkl')

# Criar dados de exemplo
sample = pd.DataFrame({
    'longitude': [-79.4163],
    'latitude': [43.70011],
    'state': ['ON'],
    'building_type_txt_id': ['highrise'],
    'total_rooms': [2],
    'total_bathrooms': [2],
    'size': [700],
    'allow_pets': [1],
    'allow_smoking': [0],
    'furnished': [0],
    'count_private_parking': [1],
    'student_friendly': [0]
})

# Fazer previsão
predicted_price = model.predict(sample)
print(f"Preço previsto: ${predicted_price[0]:.2f}")
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas & NumPy**: Manipulação de dados
- **Scikit-learn**: Machine Learning
- **Matplotlib & Seaborn**: Visualização
- **Joblib**: Persistência do modelo


## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do Tech Challenge da FIAP.

---

**FIAP - Faculdade de Informática e Administração Paulista**

**Curso**: Machine Learning Engineering - 5MLET - Pós Tech FIAP

**Fase**: Tech Challenge Fase 3

**Ano**: 2025