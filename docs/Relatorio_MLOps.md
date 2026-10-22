# RELATÓRIO TÉCNICO DE ENGENHARIA DE DADOS E MLOPS
**Projeto:** AeroMotion Dynamics - Predição de Falhas em Turbinas Industriais  
**Métricas Extraídas do Treinamento Oficial (Split Temporal 70/30)**  

---

### 1. Decisões de Pré-Processamento e Engenharia de Features
* **Filtragem de EMI (Interferência Eletromagnética):** Identificou-se 58 ocorrências de saturação térmica a 999.9 °C e 45 leituras de vibração negativa a -99.0 mm/s. Esses valores foram anulados e tratados via interpolação linear isolada por equipamento, preservando a coerência física de cada bancada.
* **Engenharia de Features:** Criação das variáveis móveis de 1h e 6h (`vibracao_mean_1h`, `temp_mean_1h`, etc.) para captura da inércia mecânica, somadas à métrica de *Severidade Dinâmica* ($\frac{\text{Vibração RMS}}{\text{Rotação RPM}}$), que demonstrou alta correlação com modos de desbalanceamento e desgaste de mancais.

---

### 2. Tabela Comparativa de Desempenho Preditivo

| Modelo | Classe de Falha | Precisão | Recall | F1-Score | Acurácia Global |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 0 (Normal) | 99,62% | 99,91% | 99,76% | **99,47%** |
| | 1 (Desgaste Rolamento) | 97,73% | 93,99% | 95,82% | |
| | 2 (Desbalanceamento Rotor) | 98,01% | 98,01% | 98,01% | |
| | 3 (Cavitação/Superaquec.) | 100,00% | 98,13% | 99,06% | |
| **XGBoost Classifier** | 0 (Normal) | 99,72% | 99,97% | 99,84% | **99,67%** |
| | 1 (Desgaste Rolamento) | 100,00% | 94,54% | 97,19% | |
| | 2 (Desbalanceamento Rotor) | 98,68% | 99,34% | 99,01% | |
| | 3 (Cavitação/Superaquec.) | 99,07% | 100,00% | 99,53% | |

* **Conclusão Comparativa:** Ambos os modelos superaram a meta de **Recall > 90%** em todas as classes críticas. O **XGBoost** foi eleito o modelo primário por apresentar **100% de recall na classe 3 (Cavitação/Superaquecimento)** e menor taxa de falsos negativos.

---

### 3. Plano Contínuo de MLOps (Janela de 12 Meses)
1. **Monitoramento de Data Drift (Deriva dos Dados):**
   * Aplicação do teste bicaudal de Kolmogorov-Smirnov (KS-test) semanal sobre as distribuições de vibração e temperatura dos sensores IoT contra a base de referência.
   * Disparo de alerta quando o valor de p-value for inferior a 0.05.
2. **Monitoramento de Concept Drift (Deriva de Conceito):**
   * Avaliação mensal da degradação de F1-Score a partir das ordens de serviço confirmadas pela equipe de manutenção mecânica.
3. **Estratégia de Retreinamento:**
   * Pipeline agendado a cada 90 dias com incorporação de novas amostras validadas, aplicando divisão temporal rolling window e reajuste dos hiperparâmetros.
4. **Detecção de Anomalias Não Supervisionada:**
   * O algoritmo Isolation Forest opera em paralelo na API sinalizando comportamentos fora da curva normal que ainda não foram rotulados.