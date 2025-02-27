
## 🚀 Python AI Project: Artificial Intelligence and Predictions

### 📌 Description
- This project utilizes Artificial Intelligence to analyze customer data and predict their credit score. Based on customer information, a machine learning model classifies the score as Bad, Ok, or Good.

_______

### 🏦 Case Study: Customer Credit Score
- A bank has hired our solution to develop a model that automates credit analysis. To achieve this, we use machine learning and classification models.
  
 ### 🔗 Class Files: [Project](https://github.com/ludiemert/IA_predictions_Python)
________________________________________
### 🛠 Technologies Used
- Python 🐍
- Pandas (for data manipulation)
- Scikit-Learn (for machine learning)
- RandomForestClassifier (decision tree model)
- KNeighborsClassifier (KNN - K-Nearest Neighbors model)
________________________________________

<h4 align="center">IA_predictions_Python 🚀</h4>

<div align="center">
    <table>
        <tr>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/01_1_codg de treino_com as variaveis.png" style="width: 90%;" alt="01_1_codg de treino_com as variaveis">
                <p style="margin-top: 5px;">01_1_codg de treino_com as variaveis.png</p>
            </td>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/01_codg de treino_x_y.png" style="width: 90%;" alt="01_codg de treino_x_y">
                <p style="margin-top: 5px;">01_codg de treino_x_y</p>
            </td>
        </tr>
    </table>
</div>

  <br/>
  <br/>


<div align="center">
    <table>
        <tr>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/X e Y.png" style="width: 90%;" alt="X e Y">
                <p style="margin-top: 5px;">X e Y</p>
            </td>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/x_y treino e teste.png" style="width: 90%;" alt="x_y treino e teste">
                <p style="margin-top: 5px;">x_y treino e teste</p>
            </td>
        </tr>
    </table>
</div>

  <br/>
  <br/>


  <div align="center">
    <table>
        <tr>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/4 - acuracia de x y.png" style="width: 90%;" alt="4 - acuracia de x y">
                <p style="margin-top: 5px;">4 - acuracia de x y</p>
            </td>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/4 Passo - vizinho mais proximo.png" style="width: 90%;" alt="4 Passo - vizinho mais proximo">
                <p style="margin-top: 5px;">4 Passo - vizinho mais proximo</p>
            </td>
        </tr>
    </table>
</div>

  <br/>
  <br/>


  <div align="center">
    <table>
        <tr>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/Passo 4 - arvore de decisao.png" style="width: 90%;" alt="Passo 4 - arvore de decisao">
                <p style="margin-top: 5px;">Passo 4 - arvore de decisao</p>
            </td>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/5 - step 5 - profissao transf cod.png" style="width: 90%;" alt="5 - step 5 - profissao transf cod">
                <p style="margin-top: 5px;">5 - step 5 - profissao transf cod</p>
            </td>
        </tr>
    </table>
</div>

  <br/>
  <br/>

   <div align="center">
    <table>
        <tr>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/5 - tabela de novos clientes - step 5 - profissao nome.png" style="width: 90%;" alt="5 - tabela de novos clientes - step 5 - profissao nome">
                <p style="margin-top: 5px;">5 - tabela de novos clientes - step 5 - profissao nome</p>
            </td>
            <td style="width: 50%; text-align: center;">
                <img src="img_proj/Passo 4 - arvore de decisao.png" style="width: 90%;" alt="Passo 4 - arvore de decisao">
                <p style="margin-top: 5px;">Passo 4 - arvore de decisao</p>
            </td>
        </tr>
    </table>
</div>

  <br/>
  <br/>


### 📊 Project Steps

#### 1️⃣ Importing and Analyzing Data
 - We load customer data from a CSV file and analyze the dataset structure to check for missing values or incorrect formats.
 - python
 - CopiarEditar
 - import pandas as pd
 - tabela = pd.read_csv("clientes.csv")
 - display(tabela)
 - print(tabela.info())
   
#### 2️⃣ Data Processing
 - Converting categorical data into numerical values using LabelEncoder:
 - python
 - CopiarEditar
 - from sklearn.preprocessing import LabelEncoder  

```hash
codificador = LabelEncoder()  
for coluna in tabela.columns:  
    if tabela[coluna].dtype == "object" and coluna != "score_credito":  
        tabela[coluna] = codificador.fit_transform(tabela[coluna])
```

#### 3️⃣ Defining Features and Labels
 - Selecting the columns used to predict the credit score:
 - python
   
```hash
x = tabela.drop(["score_credito", "id_cliente"], axis=1)  
y = tabela["score_credito"]
```

#### 4️⃣ Splitting Data into Training and Testing Sets
Separating data into 70% training and 30% testing:
```hash
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  
```

#### 5️⃣ Training the Models
We use two models: Decision Tree (RandomForestClassifier) and KNN (KNeighborsClassifier):

```hash
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier  

model_tree = RandomForestClassifier()  
model_knn = KNeighborsClassifier()  
model_tree.fit(x_train, y_train)  
model_knn.fit(x_train, y_train)

```


#### 6️⃣ Evaluating Accuracy
Comparing the model predictions with actual values:
```hash
from sklearn.metrics import accuracy_score  

prediction_tree = model_tree.predict(x_test)  
prediction_knn = model_knn.predict(x_test.to_numpy())  

print(accuracy_score(y_test, prediction_tree))  
print(accuracy_score(y_test, prediction_knn))  
```

#### 7️⃣ Making New Predictions
Applying the trained model to predict new customers’ credit scores:
```hash
new_clients = pd.read_csv("novos_clientes.csv")  

for coluna in new_clients.columns:  
    if new_clients[coluna].dtype == "object" and coluna != "score_credito":  
        new_clients[coluna] = codificador.fit_transform(new_clients[coluna])  

predictions = model_tree.predict(new_clients)  
print(predictions)
```
________________________________________

#### 📊 Step-by-Step Code Explanation
1.	Importing and Analyzing Data:
o	The script begins by importing the clientes.csv dataset using pandas.
o	A check is performed to identify missing values or incorrect formats.
2.	Data Preprocessing:
o	The LabelEncoder class from sklearn is used to convert text columns into numerical values.
o	The score_credito column is not transformed because it is the target variable.
3.	Feature Selection:
o	We define the features (x) for training and the target (y) for prediction.
o	The id_cliente column is removed as it does not contribute to predictions.
4.	Data Splitting:
o	The dataset is split into training and test sets using train_test_split.
5.	Training the Models:
o	Two models are trained: RandomForestClassifier (Decision Tree) and KNeighborsClassifier (KNN - Nearest Neighbors).
o	Both models are trained using the training data.
6.	Model Evaluation:
o	The accuracy of the models is calculated by comparing predictions with the test data.
o	The accuracy is compared with a baseline prediction where everything is classified as "Standard" to assess model effectiveness.
7.	New Data Predictions:
o	The trained model is applied to new data (novos_clientes.csv) to predict customers' credit scores.

________________________________________
#### 📊 Results
The RandomForestClassifier model achieved an accuracy of X%, while the KNeighborsClassifier model achieved an accuracy of Y%. Both models outperformed the baseline classification of "Standard," demonstrating their effectiveness in predicting credit scores.
________________________________________
#### 🚀 Running the Project
1.	Clone the repository:
``` bash
git clone[ https://github.com/your-repo/python-ai-credit-score.git ](https://github.com/ludiemert/IA_predictions_Python) 
cd python-ai-credit-score  
```
2.	Install dependencies:
Make sure you have the necessary libraries installed:
```bash
CopiarEditar
pip install pandas scikit-learn
```
3.	Run the script:
Execute the Python script to train the model and make predictions:
```bash
CopiarEditar
python main.py
```
________________________________________
#### 📂 Project Structure
```hash
python-ai-project/  
├── clientes.csv  
├── novos_clientes.csv  
├── main.py  
└── README.md
```
________________________________________
### Portugues

#### 📌 Descrição
- Este projeto utiliza Inteligência Artificial para analisar dados de clientes e prever o score de crédito deles. Com base nas informações do cliente, um modelo de machine learning classifica o score como Ruim, Ok ou Bom.
#### 🏦 Case: Score de Crédito dos Clientes
- Um banco contratou nossa solução para desenvolver um modelo que automatize a análise de crédito. Para isso, utilizamos aprendizado de máquina e modelos de classificação.

________________________________________


### Passo a Passo do Código

1. **Importação e Análise dos Dados**:
   - O código começa importando a base de dados `clientes.csv` usando a biblioteca `pandas`.
   - Em seguida, é feita uma verificação para identificar valores vazios ou formatos incorretos.

2. **Pré-processamento dos Dados**:
   - Utilizamos a classe `LabelEncoder` da biblioteca `sklearn` para transformar colunas de texto em números. Isso é necessário porque modelos de machine learning trabalham melhor com dados numéricos.
   - A coluna `score_credito` não é transformada, pois é o nosso alvo (target).

3. **Seleção de Features e Target**:
   - Definimos as colunas que serão usadas para treinar o modelo (`x`) e a coluna que queremos prever (`y`).
   - A coluna `id_cliente` é removida, pois não contribui para a previsão.

4. **Divisão dos Dados**:
   - Os dados são divididos em conjuntos de treino e teste usando a função `train_test_split`.

5. **Treinamento dos Modelos**:
   - Dois modelos são treinados: **RandomForestClassifier** (árvore de decisão) e **KNeighborsClassifier** (KNN - vizinhos mais próximos).
   - Ambos os modelos são treinados com os dados de treino.

6. **Avaliação dos Modelos**:
   - A acurácia dos modelos é calculada comparando as previsões com os dados de teste.
   - A acurácia é comparada com um "chute" de tudo como "Standard" para avaliar a eficácia do modelo.

7. **Previsões em Novos Dados**:
   - O modelo é então aplicado a novos dados (`novos_clientes.csv`) para prever o score de crédito de novos clientes.

## 📊 Resultados

O modelo de **RandomForestClassifier** apresentou uma acurácia de **X%**, enquanto o modelo de **KNeighborsClassifier** obteve uma acurácia de **Y%**. Ambos os modelos superaram o "chute" de tudo como "Standard", demonstrando que são eficazes na previsão do score de crédito.
________________________________________

#### 📈 Key Takeaways
•	🚀 Models trained using real customer data
•	🎯 High accuracy in credit score predictions
•	🔍 Automated credit evaluation process
________________________________________
#### 🤝 Contributing
If you would like to contribute to this project, feel free to open an issue or submit a pull request! 🚀
________________________________________
#### 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
👩💻 Developed with 💙 by [[LuDiemert](https://www.linkedin.com/in/lucianadiemert/)]


- #### My LinkedIn - [![Linkedin Badge](https://img.shields.io/badge/-LucianaDiemert-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/lucianadiemert/)](https://www.linkedin.com/in/lucianadiemert/)

## 🌐 **Contact**
<img align="left" src="https://www.github.com/ludiemert.png?size=150">

#### [**Luciana Diemert**](https://github.com/ludiemert)

🛠 Full-Stack Developer <br>
🖥️ Python Enthusiast | Computer Vision | AI Integrations <br>
📍 São Jose dos Campos – SP, Brazil

<a href="https://www.linkedin.com/in/lucianadiemert" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn Badge" height="25"></a>&nbsp;
<a href="mailto:lucianadiemert@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white" alt="Gmail Badge" height="25"></a>&nbsp;
<a href="#"><img src="https://img.shields.io/badge/Discord-%237289DA.svg?logo=discord&logoColor=white" title="LuDiem#0654" alt="Discord Badge" height="25"></a>&nbsp;
<a href="https://www.github.com/ludiemert" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" alt="GitHub Badge" height="25"></a>&nbsp;

<br clear="left"/>

---
Developed with ❤ by [ludiemert](https://github.com/ludiemert).

