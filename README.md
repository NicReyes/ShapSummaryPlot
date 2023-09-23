# ShapSummaryPlot
### Bienvenido!! 
# ShapSummaryPlot

Una técnica que se utiliza para explicar las predicciones de un modelo de manera más interpretable y entender cómo las características o variables de entrada individuales contribuyen a las predicciones del modelo.


## Installation

Usa the package manager [pip](https://pip.pypa.io/en/stable/) to install sahp.

```bash
!pip install shap
```
Crea un objeto explainer para calcular las contribuciones de cada característica en el modelo best_model
```bash
explainer = shap.TreeExplainer(best_model)
```
Prepara y manipula el conjunto de datos de prueba (test_df) 

```python
# Crear un nuevo DataFrame 'test_df' utilizando el conjunto de prueba
test_df = last_X_test

# Agregar la columna de la característica objetivo 'target_feature' al DataFrame
test_df[target_feature] = last_y_test

# Muestrear el DataFrame 'test_df' al 80% de su tamaño original
test_df = test_df.sample(frac=0.8)
```
Prepara los datos que se utilizarán para calcular los valores SHAP

```python
X_explain = test_df.drop(target_feature, axis=1)
shap_values = explainer.shap_values(X_explain, check_additivity=False)
```
Crea un gráfico de resumen de los valores SHAP calculados en el paso anterior
```python
shap.summary_plot(shap_values, X_explain)
```
#### Este gráfico proporciona una representación visual de cómo cada característica afecta las predicciones del modelo. Ayuda a comprender cuáles características tienen la mayor influencia en las predicciones y su dirección de impacto (positiva o negativa).
