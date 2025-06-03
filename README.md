# KNN

```markdown
# KNN Classifier for Iris Dataset

This project demonstrates a K-Nearest Neighbors (KNN) classifier implementation on the Iris dataset, including hyperparameter tuning, decision boundary visualization, and feature correlation analysis.

## Model Training

Trained a KNN classifier using scikit-learn with:
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
```

## Hyperparameter Tuning

Used validation curves to find optimal k value (k=11) based on F1 score:
```python
from sklearn.model_selection import validation_curve
param_range = range(1, 30)
train_scores, val_scores = validation_curve(
    KNeighborsClassifier(), X, y, 
    param_name="n_neighbors", 
    param_range=param_range,
    scoring="f1_weighted"
)
```
![Validation Curve](Screenshot%2025-06-03%173139.png)


![Decision Boundary](Screenshot%202025-06-03%20180802.png)

## Decision Boundary Visualization

Created decision boundary plots for different feature pairs:

```python
from sklearn.inspection import DecisionBoundaryDisplay
DecisionBoundaryDisplay.from_estimator(
    knn, X[:, [0,1]], response_method="predict",
    cmap=cmap_light, plot_method="pcolormesh"
)
```

![Sepal Length vs Width](Screenshot%202025-06-03%20180752.png)
![Petal Length vs Width](Screenshot%202025-06-03%20180809.png)

## Feature Correlation Analysis

Visualized feature correlations using Seaborn:
```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

![Correlation Matrix](Screenshot%202025-06-03%20181431.png)

## Key Findings
- Petal measurements show strong positive correlation (0.96)
- Sepal width has negative correlation with other features
- Optimal k value found to be 11 through validation curves
- Decision boundaries clearly separate classes when using petal measurements
```
