# 🚀 Spaceship Titanic Survival Prediction with Neural Network

## 📌 Overview
This project builds a **Neural Network Model** using `TensorFlow` and `Keras` to predict whether a passenger was **transported or not** in the Spaceship Titanic dataset. The dataset is preprocessed, encoded, and trained with a deep learning model.

---
## 📂 Dataset
The dataset consists of two files:
- **`train.csv`** - Contains labeled training data.
- **`test.csv`** - Contains unlabeled test data for predictions.

---
## 🛠 Libraries Used
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
```

---
## 🏗 Data Preprocessing
### ✅ Handling Missing Values
- Categorical columns filled with **mode**.
- Numerical columns filled with **median**.

### 🔹 Feature Engineering
- **Cabin** split into `Deck`, `Num`, and `Side`.
- Unnecessary columns (`Name`, `PassengerId`, `Cabin`) dropped.

### 🔄 Encoding Categorical Variables
- Used `LabelEncoder()` to convert categorical values into numerical format.

### 📊 Data Scaling
- Used `StandardScaler()` for feature scaling.

### ✂️ Train-Test Split
- **80%** training data
- **20%** validation data

---
## 🔥 Model Architecture
```python
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
```

### ⚙️ Compilation
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 🎯 Training
```python
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
```

---
## 📈 Model Evaluation
```python
model.evaluate(x_val, y_val)
```

---
## 🔮 Predictions
```python
y_pred = model.predict(x_val)
y_pred = (np.array(y_pred) > 0.5).astype(int)
```

### 🔍 First 10 Predictions
```python
print(y_pred[:10])
print(y_val[:10])
```

---
## 📊 Confusion Matrix
```python
conf_mat = tf.math.confusion_matrix(labels=y_val, predictions=y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()
```

---
## 💾 Results
✅ **Neural Network successfully trained & evaluated!**
📌 Confusion matrix visualized for better insights.

---
## 📜 Conclusion
- A deep learning model was implemented for **Spaceship Titanic survival prediction**.
- Feature engineering and preprocessing steps improved accuracy.
- Further tuning (more layers, epochs, hyperparameter tuning) can enhance results.

🔗 **Happy Coding! 🚀**

