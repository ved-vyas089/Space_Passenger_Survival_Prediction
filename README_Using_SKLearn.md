# 🚀 Spaceship Titanic - Machine Learning Model

This repository contains a Machine Learning pipeline to predict passenger survival (`Transported`) in the Spaceship Titanic Kaggle competition.

## 📂 Dataset
The dataset consists of two CSV files:
- `train.csv`: Training data
- `test.csv`: Test data

## 📌 Features Used
- **Categorical:** `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`
- **Numerical:** `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`

## 🛠 Preprocessing Steps
1. **Handling Missing Values**:
   - Categorical columns filled with mode 🏠
   - Numerical columns filled with median 🔢
2. **Feature Engineering**:
   - `Cabin` split into `Deck`, `Num`, `Side` 🚢
3. **Encoding**:
   - Label encoding for categorical variables 🎭
4. **Scaling**:
   - Standardization of numerical features 📊

## 📊 Exploratory Data Analysis (EDA)
- 🔹 Count plots for `CryoSleep` and `VIP`
- 🔹 Pie charts for `HomePlanet` and `Destination` distributions
- 🔹 Histogram of `Age`
- 🔹 Bar plot of average expenditures

## 🤖 Model Training
The following models are used:
- 🌲 **RandomForestClassifier**
- ⚡ **XGBoost (XGBClassifier)**
- 💡 **LightGBM (LGBMClassifier)**

### 📈 Performance Evaluation
Each model is evaluated using:
- **Accuracy Score** 📊
- **Classification Report** 📜
- **Confusion Matrix** 🔥

## 🔮 Predictions & Submission
1. **Test Data Preprocessing** 🛠
2. **Making Predictions using LGBM Model** 🎯
3. **Saving Predictions to `submission.csv`** 📝

## 📜 Output Example
```bash
Submission file created: submission.csv
   PassengerId  Transported
0         001        True
1         002       False
2         003        True
...
```

## 🚀 Run the Code
```bash
python spaceship_titanic.py
```

## 🏆 Kaggle Submission
Upload `submission.csv` to Kaggle to check your model's performance!

---
🔗 *Happy coding! May your model have high accuracy!* 🎯

