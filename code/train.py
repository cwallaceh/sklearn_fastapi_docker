# Import packages
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import gzip


# Load the dataset
data = pd.read_csv('data/breast_cancer.csv')

# Preselected feature
selected_features = [
    'concavity_mean',
    'concave_points_mean',
    'perimeter_se',
    'area_se',
    'texture_worst',
    'area_worst'
]

# Preprocess dataset
data = data.set_index('id')
data['diagnosis'] = data['diagnosis'].replace(['B', 'M'], [0, 1])  # Encode y, B -> 0 , M -> 1

# Split into train and test set, 80%-20%
y = data.pop('diagnosis')
X = data
X = X[selected_features.copy()]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ensemble of 3 models
estimators = []
estimators.append(('logistic', LogisticRegression()))
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))

# Create the Ensemble Model
ensemble = VotingClassifier(estimators)

# Make preprocess Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer()),  # Missing value Imputer
    ('scaler', MinMaxScaler(feature_range=(0, 1))),  # Min Max Scaler
    ('model', ensemble)  # Ensemble Model
])

# Train the model
pipe.fit(X_train, y_train)

# Test Accuracy
print("Accuracy: %s%%" % str(round(pipe.score(X_test, y_test), 3) * 100))

# Export model
joblib.dump(pipe, gzip.open('model/model_binary.dat.gz', "wb"))
