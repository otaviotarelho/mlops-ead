import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_dataset():
    data = load_breast_cancer()
    X,y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(random_state=0, criterion="entropy", class_weight="balanced", max_depth=3,
                                   min_samples_leaf=3, min_samples_split=2)
    model.fit(X_train, y_train)
    return model

def random_select_test_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    idx = np.random.randint(0, X.shape[0])
    sample_features = X[idx].reshape(1, -1)
    sample_label = y[idx]

    print("Index:", idx)
    print("True label:", sample_label)
    print("True label Formated:", print_label(sample_label))
    print("\n")
    return sample_features

def predict(model: DecisionTreeClassifier, X):
    return model.predict(X)

def print_label(result) -> str:
    return "Malignant" if result == 0 else "Benign"

if __name__ == '__main__':
    model: DecisionTreeClassifier | None = None

    while True:
        print("Breast Cancer Decision Tree Classifier")
        print("\n==== MENU ====")
        print("1. Train model")
        print("2. Enter and Predict test data")
        print("3. Random Select from Dataset")
        print("4. Exit")

        try:
            option = int(input("Option: "))
        except ValueError:
            print("Invalid Option. Please enter a number between 1 and 4. \n")
            continue

        if option == 1:
            model = train_dataset()
            print("Model Trained... Select new option\n")
        elif option == 2:
            if model is None:
                print("No model selected. Please select a model on option 1.\n")
                continue

            values = input("Enter 30 feature values separated by commas:\n")

            try:
                values = input("Enter 30 feature values separated by commas:\n")
                features_list = list(map(float, values.split(",")))
                features = np.array(features_list).reshape(1, -1)

                prediction = predict(model, features)
                print("Model Prediction:", prediction)
                print(f"Model Detailed Conclusion: {print_label(prediction[0])}\n")
            except ValueError:
                print("Invalid input. Please enter only numbers.\n")
                continue

        elif option == 3:
            if model is None:
                print("No model selected. Please select a model on option 1.\n")
                continue

            random_test = random_select_test_data()
            prediction = predict(model, random_test)

            print("Model Prediction:", prediction[0])
            print(f"Model Detailed Conclusion: {print_label(prediction[0])}\n")
        elif option == 4:
            break