import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def confusion_matrix(y_test, predictions):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(predictions)):
        if predictions[i] == 'fire' and y_test[i] == 'fire':
            tp += 1
        elif predictions[i] == 'not fire' and y_test[i] == 'not fire':
            tn += 1
        elif predictions[i] == 'fire' and y_test[i] == 'not fire':
            fp += 1
        else:
            fn += 1
    tp_percent, tn_percent, fp_percent, fn_percent = tp/len(predictions), tn/len(predictions), fp/len(predictions), fn/len(predictions)
    return tp_percent, tn_percent, fp_percent, fn_percent

def accuracy(tp, tn, fp, fn):
    return ((tp + tn) / (tp + tn + fp + fn)) * 100

def plot_confusion_matrix(tp, tn, fp, fn, ax):
    tp, tn, fp, fn = tp * 100, tn * 100, fp * 100, fn * 100
    cax = ax.matshow([[tp, fp], [fn, tn]], cmap=plt.cm.Blues, alpha=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{[[tp, fp], [fn, tn]][i][j]:.3f}', va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (Percentages)')
    fig.colorbar(cax)

def show_coefficients(logmodel, feature_names):
    coefs = logmodel.coef_[0]
    coef_dict = {feature_names[i]: coefs[i] for i in range(len(feature_names))}

    print("Coefficients:")
    for feature, coef in coef_dict.items():
        print(f"{feature}: {coef:.4f}")

    most_influential_feature = max(coef_dict, key=lambda k: abs(coef_dict[k]))
    print(f"\nMost Influential Feature: {most_influential_feature} (Coefficient: {coef_dict[most_influential_feature]:.4f})")

def plot_coefficients(logmodel, feature_names, ax):
    coefs = logmodel.coef_[0]
    coef_dict = {feature_names[i]: coefs[i] for i in range(len(feature_names))}
    sorted_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    ax.bar([x[0] for x in sorted_features], [x[1] for x in sorted_features])
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels([x[0] for x in sorted_features], rotation=90)
    ax.set_title('Feature Coefficients')

def test_formula(logmodel):
    intercept = logmodel.intercept_[0]
    coefs = logmodel.coef_[0]
    str_formula = f"1/(1 + e^(-({intercept:.4f}"
    for i in range(len(coefs)):
        str_formula += f" + {coefs[i]:.4f} * x{i+1}"
    str_formula += "))"
    print(f"\nFormula: {str_formula}")




if __name__ == "__main__":
    data_paths = ['data/whole_dataset.csv', 'data/Bejaia.csv', 'data/Sidi_Bel-abbes.csv']
    # data_paths = ['data/fire_0_1.csv']
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Confusion Matrix and Coefficients per area')

    for i, data_path in enumerate(data_paths):
        data = pd.read_csv(data_path)
        useful_data = data[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes']]
        # useful_data = data[['Temperature', 'RH', 'Ws', 'Rain','Classes']]
        X = useful_data.drop('Classes', axis=1)
        Y = useful_data['Classes']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        logmodel = LogisticRegression(max_iter=1000)
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)
        y_test = y_test.to_numpy()
        tp, tn, fp, fn = confusion_matrix(y_test, predictions)
        print(f'Accuracy: {accuracy(tp, tn, fp, fn)}')

        # Plotting confusion matrix
        plot_confusion_matrix(tp, tn, fp, fn, axs[0, i])
        title_name = data_path.split('/')[1].split('.')[0].replace('_', ' ').title()
        if title_name == 'Whole Dataset':
            title_name = 'Both Areas'
        axs[0, i].set_title(f'Area - {title_name}')

        # Plotting coefficients
        feature_names = X.columns.tolist()
        plot_coefficients(logmodel, feature_names, axs[1, i])
        axs[1, i].set_title(f'Area - {title_name}')

        print("-----------------------------------")
        # test_formula(logmodel)
        show_coefficients(logmodel, feature_names)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()









