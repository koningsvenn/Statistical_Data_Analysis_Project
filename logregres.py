import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# This function is used to test the number of features to use in the model
def test_RFE(data_path):
    data = pd.read_csv(data_path)
    useful_data = data[['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain', 'Classes']]
    X = useful_data.drop('Classes', axis=1)
    X = standardize_data(X)
    y = useful_data['Classes']

    logmodel = LogisticRegression(max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for n_features in range(1, len(X.columns) + 1):
        rfe = RFE(logmodel, n_features_to_select=n_features)
        score = cross_val_score(logmodel, rfe.fit_transform(X, y), y, cv=cv, scoring='accuracy')
        scores.append(score.mean())

    best_n_features = np.argmax(scores) + 1
    print("Best number of features:", best_n_features)
    print(f'scores: {scores}')

    rfe = RFE(logmodel, n_features_to_select=4)
    rfe.fit(X, y)
    for i in range(len(rfe.support_)):
        if rfe.support_[i]:
            print(X.columns[i])

    plt.plot(range(1, len(X.columns) + 1), scores)
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('RFE')
    plt.savefig('RFE.png')
    plt.show()


# This function is used to calculate the confusion matrix
def confusion_matrix(y_test, predictions):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and y_test[i] == 1:
            tp += 1
        elif predictions[i] == 0 and y_test[i] == 0:
            tn += 1
        elif predictions[i] == 1 and y_test[i] == 0:
            fp += 1
        else:
            fn += 1
    tp_percent, tn_percent, fp_percent, fn_percent = tp/len(predictions), tn/len(predictions), fp/len(predictions), fn/len(predictions)
    return tp_percent, tn_percent, fp_percent, fn_percent

# This function is used to calculate the accuracy
def accuracy(tp, tn, fp, fn):
    return ((tp + tn) / (tp + tn + fp + fn)) * 100

# This function is used to plot the confusion matrix
def plot_confusion_matrix(tp, tn, fp, fn, ax):
    tp, tn, fp, fn = tp * 100, tn * 100, fp * 100, fn * 100
    cax = ax.matshow([[tp, fp], [fn, tn]], cmap=plt.cm.Blues, alpha=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{[[tp, fp], [fn, tn]][i][j]:.3f}', va='center', ha='center')

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(['Fire', 'No Fire'])
    ax.set_yticklabels(['Fire', 'No Fire'])

# This function is used to show the coefficients of the model
def show_coefficients(logmodel, feature_names):
    coefs = logmodel.coef_[0]
    coef_dict = {feature_names[i]: coefs[i] for i in range(len(feature_names))}

    print("Coefficients:")
    for feature, coef in coef_dict.items():
        print(f"{feature}: {coef:.4f}")

    most_influential_feature = max(coef_dict, key=lambda k: abs(coef_dict[k]))
    print(f"\nMost Influential Feature: {most_influential_feature} (Coefficient: {coef_dict[most_influential_feature]:.4f})")

# This function is used to plot the coefficients of the model
def plot_coefficients(logmodel, feature_names, ax):
    coefs = logmodel.coef_[0]
    coef_dict = {feature_names[i]: coefs[i] for i in range(len(feature_names))}
    sorted_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    ax.bar([x[0] for x in sorted_features], [x[1] for x in sorted_features])
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels([x[0] for x in sorted_features], rotation=90)
    ax.set_title('Feature Coefficients')

# This function is used to show the formula of the model
def test_formula(logmodel):
    intercept = logmodel.intercept_[0]
    coefs = logmodel.coef_[0]
    str_formula = f"1/(1 + e^(-({intercept:.4f}"
    for i in range(len(coefs)):
        str_formula += f" + {coefs[i]:.4f} * x{i+1}"
    str_formula += "))"
    print(f"\nFormula: {str_formula}")


# This function is used to test the model based on the type of data
# test_type: 'FWI', 'Observations' or 'All'
# 'FWI': test the model based on the fire behavior index
# 'Observations': test the model based on the weather observations
# 'All': test the model based on all the data
def test_classification(test_type):
    data_paths = ['data_0_1/Both.csv', 'data_0_1/Sidi_BEl-abbes.csv', 'data_0_1/Bejaia.csv']
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Confusion Matrix and Coefficients per area')

    for i, data_path in enumerate(data_paths):
        data = pd.read_csv(data_path)

        if test_type == 'FWI':
            useful_data = data[['ISI', 'BUI', 'FWI', 'Classes']]
        elif test_type == 'Observations':
            useful_data = data[['Temperature', 'RH', 'Ws', 'Rain', 'Classes']]
        elif test_type == 'All':
            useful_data = data[['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain', 'Classes']]
        else:
            raise ValueError("Invalid test type. Supported types: 'FWI', 'Observations' and 'All'")

        X = useful_data.drop('Classes', axis=1)
        Y = useful_data['Classes']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        logmodel = LogisticRegression(max_iter=1000)
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)

        # Calculating confusion matrix
        y_test = list(y_test)
        tp, tn, fp, fn = confusion_matrix(y_test, predictions)
        print(f'Accuracy: {accuracy(tp, tn, fp, fn)}')

        # Plotting confusion matrix
        plot_confusion_matrix(tp, tn, fp, fn, axs[0, i])
        title_name = data_path.split('/')[1].split('.')[0].replace('_', ' ')
        axs[0, i].set_title(f'Area - {title_name} accuracy: {accuracy(tp, tn, fp, fn):.2f}%')

        # Plotting coefficients
        feature_names = X.columns.tolist()
        plot_coefficients(logmodel, feature_names, axs[1, i])
        axs[1, i].set_title(f'Area - {title_name}')

        if test_type == 'FWI':
            plt.savefig('logregres_fwi.png')
        elif test_type == 'Observations':
            plt.savefig('logregres_obs.png')
        elif test_type == 'All':
            plt.savefig('logregres_all.png')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# This function is used to standardize the data
def standardize_data(data):
    data_copy = data.copy()
    for column in data_copy.columns:
        data_copy[column] = (data_copy[column] - data_copy[column].mean()) / data_copy[column].std()
    return data_copy

# This function is used to test the models based on cross prediction
# Thus the model trained on one area is tested on the other area
def cross_predict_n(n):
    sidi_coefficients = [[], [], [], [], [], [], [], [], [], []]
    bejaia_coefficients = [[], [], [], [], [], [], [], [], [], []]
    sidi_acc_list, bejaia_acc_list = [], []
    for i in range(n):
        data_paths = ['data_0_1/Sidi_BEl-abbes.csv', 'data_0_1/Bejaia.csv']

        data_bejaia = pd.read_csv(data_paths[1])
        data_sidi = pd.read_csv(data_paths[0])

        bejaia_X = data_bejaia[['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain']]
        sidi_X = data_sidi[['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain']]

        bejaia_X, sidi_X = standardize_data(bejaia_X), standardize_data(sidi_X)
        bejaia_Y, sidi_Y = data_bejaia['Classes'], data_sidi['Classes']

        bejaia_X_train, bejaia_X_test, bejaia_y_train, bejaia_y_test = train_test_split(bejaia_X, bejaia_Y, test_size=0.2)
        sidi_X_train, sidi_X_test, sidi_y_train, sidi_y_test = train_test_split(sidi_X, sidi_Y, test_size=0.2)
        bejaia_logmodel, sidi_logmodel = LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)
        bejaia_logmodel.fit(bejaia_X_train, bejaia_y_train)
        sidi_logmodel.fit(sidi_X_train, sidi_y_train)

        sidi_pred = bejaia_logmodel.predict(sidi_X_test)
        bejaia_pred = sidi_logmodel.predict(bejaia_X_test)

        # Calculating confusion matrix
        sidi_y_test, bejaia_y_test = list(sidi_y_test), list(bejaia_y_test)
        sidi_tp, sidi_tn, sidi_fp, sidi_fn = confusion_matrix(sidi_y_test, sidi_pred)
        bejaia_tp, bejaia_tn, bejaia_fp, bejaia_fn = confusion_matrix(bejaia_y_test, bejaia_pred)
        sidi_acc_list.append(accuracy(sidi_tp, sidi_tn, sidi_fp, sidi_fn))
        bejaia_acc_list.append(accuracy(bejaia_tp, bejaia_tn, bejaia_fp, bejaia_fn))

        # store the coefficients
        sidi_coefs = sidi_logmodel.coef_[0]
        bejaia_coefs = bejaia_logmodel.coef_[0]
        for i in range(len(sidi_coefs)):
            sidi_coefficients[i].append(sidi_coefs[i])
            bejaia_coefficients[i].append(bejaia_coefs[i])

    # plot the coefficients
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Cross Predict Accuracy: Sidi Bel-abbes: {np.mean(sidi_acc_list):.2f}% Bejaia: {np.mean(bejaia_acc_list):.2f}%')

    sidi_mean_list = [np.mean(sidi_coefficients[j]) for j in range(len(sidi_coefficients))]
    sidi_std_list = [np.std(sidi_coefficients[j]) for j in range(len(sidi_coefficients))]
    axs[0].bar(range(len(sidi_mean_list)), sidi_mean_list, yerr=sidi_std_list)
    axs[0].set_xticks(range(len(sidi_mean_list)))
    axs[0].set_xticklabels(['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain'], rotation=90)
    axs[0].set_ylabel('Coefficient')
    axs[0].set_xlabel('Feature')
    axs[0].set_title(f'Area - Sidi Bel-abbes')

    bejaia_mean_list = [np.mean(bejaia_coefficients[j]) for j in range(len(bejaia_coefficients))]
    bejaia_std_list = [np.std(bejaia_coefficients[j]) for j in range(len(bejaia_coefficients))]
    axs[1].bar(range(len(bejaia_mean_list)), bejaia_mean_list, yerr=bejaia_std_list)
    axs[1].set_xticks(range(len(bejaia_mean_list)))
    axs[1].set_xticklabels(['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain'], rotation=90)
    axs[1].set_ylabel('Coefficient')
    axs[1].set_xlabel('Feature')
    axs[1].set_title(f'Area - Bejaia')

    plt.tight_layout()
    plt.savefig('logregres_cross_predict.png')
    plt.show()

# This function is used to create the model n times and store the coefficients and accuracy
# datapath: the path of the data
# n: the number of times to create the model
# mode: 'Obs', 'FWI', 'Fuel' or None
# 'Obs': create the model based on the weather observations
# 'FWI': create the model based on the fire behavior index
# 'Fuel': create the model based on the fuel moisture codes
def create_model_n_times(datapath, n, mode=None):
    data = pd.read_csv(datapath)
    if mode == 'Obs':
        useful_data = data[['Temperature', 'RH', 'Ws', 'Rain', 'Classes']]
    elif mode == 'FWI':
        useful_data = data[['ISI', 'BUI', 'FWI', 'Classes']]
    elif mode == 'Fuel':
        useful_data = data[['FFMC', 'DMC', 'DC', 'Classes']]
    else:
        useful_data = data[['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temperature', 'RH', 'Ws', 'Rain', 'Classes']]


    X = useful_data.drop('Classes', axis=1)
    X = standardize_data(X)
    Y = useful_data['Classes']

    # create a list of lists to store all the coefficients
    if mode == 'Obs':
        coefficients = [[], [], [], []]
    elif mode == 'FWI':
        coefficients = [[], [], []]
    elif mode == 'Fuel':
        coefficients = [[], [], []]
    else:
        coefficients = [[], [], [], [], [], [], [], [], [], []]

    accuracy_list = []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        logmodel = LogisticRegression(max_iter=1000)
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)

        # Calculating confusion matrix
        y_test = list(y_test)
        tp, tn, fp, fn = confusion_matrix(y_test, predictions)
        accuracy_list.append(accuracy(tp, tn, fp, fn))

        # store the coefficients
        coefs = logmodel.coef_[0]
        for i in range(len(coefs)):
            coefficients[i].append(coefs[i])

    return coefficients, accuracy_list

# Creates the model n times with only the weather obesetvations
def wheather_obs():
    data_paths = ['data_0_1/Both.csv', 'data_0_1/Sidi_BEl-abbes.csv', 'data_0_1/Bejaia.csv']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weather Observations')

    for i, data_path in enumerate(data_paths):
        coefs, acc = create_model_n_times(data_path, 1000, 'Obs')
        mean_list = [np.mean(coefs[j]) for j in range(len(coefs))]
        std_list = [np.std(coefs[j]) for j in range(len(coefs))]
        axs[0, i].bar(range(len(mean_list)), mean_list, yerr=std_list)
        axs[0, i].set_xticks(range(len(mean_list)))
        axs[0, i].set_xticklabels(['Temperature', 'RH', 'Ws', 'Rain'], rotation=90)
        axs[0, i].set_ylabel('Coefficient')
        axs[0, i].set_xlabel('Feature')
        title_name = data_path.split('/')[1].split('.')[0].replace('_', ' ')
        axs[0, i].set_title(f'Area - {title_name}')

        acc_mean, acc_std = np.mean(acc), np.std(acc)
        acc_ci = 1.96 * acc_std / np.sqrt(len(acc))
        axs[1, i].hist(acc, bins=10, label=f'Accuracy')
        axs[1, i].axvline(x=acc_mean, color='red', label=f'Mean')
        axs[1, i].axvline(x=acc_mean + acc_ci, color='red', linestyle='--', label=f'95% CI')
        axs[1, i].axvline(x=acc_mean - acc_ci, color='red', linestyle='--')
        axs[1, i].set_xlabel('Accuracy')
        axs[1, i].set_ylabel('Frequency')
        axs[1, i].set_title(f'Area - {title_name}')
        axs[1, i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('logregres_obs.png')
    plt.show()

# creates the model n times with only the fuel moisture codes
def fuel():
    data_paths = ['data_0_1/Both.csv', 'data_0_1/Sidi_BEl-abbes.csv', 'data_0_1/Bejaia.csv']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fuel moisture codes')

    for i, data_path in enumerate(data_paths):
        coefs, acc = create_model_n_times(data_path, 1000, 'Fuel')
        mean_list = [np.mean(coefs[j]) for j in range(len(coefs))]
        std_list = [np.std(coefs[j]) for j in range(len(coefs))]
        axs[0, i].bar(range(len(mean_list)), mean_list, yerr=std_list)
        axs[0, i].set_xticks(range(len(mean_list)))
        axs[0, i].set_xticklabels(['FFMC', 'DMC', 'DC'], rotation=90)
        axs[0, i].set_ylabel('Coefficient')
        axs[0, i].set_xlabel('Feature')
        title_name = data_path.split('/')[1].split('.')[0].replace('_', ' ')
        axs[0, i].set_title(f'Area - {title_name}')

        acc_mean, acc_std = np.mean(acc), np.std(acc)
        acc_ci = 1.96 * acc_std / np.sqrt(len(acc))
        axs[1, i].hist(acc, bins=10, label=f'Accuracy')
        axs[1, i].axvline(x=acc_mean, color='red', label=f'Mean')
        axs[1, i].axvline(x=acc_mean + acc_ci, color='red', linestyle='--', label=f'95% CI')
        axs[1, i].axvline(x=acc_mean - acc_ci, color='red', linestyle='--')
        axs[1, i].set_xlabel('Accuracy')
        axs[1, i].set_ylabel('Frequency')
        axs[1, i].set_title(f'Area - {title_name}')
        axs[1, i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('logregres_fuel.png')
    plt.show()

# Creates the model n times with only the fire behavior index
def fire():
    data_paths = ['data_0_1/Both.csv', 'data_0_1/Sidi_BEl-abbes.csv', 'data_0_1/Bejaia.csv']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fire behavior index')

    for i, data_path in enumerate(data_paths):
        coefs, acc = create_model_n_times(data_path, 1000, 'FWI')
        mean_list = [np.mean(coefs[j]) for j in range(len(coefs))]
        std_list = [np.std(coefs[j]) for j in range(len(coefs))]
        axs[0, i].bar(range(len(mean_list)), mean_list, yerr=std_list)
        axs[0, i].set_xticks(range(len(mean_list)))
        axs[0, i].set_xticklabels(['ISI', 'BUI', 'FWI'], rotation=90)
        axs[0, i].set_ylabel('Coefficient')
        axs[0, i].set_xlabel('Feature')
        title_name = data_path.split('/')[1].split('.')[0].replace('_', ' ')
        axs[0, i].set_title(f'Area - {title_name}')

        acc_mean, acc_std = np.mean(acc), np.std(acc)
        acc_ci = 1.96 * acc_std / np.sqrt(len(acc))
        axs[1, i].hist(acc, bins=10, label=f'Accuracy')
        axs[1, i].axvline(x=acc_mean, color='red', label=f'Mean')
        axs[1, i].axvline(x=acc_mean + acc_ci, color='red', linestyle='--', label=f'95% CI')
        axs[1, i].axvline(x=acc_mean - acc_ci, color='red', linestyle='--')
        axs[1, i].set_xlabel('Accuracy')
        axs[1, i].set_ylabel('Frequency')
        axs[1, i].set_title(f'Area - {title_name}')
        axs[1, i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('logregres_fire.png')
    plt.show()


if __name__ == "__main__":
    # test_RFE('data_0_1/Both.csv')
    test_classification('All')
    # wheather_obs()
    # fire()
    # fuel()


