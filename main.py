# Import libraries necessary for this project
import numpy as np
import pandas as pd
import sys
sys.path.append("C:/Users/fsteffenino/Desktop/Udacity/MLNanodegree/ProjectPractice/titanic_survival_exploration/")


def normalize_cabin(data):
    for i, line in data.iterrows():
        line['Cabin'] = line['Cabin'][0]
        data.set_value(i, 'Cabin', line['Cabin'])
    return data


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"


def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            if passenger['Pclass'] < 3:
                predictions.append(1)
            else:
                if passenger['Age'] < 40 or passenger['Age'] > 59:
                    predictions.append(1)
                else:
                    predictions.append(0)
        else:
            if passenger['Age'] < 10:
                predictions.append(1)
            else:
                predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)


# ----------------------------------------------------
# LOAD DATA and create objects

# Load the dataset in DataFrame object
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)
normalized_data = full_data.copy()

# Store the 'Survived' feature in a new variable and remove it from the working dataset
outcomes = normalized_data['Survived']
# normalized_data = normalized_data.drop('Survived', axis=1)
normalized_data = normalized_data.drop('PassengerId', axis=1)
normalized_data = normalized_data.drop('Ticket', axis=1)
normalized_data['FamilySize'] = 0
normalized_data['FamilySize'] = normalized_data.apply(lambda row: row.SibSp+row.Parch, axis=1)
normalized_data = normalized_data.drop('SibSp', axis=1)
normalized_data = normalized_data.drop('Parch', axis=1)
# ----------------------------------------------------
# ANALYZE data
print("------------------------------")
print("missing values by column -> ")
print(normalized_data.isnull().sum())
print("------------------------------")
print("assign default values to missing ages and cabins")
normalized_data.Embarked.fillna('S', inplace=True)
normalized_data.Cabin.fillna('Z', inplace=True)
normalized_data = normalize_cabin(normalized_data)
medians = dict()
medians['Mr'] = normalized_data[(normalized_data.Name.str.contains('Mr.', regex=False)==True)].median().Age
medians['Mrs'] = normalized_data[(normalized_data.Name.str.contains('Mrs.', regex=False)==True)].median().Age
medians['Miss'] = normalized_data[(normalized_data.Name.str.contains('Miss.', regex=False)==True)].median().Age
medians['Master'] = normalized_data[(normalized_data.Name.str.contains('Master.', regex=False)==True)].median().Age
medians['Dr'] = normalized_data[(normalized_data.Name.str.contains('Dr.', regex=False)==True)].median().Age
print("-----------  normalize names")
normalized_data['Name'] = normalized_data.apply(lambda row: 'Major' if 'Major.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Don' if 'Don.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Master' if 'Master.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Mr' if 'Mr.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Mrs' if 'Mrs.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Miss' if 'Miss.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Dr' if 'Dr.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Mme' if 'Mme.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Rev' if 'Rev.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Ms' if 'Ms.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Lady' if 'Lady.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Sir' if 'Sir.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Mlle' if 'Mlle.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Col' if 'Col.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Capt' if 'Capt.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Countess' if 'Countess.' in row.Name else row.Name, axis=1)
normalized_data['Name'] = normalized_data.apply(lambda row: 'Jonkheer' if 'Jonkheer.' in row.Name else row.Name, axis=1)
print("------------ normalize ages")
normalized_data['Age'] = normalized_data.apply(lambda row: row.Age if row.Age == row.Age else medians[row.Name], axis=1)
normalized_data = normalized_data.drop('Name', axis=1)
print(normalized_data.head())

# ---------------------------------------------------
# PREDICTIONS
predict_result = predictions_3(normalized_data)
print(accuracy_score(outcomes, predict_result))
# ----------------------------------------------------

# print(normalized_data[normalized_data.Sex == 'male'].groupby(['FamilySize', 'Survived']).count())

