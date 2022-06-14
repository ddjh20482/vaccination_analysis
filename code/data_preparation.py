import pandas as pd
import seaborn as sns
import numpy as np

# handling missing values
def missing(all_data_no_missing):
    
    all_data_no_missing.h1n1_concern.fillna(4, inplace=True)
    all_data_no_missing.h1n1_knowledge.fillna(3, inplace=True)
    all_data_no_missing.behavioral_antiviral_meds.fillna(2, inplace=True)
    all_data_no_missing.behavioral_avoidance.fillna(2, inplace=True)
    all_data_no_missing.behavioral_face_mask.fillna(2, inplace=True)
    all_data_no_missing.behavioral_wash_hands.fillna(2, inplace=True)
    all_data_no_missing.behavioral_large_gatherings.fillna(2, inplace=True)
    all_data_no_missing.behavioral_outside_home.fillna(2, inplace=True)
    all_data_no_missing.behavioral_touch_face.fillna(2, inplace=True)
    all_data_no_missing.doctor_recc_h1n1.fillna(2, inplace=True)
    all_data_no_missing.chronic_med_condition.fillna(2, inplace=True)
    all_data_no_missing.child_under_6_months.fillna(2, inplace=True)
    all_data_no_missing.health_worker.fillna(2, inplace=True)
    all_data_no_missing.health_insurance.fillna(2, inplace=True)
    all_data_no_missing.education.fillna('N/A', inplace=True)
    all_data_no_missing.income_poverty.fillna('N/A', inplace=True)
    all_data_no_missing.marital_status.fillna('N/A', inplace=True)
    all_data_no_missing.rent_or_own.fillna('N/A', inplace=True)
    all_data_no_missing.employment_status.fillna('N/A', inplace=True)
    all_data_no_missing.household_adults.fillna(4, inplace=True)
    all_data_no_missing.household_children.fillna(4, inplace=True)
    all_data_no_missing.employment_industry.fillna('N/A', inplace=True)
    all_data_no_missing.employment_occupation.fillna('N/A', inplace=True)
    
    all_data_no_missing.opinion_h1n1_vacc_effective.fillna(3, inplace=True)
    all_data_no_missing.opinion_h1n1_risk.fillna(3, inplace=True)
    all_data_no_missing.opinion_h1n1_sick_from_vacc.fillna(3, inplace=True)
    
    return all_data_no_missing

# creating dummies
def dummy(data, col):
    data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis = 1)
    data.drop(col, axis = 1, inplace = True)
    return data


#Linear Regression definition
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# setting default split values
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report

def scores(X_train, y_train, X_test, y_test, model):

    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    baseline_scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        return_train_score=True,
        cv=splitter
    )

    print("CV score:     {:.2%}".format(baseline_scores["test_score"].mean()))
    print("X-test score: {:.2%}".format(model.score(X_test, y_test)))
    #print("RMSE:         {:.2%}".format(mean_squared_error(y_test, y_hat_test, squared=False)))
    print("RMSE:         {:.4}".format(mean_squared_error(y_test, y_hat_test, squared=False)))
    print("")
    print("Train score")
    #print("Pricision: {:.2%}".format(precision_score(y_train, y_hat_train)))
    #print("Recall:    {:.2%}".format(recall_score(y_train, y_hat_train)))
    #print("Accuracy:  {:.2%}".format(accuracy_score(y_train, y_hat_train)))
    #print("F1:        {:.2%}".format(f1_score(y_train, y_hat_train)))
    print(classification_report(y_train, y_hat_train))
    
    print("")
    print("")
    print("X-test score")
    #print("Pricision: {:.2%}".format(precision_score(y_test, y_hat_test)))
    #print("Recall:    {:.2%}".format(recall_score(y_test, y_hat_test)))
    #print("Accuracy:  {:.2%}".format(accuracy_score(y_test, y_hat_test)))
    #print("F1:        {:.2%}".format(f1_score(y_test, y_hat_test)))
    print("")
    print(classification_report(y_test, y_hat_test))
    
    pass

def scores2(X_train, y_train, X_test, y_test, model):

    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    baseline_scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        return_train_score=True,
        cv=splitter
    )

    print("CV score:     {:.2%}".format(baseline_scores["test_score"].mean()))
    print("X-test score: {:.2%}".format(model.score(X_test, y_test)))
    #print("RMSE:         {:.2%}".format(mean_squared_error(y_test, y_hat_test, squared=False)))
    print("RMSE:         {:.4}".format(mean_squared_error(y_test, y_hat_test, squared=False)))
    print("")
    print("X-test score")
    #print("Pricision: {:.2%}".format(precision_score(y_test, y_hat_test)))
    #print("Recall:    {:.2%}".format(recall_score(y_test, y_hat_test)))
    #print("Accuracy:  {:.2%}".format(accuracy_score(y_test, y_hat_test)))
    #print("F1:        {:.2%}".format(f1_score(y_test, y_hat_test)))
    print("")
    print(classification_report(y_test, y_hat_test))
    
    pass
