# -*- coding: utf-8 -*-
# =============================================================================
# Directories and Imports
# =============================================================================
import os
import ast

import numpy as np
import pandas as pd
import itertools
import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from mlxtend import classifier, feature_selection
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.feature_selection import ColumnSelector

project_directory = os.getcwd()
data_directory = os.path.join(project_directory, 'Data')
save_directory = os.path.join(project_directory, 'Results', 'ML')

if not os.path.exists("{}".format(save_directory)):
    print('creating path for saving')
    os.makedirs("{}".format(save_directory))

# =============================================================================
# Set Parameters
# =============================================================================
N_JOBS = -1
N_ITER = 100
SEED = 42
scoring = "f1"
SFS = False
# TODO: Set Parameters
SPLIT = "Median"
crossvalgroups = 1

# =============================================================================
# Functions
# =============================================================================
def set_classifiers(SEED):
    classifiers = [
        (LinearDiscriminantAnalysis(), {
            'solver': ['svd', 'lsqr', 'eigen']
        }),
        (LogisticRegression(random_state=SEED), {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000],
            'solver': ['liblinear'],
            'multi_class': ['auto']
        }),
        (SVC(random_state=SEED), {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'probability': [True]
        }),
        (KNeighborsClassifier(), {
            'n_neighbors': range(1, 9),
            'weights': ['distance'],
            'algorithm': ['ball_tree', 'auto', 'kd_tree'],
            'leaf_size': range(4, 16),
            'n_jobs': [N_JOBS]
        }),
        (RandomForestClassifier(random_state=SEED), {
            'max_depth': [3, 6, 9],
            'max_features': [None],
            'n_estimators': [1, 10, 100],
            'criterion': ['gini', 'entropy']
        }),
        (GaussianNB(), {
        }),
    ]
    clf_names = ['LDA', 'LR', 'SVM', 'KNN', 'RFC', 'GNB']
    return zip(clf_names, classifiers)


def single_classifiers(df, label, features, crossvalgroups=1):
    # df = df_clf
    print(f'Start Classification: {label}')
    df = df.reset_index(drop=True)
    groups = df['Subject']
    df = df.drop(columns='Subject')
    y = df[label]

    X = df[sum(list(features.values()), [])].copy()

    print('Shape of data:', X.shape)
    print('Shape of data:', y.shape)

    # Train-Test-Split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groups)

    i = 0
    for train_index, test_index in logo.split(X, y, groups):
        # if i <= 9:
        #     i += 1
        #     continue
        # train_index = groups.loc[groups != 2].index.tolist()
        # test_index = groups.loc[groups == 2].index.tolist()
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index, :].reset_index(drop=True), X.loc[test_index, :]
        y_train, y_test = y.loc[train_index].reset_index(drop=True), y.loc[test_index]
        groups_train = groups.loc[train_index].reset_index(drop=True)

        for modality, features_modality in zip(features.keys(), features.values()):
            # modality = 'brain'
            # features_modality = brain_features
            # if (modality == 'physiology') | (modality == 'brain'):
            #     continue
            print(modality)
            print(features_modality)

            dict_classifiers = set_classifiers(SEED)

            for est_idx, (name, (estimator, params)) in enumerate(dict_classifiers):
                # names, classifiers = zip(*dict_classifiers)
                # est_idx = 2
                # name = names[est_idx]
                # estimator, params = classifiers[est_idx]
                print("Estimator", name)
                start = datetime.datetime.now()
                param_grid = {}
                # Pipeline
                if modality == 'performance':
                    pipe = Pipeline([('scaler', StandardScaler()),
                                     ('columnselector', feature_selection.ColumnSelector()),
                                     ('clf', estimator)])
                else:
                    if SFS:
                        pipe = Pipeline([('scaler', StandardScaler()),
                                         ('columnselector', feature_selection.ColumnSelector()),
                                         ('sfs', SequentialFeatureSelector(estimator, scoring=scoring)),
                                         ('clf', estimator)])
                        if modality == 'brain':
                            steps = [2 ** j for j in range(1, len(features_modality) - 1) if 2 ** j < 20]
                            steps = steps[::2]
                        else:
                            steps = [2 ** j for j in range(1, len(features_modality) - 1) if
                                     2 ** j < len(features_modality) - 1]

                        param_grid['sfs__n_features_to_select'] = steps
                    else:
                        pipe = Pipeline([('scaler', StandardScaler()),
                                         ('columnselector', feature_selection.ColumnSelector()),
                                         ('clf', estimator)])

                column_indices = [X.columns.get_loc(col) for col in features_modality]
                param_grid['columnselector__cols'] = [tuple(column_indices)]

                keys = list(params.keys())
                for key in keys:
                    param_grid['clf__' + key] = params[key]

                # Cross-validated Grid Search
                lpgo = LeavePGroupsOut(n_groups=crossvalgroups)
                candidates = np.prod([len(item) for item in param_grid.values()])
                n_iter = N_ITER if candidates > N_ITER else candidates
                gs = RandomizedSearchCV(pipe, param_grid, cv=lpgo, n_iter=n_iter, n_jobs=N_JOBS, verbose=10,
                                        refit=scoring, scoring=scoring, random_state=SEED)
                gs.fit(X_train, y_train, groups=groups_train)

                model = gs.best_estimator_
                params = gs.best_params_

                end = datetime.datetime.now()

                train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
                train_f1 = f1_score(y_train, model.predict(X_train), average='binary')
                train_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_train, model.predict(X_train)).ravel())))

                test_acc = balanced_accuracy_score(y_test, model.predict(X_test))
                test_f1 = f1_score(y_test, model.predict(X_test), average='binary')
                test_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_test, model.predict(X_test)).ravel())))

                df_results = pd.DataFrame(
                    [[name, 'Standard Scaler', crossvalgroups, label, SPLIT, modality, groups.unique()[i], SFS,
                      train_f1, test_f1, train_acc, test_acc, train_cm, test_cm, params, end - start]],
                    columns=['classifier', 'scaler', 'validation set', 'label', 'split', 'modality', 'subject', 'SFS',
                             'train_f1', 'test_f1', 'train_acc', 'test_acc', 'train_confusion_matrix',
                             'test_confusion_matrix', 'parameter_estimator', 'fit_time'])
                df_results.to_csv(os.path.join(save_directory, 'classification_results_unimodal.csv'),
                                  sep=';', decimal=',', header=False, mode='a', index=False)

                print("Finished Estimator:", name)
        i += 1


def create_pipelines(modality, subject):
    pipes = []
    names = []
    dict_classifiers = set_classifiers(SEED)
    for est_idx, (name, (estimator, params)) in enumerate(dict_classifiers):
        # names, classifiers = zip(*dict_classifiers)
        # est_idx = 0
        # name = names[est_idx]
        # estimator, params = classifiers[est_idx]

        df_params = pd.read_csv(os.path.join(save_directory, 'classification_results_unimodal.csv'), sep=';', decimal=',')
        df_params = df_params.loc[df_params['split'] == SPLIT]
        df_params = df_params.loc[df_params['validation set'] == crossvalgroups]
        df_params = df_params.loc[df_params['subject'] == subject]
        df_params = df_params.loc[df_params['classifier'] == name]
        df_params = df_params.loc[df_params['modality'] == modality].reset_index(drop=True)
        parameter = ast.literal_eval(df_params.at[0, 'parameter_estimator'])

        print("Estimator", name)
        names.append(name)
        # Pipeline
        if (modality == 'performance') | (modality == 'state'):
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('columnselector', ColumnSelector()),
                             ('clf', estimator)])
        else:
            if SFS:
                pipe = Pipeline([('scaler', StandardScaler()),
                                 ('columnselector', feature_selection.ColumnSelector()),
                                 ('sfs', SequentialFeatureSelector(estimator, scoring=scoring)),
                                 ('clf', estimator)])
                pipe.set_params(**{'sfs__n_features_to_select': parameter['sfs__n_features_to_select']})
            else:
                pipe = Pipeline([('scaler', StandardScaler()),
                                 ('columnselector', feature_selection.ColumnSelector()),
                                 ('clf', estimator)])

        pipe.set_params(**{'columnselector__cols': parameter["columnselector__cols"]})

        keys = list(params.keys())
        for key in keys:
            pipe.set_params(**{'clf__' + key: parameter["clf__" + key]})

        pipes.append(pipe)

    return pipes, names


def fit_unimodal_voting(modality, X_train, y_train, X_test, y_test, groups, crossvalgroups, subject):
    # groups = groups_train
    # n_iter = 5
    parameters = {}

    start = datetime.datetime.now()
    pipes, names = create_pipelines(modality, subject)

    voting_pipe = Pipeline([("um_voting", EnsembleVoteClassifier(pipes))])
    weights = [list(item) for item in itertools.product(range(0, 3), repeat=len(pipes))
               if not ((sum(item) == 0) | (sum(item) == 6) | item.count(2) == 1) and item.count(1) >= 1 and item.count(1) > item.count(2)]
    parameters['um_voting__voting'] = ['soft', 'hard']
    parameters['um_voting__weights'] = weights

    # Cross-Validated Grid Search
    lpgo = LeavePGroupsOut(n_groups=crossvalgroups)
    candidates = np.prod([len(item) for item in parameters.values()])
    n_iter = N_ITER if candidates > N_ITER else candidates
    gs = RandomizedSearchCV(voting_pipe, parameters, cv=lpgo, n_iter=n_iter, n_jobs=N_JOBS, verbose=10,
                      scoring=scoring, error_score='raise', return_train_score=True, random_state=SEED)
    try:
        gs.fit(X_train, y_train, groups=groups)

        model = gs.best_estimator_
        model_params = gs.best_params_

        weights_classifier = [(name, weight) for name, weight in zip(names, model_params['um_voting__weights'])]
        model_params['um_voting__weights'] = weights_classifier

        # Unimodal - Combination of Classifiers:
        train_f1 = gs.cv_results_['mean_train_score'][gs.best_index_]
        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
        train_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_train, model.predict(X_train)).ravel())))

        test_f1 = f1_score(y_test, model.predict(X_test), average='binary')
        test_acc = balanced_accuracy_score(y_test, model.predict(X_test))
        test_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_test, model.predict(X_test)).ravel())))

        end = datetime.datetime.now()
        df_results = pd.DataFrame(
            [['Unimodal Voting', 'Standard Scaler', crossvalgroups, label, SPLIT, modality, subject, SFS,
              train_f1, test_f1, train_acc, test_acc, train_cm, test_cm, model_params, end - start]],
            columns=['classifier', 'scaler', 'validation set', 'label', 'split', 'modality', 'subject', 'SFS',
                     'train_f1', 'test_f1', 'train_acc', 'test_acc', 'train_confusion_matrix',
                     'test_confusion_matrix', 'parameter_estimator', 'fit_time'])
        df_results.to_csv(os.path.join(save_directory, 'classification_results_voting.csv'),
                          sep=';', decimal=',', header=False, mode='a', index=False)
        return model
    except Exception as e:
        print(f"Modality {modality} failed.")
        print(e)


def fit_multimodal_voting(models, X_train, y_train, X_test, y_test, groups, crossvalgroups, subject):
    # groups = groups_train
    start = datetime.datetime.now()
    parameters = {}
    voting_pipe = Pipeline([("mm_voting", EnsembleVoteClassifier(list(models.values())))])
    weights = [list(item) for item in itertools.product(range(0, 3), repeat=len(models.values()))
               if not ((sum(item) == 0) | (sum(item) == len(models))) and item.count(1) >= 1 and item.count(1) < 3]
    parameters['mm_voting__voting'] = ['soft', 'hard']
    parameters['mm_voting__weights'] = weights

    # Cross-Validated Grid Search
    lpgo = LeavePGroupsOut(n_groups=crossvalgroups)
    candidates = np.prod([len(item) for item in parameters.values()])
    n_iter = N_ITER if candidates > N_ITER else candidates
    gs = RandomizedSearchCV(voting_pipe, parameters, cv=lpgo, n_iter=n_iter, n_jobs=N_JOBS, verbose=10,
                            scoring=scoring, error_score='raise', return_train_score=True, random_state=SEED)
    try:
        gs.fit(X_train, y_train, groups=groups)

        model_multimodal = gs.best_estimator_
        model_multimodal_params = gs.best_params_

        weights_classifier = [(name, weight) for name, weight in
                              zip(models.keys(), model_multimodal_params['mm_voting__weights'])]
        model_multimodal_params['mm_voting__weights'] = weights_classifier

        train_acc = balanced_accuracy_score(y_train, model_multimodal.predict(X_train))
        train_f1 = gs.cv_results_['mean_train_score'][gs.best_index_]
        train_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_train, model_multimodal.predict(X_train)).ravel())))

        test_acc = balanced_accuracy_score(y_test, model_multimodal.predict(X_test))
        test_f1 = f1_score(y_test, model_multimodal.predict(X_test), average='binary')
        test_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_test, model_multimodal.predict(X_test)).ravel())))

        end = datetime.datetime.now()
        df_results = pd.DataFrame(
            [['Multimodal Voting', 'Standard Scaler', crossvalgroups, label, SPLIT, 'all', subject, SFS,
              train_f1, test_f1, train_acc, test_acc, train_cm, test_cm, model_multimodal_params, end - start]],
            columns=['classifier', 'scaler', 'validation set', 'label', 'split', 'modality', 'subject', 'SFS',
                     'train_f1', 'test_f1', 'train_acc', 'test_acc', 'train_confusion_matrix',
                     'test_confusion_matrix', 'parameter_estimator', 'fit_time'])
        df_results.to_csv(os.path.join(save_directory, 'classification_results_voting.csv'),
                          sep=';', decimal=',', header=False, mode='a', index=False)
        return model_multimodal
    except Exception as e:
        print(f"Multimodal Voting failed.")
        print(e)


def voting_classifiers(df, label, features, crossvalgroups=1):
    # df = df_clf
    print(f'Start Classification: {label}')
    df = df.reset_index(drop=True)
    groups = df['Subject']
    df = df.drop(columns='Subject')
    y = df[label]

    X = df[sum(list(features.values()), [])].copy()

    print('Shape of data:', X.shape)
    print('Shape of data:', y.shape)

    # Train-Test-Split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groups)

    i = 0
    for train_index, test_index in logo.split(X, y, groups):
        # if i < 15:
        #     i += 1
        #     continue

        # train_index = groups.loc[groups != 2].index.tolist()
        # test_index = groups.loc[groups == 2].index.tolist()
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index, :].reset_index(drop=True), X.loc[test_index, :]
        y_train, y_test = y.loc[train_index].reset_index(drop=True), y.loc[test_index]
        groups_train = groups.loc[train_index].reset_index(drop=True)

        models = {}

        for modality, features_modality in zip(features.keys(), features.values()):
            # modality = 'physiology'
            # features_modality = physio_features
            #if modality == 'performance':
                #continue
            print(modality)
            print(features_modality)

            # Unimodal Classifier Based on Combined Models (with Voting Classifier)
            models[modality] = fit_unimodal_voting(modality, X_train, y_train, X_test, y_test, groups=groups_train,
                                                   crossvalgroups=crossvalgroups, subject=groups.unique()[i])

        # Multimodal Classifier Based on Unimodal Classifiers (with Voting Classifier)
        multimodal_model = fit_multimodal_voting(models, X_train, y_train, X_test, y_test, groups=groups_train,
                                                 crossvalgroups=crossvalgroups, subject=groups.unique()[i])

        # Dummy Classifier
        start = datetime.datetime.now()
        dummy = DummyClassifier(strategy="stratified", random_state=SEED)
        dummy.fit(X_test, y_test)

        train_acc = balanced_accuracy_score(y_train, dummy.predict(X_train))
        train_f1 = f1_score(y_train, dummy.predict(X_train), average='binary')
        train_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_train, dummy.predict(X_train)).ravel())))
        print('Train Accuracy:', round(train_acc, 3), 'Train F1-Score:', round(train_f1, 3))

        test_acc = balanced_accuracy_score(y_test, dummy.predict(X_test))
        test_f1 = f1_score(y_test, dummy.predict(X_test), average='binary')
        test_cm = dict(zip(['tn', 'fp', 'fn', 'tp'], list(confusion_matrix(y_test, dummy.predict(X_test)).ravel())))
        print('Test Accuracy:', round(test_acc, 3), 'Test F1-Score:', round(test_f1, 3))

        end = datetime.datetime.now()
        df_results = pd.DataFrame(
            [['Dummy', '', crossvalgroups, label, SPLIT, 'all', groups.unique()[i], SFS,
              train_f1, test_f1, train_acc, test_acc, train_cm, test_cm, '', end - start]],
            columns=['classifier', 'scaler', 'validation set', 'label', 'split', 'modality', 'subject', 'SFS',
                     'train_f1', 'test_f1', 'train_acc', 'test_acc', 'train_confusion_matrix',
                     'test_confusion_matrix', 'parameter_estimator', 'fit_time'])
        df_results.to_csv(os.path.join(save_directory, 'classification_results.csv'),
                          sep=';', decimal=',', header=False, mode='a', index=False)
        i += 1

# =============================================================================
# Run Classifier
# =============================================================================
df = pd.read_csv(os.path.join(data_directory, 'master.csv'))
if 'Subject' not in df.columns and 'participant_number' in df.columns:
        df = df.rename(columns={'participant_number':'Subject'})
        
drop = ['Trial', 'Round', 'Emotion Condition',
        # 'Accuracy', 'Speed', 'Nasa TLX - Frustration', 'EmojiGrid - Valence', 'EmojiGrid - Arousal',
        'Nasa TLX - Effort', 'Aggregated Performance Score', 'Agreeableness']
drop = [c for c in drop if c in df.columns]

condition = 'All'  # 'All', 'Auditive Distraction', 'Silence'
if condition == 'Auditive Distraction':
    df = df.loc[df['Emotion Condition'] != 'Silence']
elif condition == 'Silence':
    df = df.loc[df['Emotion Condition'] == 'Silence']

medians = []
quantiles = []

label = 'tutorial_split_feedback_score_subexperiment'
print(f'Label: {label}')
if label == 'Condition' and 'Load Condition' in df.columns:
    drop += ['Load Condition']
elif label == 'Load Condition' and 'Condition' in df.columns:
    drop += ['Condition']
df_clf = df.drop(columns=drop)

brain_features = [c for c in ['O2Hb_highest_peak', 'O2Hb_average_peak','O2Hb_difference_peak', 'O2Hb_auc'] if c in df_clf.columns]
visual_features = [c for c in ['duration_fixation', 'duration_saccade', 'count_fixations', 'count_saccades'] if c in df_clf.columns]
performance_features = [c for c in ['accuracy_subexperiment', 'trial_duration_mean'] if c in df_clf.columns]
# integrate performance?
features = dict(zip(['visual', 'brain', 'performance'],
                    [visual_features, brain_features, performance_features]))
features = {m: cols for m, cols in features.items() if len(cols) > 0}

columns = brain_features + visual_features + performance_features + [label]
df_clf = df_clf.dropna(subset=columns)

if df_clf[label].dtype == 'O':
    df_clf[label] = df_clf[label].str.lower().replace({'low': 0, 'high': 1})
else:
    df_clf[label] = df_clf[label].astype(int)

single_classifiers(df_clf, label, features, crossvalgroups)
df_results = pd.read_csv(os.path.join(save_directory, 'classification_results_unimodal.csv'), header=None, sep=';', decimal=',')
df_results.columns = ['classifier', 'scaler', 'validation set', 'label', 'split', 'modality', 'subject', 'SFS',
                      'train_f1', 'test_f1', 'train_acc', 'test_acc', 'train_confusion_matrix',
                      'test_confusion_matrix', 'parameter_estimator', 'fit_time']
df_results.to_csv(os.path.join(save_directory, 'classification_results_unimodal.csv'), index=False, header=True, decimal=',', sep=';')

voting_classifiers(df_clf, label, features, crossvalgroups)
df = pd.read_csv(os.path.join(save_directory, 'classification_results_voting.csv'), header=None, sep=';', decimal=',')
df.columns = ['classifier', 'scaler', 'validation set', 'label', 'split', 'modality', 'subject', 'SFS', 
              'train_f1', 'test_f1', 'train_acc', 'test_acc', 'train_confusion_matrix', 
              'test_confusion_matrix', 'parameter_estimator', 'fit_time']
df.to_csv(os.path.join(save_directory, 'classification_results_voting.csv'), index=False, header=True, decimal=',', sep=';')
