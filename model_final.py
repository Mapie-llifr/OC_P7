# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# Garbage Collector
import gc

# Dealing with NaN
from sklearn.impute import SimpleImputer

# Models
from sklearn.dummy import DummyClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
#from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline

# MLflow
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts
from mlflow.models import infer_signature

# Metrics
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# Model selection
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold  #KFold, 
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

import time
from contextlib import contextmanager
import re
from joblib import dump, load


"""
experiment_id = mlflow.create_experiment(name="Model final") 
print(experiment_id)

"""
experiment_id = 513438992690655157


def impute_numeric_nan(df): 
    
    numeric_columns = [col for col in df.columns if df[col].dtype != 'object']
    for num_col in numeric_columns : 
        strategy_median = df[num_col].median(skipna=True)
        df[num_col] = df[num_col].fillna(value=strategy_median)
    return df


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

"""
# Preprocess application_train.csv
def application_train(num_rows = None, nan_as_category = True):
    # Read data
    df = pd.read_csv(path + 'application_train.csv', nrows= num_rows)
    print("Number of samples: {}".format(len(df)))

    # Remove 4 applications with XNA CODE_GENDER 
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Fill NaN in numerical columns
    df = impute_numeric_nan(df)
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    return df
"""

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv(path + 'bureau.csv', nrows = num_rows)
    # Fill NaN in numerical columns
    bureau = impute_numeric_nan(bureau)
    
    bb = pd.read_csv(path + 'bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv(path + 'previous_application.csv', nrows = num_rows)
    prev = prev.drop(prev[prev['AMT_CREDIT'].isnull()].index)
    prev = impute_numeric_nan(prev)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv(path + 'POS_CASH_balance.csv', nrows = num_rows)
    pos = impute_numeric_nan(pos)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv(path + 'installments_payments.csv', nrows = num_rows)

    # Difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Fill NaN with median strategy
    ins = impute_numeric_nan(ins)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv(path + 'credit_card_balance.csv', nrows = num_rows)
    cc = impute_numeric_nan(cc)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def impute_nan (df) : 
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    df_array = imputer.fit_transform(df)

    df = pd.DataFrame(df_array, columns=df.columns)
    
    return df


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    

# Preprocess application.csv
def application(df, num_rows = None, nan_as_category = True):
    # Read data
    print("Number of samples: {}".format(len(df)))

    # Remove 4 applications with XNA CODE_GENDER 
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Fill NaN in numerical columns
    df = impute_numeric_nan(df)
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    return df


def make_data (df): 
    
    with timer("Process application"):
        df = application(df)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance()
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
        df = impute_nan(df)
    with timer("Process previous_applications"):
        prev = previous_applications()
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
        df = impute_nan(df)
    with timer("Process POS-CASH balance"):
        pos = pos_cash()
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
        df = impute_nan(df)
    with timer("Process installments payments"):
        ins = installments_payments()
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
        df = impute_nan(df)
    with timer("Process credit card balance"):
        cc = credit_card_balance()
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
        df = impute_nan(df)
    
    return df

"""
def split_the_data (debug= False):
    df_raw = pd.read_csv(path + 'application_train.csv')
    if debug : 
        df = df_raw.sample(12000)
    else : 
        df = df_raw
    del df_raw
    gc.collect()
    
    app_train, app_test = train_test_split(df, test_size=0.2, random_state=SEED)
    
    print("Making Train set ...")
    df_train = make_data(app_train)
    print("\nTrain set shape: ", df_train.shape)
    print("\nMaking test set ...")
    df_test = make_data(app_test)
    print("\nTest set shape: ", df_test.shape)
    del df
    gc.collect()
    
    return df_train, df_test
"""

def score_function (y_true, y_pred) :

    # calculate inputs for the roc curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # plot roc curve
    plt.plot(fpr, tpr, marker='.')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('roc_curve.png')
    plt.show()
    
    mlflow.log_artifact("roc_curve.png")

    # calculate and print AUROC
    roc_auc = roc_auc_score(y_true, y_pred)
    mlflow.log_metric('Test AUC_score', roc_auc)
    print('AUROC: %.3f' % roc_auc)


def debug_json_characters (df) : 
    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    df = df.rename(columns=new_names)
    return df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances01.png')
    mlflow.log_artifact("lgbm_importances01.png")
    
  
def kfold_lightgbm(experiment_id= experiment_id, num_folds= 5, debug= False):
    
    print("Pre-processing the data ...")
    mlflow.autolog()
    mlflow.start_run(experiment_id=experiment_id, run_name="LightGBM_final")
    
    if debug : 
        df = pd.read_csv(path + 'application_train.csv', nrows=80000)
    else : 
        df = pd.read_csv(path + 'application_train.csv')
        
    print("Making Data set ...")
    df = make_data(df)
    #df = pd.read_csv("Docs_projet7/df_model_final.csv")
    print("Data set shape: ", df.shape)
    
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_train = df[feats]
    y_train = df['TARGET']

    print("\nStarting LightGBM ...")
    del df
    gc.collect()
    
    # BUG : Do not support special JSON characters in feature name.
    X_train = debug_json_characters(X_train)
    
    X_train.to_csv("Docs_projet7/X_train_model_final.csv", index=False)

    # Cross validation model
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=SEED)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(X_train.shape[0])
    # Ouf_of_fold_preds are the predictions made on the train set when the row is in the validation set
    ## during cross_validation. 
    feature_importance_df = pd.DataFrame()
    
    # LightGBM parameters found by Bayesian optimization
    ## Documentation : https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    
    clf = LGBMClassifier(
                            n_estimators=10000,
                            learning_rate=0.005,
                            num_leaves=34,
                            colsample_bytree=0.9497036,
                            subsample=0.8715623,
                            max_depth=8,
                            reg_alpha=0.041545473,
                            reg_lambda=0.0735294,
                            min_split_gain=0.0222415,
                            min_child_weight=39.3259775,
                            #nthread=4,
                            #silent=-1,
                            verbose=-1, )

    under = RandomUnderSampler(sampling_strategy=0.2)
    over = SMOTE(sampling_strategy=0.5)
    
    pipe = Pipeline([('undersample', under), ('oversample', over), ('classifier', clf)], 
                    verbose=True)
    with timer("Training the final model"):    
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
            valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]     
        
            pipe.fit(train_x, train_y, 
                     classifier__eval_set= [(train_x, train_y), (valid_x, valid_y)], 
                     classifier__eval_metric= 'auc',
                     classifier__verbose= 0
                     )

            oof_preds[valid_idx] = pipe.predict_proba(valid_x)[:, 1]
        
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = pipe.named_steps['classifier'].feature_name_
            fold_importance_df["importance"] = pipe.named_steps['classifier'].feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            del train_x, train_y, valid_x, valid_y
            gc.collect()
    
    print('Full AUC score %.6f' % roc_auc_score(y_train, oof_preds))
    score_function(y_train, oof_preds)
    # plot feature importance
    display_importances(feature_importance_df)
    # recording model
    dump(pipe, 'pipeline_lightGBM_final.joblib')
    
    mlflow.end_run()
    return feature_importance_df


SEED = 11

# Files are in a folder named : 'Docs_projet7'
path = "./Docs_projet7/"

mlflow.end_run()
with timer("Training Final model"):
    feat_import = kfold_lightgbm(num_folds=5, debug=False)


# Make the data for the app : 
#app_train = pd.read_csv(path + 'application_train.csv', nrows=80000)
app_train = pd.read_csv(path + 'application_train.csv')
df_dashboard = make_data(app_train)  #Need to improve by removing the impute
df_dashboard = debug_json_characters(df_dashboard)
df_dashboard.to_csv("Docs_projet7/df_model_final.csv", index=False)
