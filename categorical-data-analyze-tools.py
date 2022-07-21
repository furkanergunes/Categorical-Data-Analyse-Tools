import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

def cat_variables(dataframe,cat_th=10,card_th=10):
    """
    returns the list of categorical variables, numerical variables and categorical but cardinal variables
    Parameters
    ----------
    dataframe # data frame
    cat_th  #  limit for unique element number for a numerical variable to be categorical variable
    card_th # limit for unique element number for a categorical variable to be cardinal variable

    Returns
    -------
    cat_variables,num_variables,cat_but_card_vars
    """
    cat_variables = [col for col in dataframe.columns if dataframe[col].dtype in ["category","bool","object"]]
    num_but_cat_variables = [col for col in dataframe.columns if (dataframe[col].dtype in ["int64","float64"])
                             and dataframe[col].nunique() < cat_th]
    cat_but_card_vars = [col for col in cat_variables if dataframe[col].nunique() > card_th]
    cat_variables = [col for col in cat_variables if col not in cat_but_card_vars]
    cat_variables += num_but_cat_variables
    num_variables = [col for col in dataframe.columns if (dataframe[col].dtype in ["int64","float64"])
                     and (dataframe[col].nunique() > cat_th)]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_variables)}")
    print(f"num_cols: {len(num_variables)}")
    print(f"cat_but_car: {len(cat_but_card_vars)}")
    print(f"num_cat_cols: {len(num_but_cat_variables)}")

    return cat_variables,num_variables,cat_but_card_vars

def cat_summary(dataframe, col_name,plot=False):
    if dataframe[col_name].dtype == 'bool':
        dataframe[col_name] = dataframe[col_name].astype(int)

    print(pd.DataFrame({col_name:df[col_name].value_counts(),
                        "Ratio":100*df[col_name].value_counts()/len(dataframe)}))
    print("###################################")

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)


def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}),end="\n\n\n")
