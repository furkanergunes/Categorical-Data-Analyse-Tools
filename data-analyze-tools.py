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

    
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outlier_thresholds(dataframe,col_name,q1=0.25,q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5*iqr
    up_limit = quartile3 + 1.5*iqr
    return low_limit,up_limit

def check_outlier(dataframe,col_name):
    low,up = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name]>up)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe,col,index=False):
    low,up = outlier_thresholds(dataframe,col)
    if not dataframe[(dataframe[col] < low) | (dataframe[col] > up)].any(axis=None):
        print("There is no outlier")
        return 0
    if dataframe[(dataframe[col] < low) | (dataframe[col] > up)].shape[0] > 10:
        print(dataframe[(dataframe[col] < low) | (dataframe[col] > up)].head())
    else:
        print(dataframe[(dataframe[col] < low) | (dataframe[col] > up)])
    if index:
        return dataframe[(dataframe[col] < low) | (dataframe[col] > up)].index
 

def remove_outlier(dataframe,col):
    low,up = outlier_thresholds(dataframe,col)
    df_without_outliers = dataframe[~((df[col] < low) | (df[col]>up))]
    return df_without_outliers

def replace_with_thresholds(dataframe,col):
    low,up = outlier_thresholds(dataframe,col)
    dataframe.loc[(dataframe[col] > up),col] = up
    dataframe.loc[(dataframe[col] < low),col] = low

    
   
