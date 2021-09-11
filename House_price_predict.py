########## İŞ PROBLEMİ ##########
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak, farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi
#projesi gerçekleştirilmek istenmektedir.


########## VERİ SETİ HİKAYESİ ##########
# Ames, Lowa’daki konut evlerinden oluşan bu veriseti içerisinde 79 açıklayıcı değişken bulunduruyor.
# Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz.
# Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır.
# Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri sizin tahmin etmeniz beklenmektedir.


########## DEĞİŞKENLER ##########
# Train veriseti 1460 gözlem ve 81 değişkenden, Test veri seti ise 1459 gözlem biriminden oluşmaktadır.


########## GÖREV ##########


import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from helpers import *



pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

train = pd.read_csv("W9/house_prices/train.csv")
test = pd.read_csv("W9/house_prices/test.csv")
df = train.append(test).reset_index(drop=True)

df.head()

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["Neighborhood"].value_counts()

##################
# Kategorik Değişken Analizi
##################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:  #10'dan fazla sınıfa sahip olan değişkenler
    cat_summary(df, col)


##################
# Sayısal Değişken Analizi
##################

df[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T



##################
# Target Analizi
##################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

def find_correlation(dataframe, numeric_cols, corr_limit=0.45):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

low_corrs, high_corrs = find_correlation(df, num_cols)

######################################
# Data Preprocessing & Feature Engineering
######################################

missing_values_table(df)


na_columns = missing_values_table(df)

# The houses which have PoolArea = 0 also have "PoolQC" = NaN
# I will change "PoolQC" = NaN into "PoolQC" = "NoPool"
# Same for Garage, FirePlace features
# I choose to drop "MiscFeature"
df["MiscFeature"].value_counts()
df.drop(['MiscFeature'], axis = 1, inplace=True)
df.drop(['MiscVal'], axis = 1, inplace=True)

df["PoolQC"].value_counts()
len(df[df["PoolArea"] > 0]["PoolArea"])
df["PoolQC"].fillna('NoPool', inplace=True)

df["Alley"].isnull().sum()
df["Alley"].value_counts()
df["Alley"].fillna('NoAlley', inplace=True)

df["Fence"].value_counts()
df["Fence"].fillna('NoFence', inplace=True)

df["Fireplaces"].value_counts()

df["FireplaceQu"].value_counts()
df["FireplaceQu"].isnull().sum()
df["FireplaceQu"].fillna('NoFirePlace', inplace=True)

df["LotFrontage"].isnull().sum()
df["LotFrontage"].fillna(0, inplace=True)

df[df["GarageCars"].isnull()]["GarageCars"]
df["GarageCars"].value_counts()

df[df["GarageType"].isnull()]["GarageType"]
df["GarageType"].value_counts()
df["GarageType"].fillna('NoGarage', inplace=True)

df[df["GarageFinish"].isnull()]["GarageFinish"]
df["GarageFinish"].value_counts()
df["GarageFinish"].fillna('NoGarage', inplace=True)

df[df["GarageQual"].isnull()]["GarageQual"]
df["GarageQual"].value_counts()
df["GarageQual"].fillna('NoGarage', inplace=True)

df[df["GarageCond"].isnull()]["GarageCond"]
df["GarageCond"].value_counts()
df["GarageCond"].fillna('NoGarage', inplace=True)

df[df["GarageArea"].isnull()]["GarageArea"]
df["GarageArea"].value_counts()
df["GarageArea"].fillna(0, inplace=True)

df[df["GarageYrBlt"].isnull()]["GarageYrBlt"]
df["GarageYrBlt"].value_counts()
df["GarageYrBlt"].fillna(0, inplace=True)

df["MiscVal"].value_counts()

missing_values_table(df)

#########################
# Feature Engineering
#########################

# New Feature-1 => LotArea - 1stFlrSF
df["LotArea"].describe().T
df[["LotArea", "1stFlrSF"]]

df["New_Area"] = df["LotArea"] - df["1stFlrSF"]
df["New_Area"].max()
df[["New_Area", "SalePrice"]]

## New Feature-2 => LotArea / 1stFlrSF
df["New_Area_Ratio"] = 1 - (df["1stFlrSF"] / df["LotArea"]) #bahçe alanının oranı
df["New_Area_Ratio"].max()

## New Feature-3 => OverallQual + OverallCond
df["TotalPoint"] = df["OverallCond"] + df["OverallQual"] #evin mükemmellk durumu

## New Feature-4 => Age Of House When Sold
df["HouseAge_Sold"] = df["YrSold"] - df["YearBuilt"] #ev satıldığında kaç yaşındaydı
df["HouseAge_Now"] = 2021 - df["YearBuilt"] #evin şuanki yaşı

## New Feature-5 => Isınma Kalitesi
df['New_HeatingQC'] = df["HeatingQC"].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) #kategorik değişkenin değerlerini numerik değerler ile değiştiririz
df['New_ExterQual'] = df['ExterQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

df['New_HeatingQC'].corr(df['New_ExterQual']) #bu 2 değişkenin korelasyonunua bakarız

df['HeatingQC_ExterQual'] = df['New_ExterQual'] * df['New_HeatingQC'] #2 değişkeni birbiri ile çarpıştırarak ısınma kalitesi için yeni değişkeni oluştururz

## New Feature-6 => Bitmiş Bodrum alanın Kalitesi
df['Finished_BsmtSF'] = df['TotalBsmtSF'] - df['BsmtUnfSF'] #bitmiş bodrum alanının metre karesi

df['BsmtFinType1'].isnull().sum() #bodrum bitmiş alanın kalitesi değişkeninde 79 tane NA değer var
df["BsmtFinType1"].fillna('NoBasement', inplace=True) #bunları NoBasement isimlenidrmesi ile dolduruyoruz
df['New_BsmtFinType1'] = df["BsmtFinType1"].replace({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NoBasement': 0}) # bodrum bitmiş alanın kalitesini
#numerik değerlere dönüştürüyoruz

scaler = MinMaxScaler(feature_range=(1, 6))
df[["Scaled_Finished_BsmtSF"]] = scaler.fit_transform(df[["Finished_BsmtSF"]]) #bitmiş bodrum alanın metre karesi değerlerini scale ettik

df['Usefull_BsmtSF'] = df['New_BsmtFinType1'] * df["Scaled_Finished_BsmtSF"] # ve bu değerleri çarpıştırarak Bitmiş Bodrum alanın Kalitesi için yeni bir
#değişken türettik


#################
# Rare Encoding
##################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "SalePrice", cat_cols)

df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]


cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", cat_cols)

##################
# Label Encoding & One-Hot Encoding
##################

cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

for col in useless_cols_new:
    cat_summary(df, col)

rare_analyser(df, "SalePrice", useless_cols_new)

for col in useless_cols_new:
    df.drop(col, axis=1, inplace=True)

df.shape

##################
# Missing Values
##################

missing_values_table(df)

test.shape

missing_values_table(train)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0) #na değerlerini median ile doldurduk

######################################
# Modeling
######################################

train_df = df[df['SalePrice'].notnull()] #salesprice'ı boş olmayanlar yani dolu olanlar train set olacak
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1) #saleprice'ı boş olanlar test set olacak

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice']) #burada train süresini hızlandırabilmek için bağımlı değişkeni standartlaştırarak seçiyoruz. Eğer eğitim süresinde
# kayda değer bir fark oluşmayacaksa normal kendi değeriyle, standartlaştırmadan devam edilmelidir.
X = train_df.drop(["Id", "SalePrice"], axis=1)

##################
# Base Models
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


##################
# Hyperparameter Optimization
##################

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y) #bulunan en iyi parametrelerle final modelini oluşturuyoruz

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

#######################################
# Feature Selection
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(final_model, X) #final modelinin değişken önem düzeylerine bakılır.


plot_importance(final_model, X, 20) #20 tanesini görselleştirip bakmak istersek


X.shape

feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})


num_summary(feature_imp, "Value", True)


feature_imp[feature_imp["Value"] > 0].shape #0'dan büyük olan kaç tane feature var

feature_imp[feature_imp["Value"] < 1].shape #1'den küçük olan kaç tane feature var


zero_imp_cols = feature_imp[feature_imp["Value"] < 1]["Feature"].values

#bunu programatik bir şekilde yapmak için:
selected_cols = [col for col in X.columns if col not in zero_imp_cols]  # x columnsta gezilir, zero_imp_cols içerisinde olmayanlar seçilir.
len(selected_cols)


lgbm_model = LGBMRegressor(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y) #bu sefer burada x'lerden selected colları seçiyoruz


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y) #en iyi parametrelerle final modeli kuruyoruz

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
