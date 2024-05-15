df = pd.read_excel('StoresData(1).xlsx')
df.head()

df = df.drop_duplicates()
df = df.dropna()

any_missing = df.isnull().any().any() 
print("Are there any missing values? ", any_missing)

df.describe()
df.info()

X = df[['Sales $m', 'Wages $m']]
X = sm.add_constant(X)
y = df['GrossProfit']

# algorithm here.

df['Basket'] = (df['Basket:2013'] + df['Basket:2014']) / 2
df['WagePerStaff'] = df['Wages $m'] / df['No. Staff']
df['gross_profit_ratio'] = df['GrossProfit'] / df['Sales $m']

df_numerical = df[['gross_profit_ratio','WagePerStaff', "Adv.$'000", 'Competitors','HrsTrading', 'Mng-Age', 'Mng-Exp','Mng-Train', 'Union%', 'Car Spaces','Basket','Age (Yrs)']]
df_categorical = df[["Loc'n (Num)", 'State (Num)', 'Sundays (Num)',"Mng-Sex (Num)", 'HomeDel (Num)']]

data_merged = pd.concat([df_numerical, df_categorical], axis=1)

print('the_numerical:')
print(df_numerical.head())
print('the_categorical:')
print(df_categorical.head())

scores = stats.zscore(data_merged) #outliers
outliers = data_merged[(np.abs(scores) > 3).any(axis=1)]

print('outliers:', outliers)

data_merged_clean = data_merged[(np.abs(scores) <= 3).all(axis=1)]
scores_merged_clean = stats.zscore(data_merged_clean)

outliers_merged_clean = data_merged_clean[(np.abs(scores_merged_clean) > 3).any(axis=1)]
any_missing = outliers_merged_clean.isnull().any().any()

print("Are there any missing values? ", any_missing)
print('Clean outliers:', outliers_merged_clean)

data_numerical_clean = data_merged_clean[['gross_profit_ratio','WagePerStaff', "Adv.$'000", 'Competitors','HrsTrading', 'Mng-Age', 'Mng-Exp','Mng-Train', 'Union%', 'Car Spaces','Basket','Age (Yrs)']].reset_index(drop=True)
data_categorical_clean = data_merged_clean[["Loc'n (Num)", 'State (Num)', 'Sundays (Num)',"Mng-Sex (Num)", 'HomeDel (Num)']].reset_index(drop=True)

num_rows = data_numerical_clean.shape[0]
print("Number of rows num: ", num_rows)

any_missing = data_numerical_clean.isnull().any().any()
print("Are there any missing values? ", any_missing)

cat_rows = data_categorical_clean.shape[0]
print("Number of rows num: ", cat_rows)

any_missing = data_categorical_clean.isnull().any().any()

print("Are there any missing values? ", any_missing)

print('Numerical:')
print(df_numerical.head())
print('Categorical:')
print(df_categorical.head())

scaler = MinMaxScaler()
data_numerical_normalized = pd.DataFrame(scaler.fit_transform(data_numerical_clean), columns=data_numerical_clean.columns)
any_missing = data_numerical_normalized.isnull().any().any()

print("Are there any missing values? ", any_missing)

num_rows = data_numerical_normalized.shape[0]
print("Number of rows num: ", num_rows)
print(data_numerical_normalized.head())

data_merged = pd.concat([data_numerical_normalized, data_categorical_clean], axis=1)

num_rows = data_merged.shape[0]
print("Number of rows num: ", num_rows)

any_missing = data_merged.isnull().any().any()
print("Are there any missing values? ", any_missing)
print(data_merged.head())

# alogrithem here.

data_predictors = data_merged.drop('gross_profit_ratio', axis=1)
vif_data = pd.DataFrame()
vif_data["feature"] = data_predictors.columns
vif_data["VIF"] = [variance_inflation_factor(data_predictors.values, i) for i in range(len(data_predictors.columns))]
print(vif_data)

rf = RandomForestRegressor(random_state=42)
rf.fit(data_predictors, data_merged['gross_profit_ratio'])
importances = rf.feature_importances_
feature_importances = pd.DataFrame({"Feature": data_predictors.columns, "Importance": importances})
feature_importances = feature_importances.sort_values("Importance", ascending=False)
print(feature_importances)

X = data_merged.drop('gross_profit_ratio', axis=1)
y = data_merged['gross_profit_ratio']

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=78)
r2_scorer = make_scorer(r2_score)

models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=78)),
    ('Random Forest', RandomForestRegressor(random_state=78)),
    ('XGBoost', XGBRegressor(random_state=78)),
    ('Support Vector Machines', SVR())
]

for name, model in models:
    scores = cross_validate(model, X_train , y_train , cv=5,scoring={'MSE': make_scorer(mean_squared_error, greater_is_better=False),
                            'MAE': make_scorer(mean_absolute_error, greater_is_better=False), 'R2': r2_scorer})

mse_scores = -scores['test_MSE']
mae_scores = -scores['test_MAE']
r2_scores = scores['test_R2']
print(f'{name}:')
print(f'  Mean Squared Error: {mse_scores.mean()} (+/- {mse_scores.std()})')
print(f'  Mean Absolute Error: {mae_scores.mean()} (+/- {mae_scores.std()})')
print(f'  R-squared: {r2_scores.mean()} (+/- {r2_scores.std()})\n')

# algorithm here.
