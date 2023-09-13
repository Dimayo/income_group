# Прогнозирование группы доходов
https://github.com/Dimayo/income_group_project/blob/main/income.ipynb<br>
Библиотеки python: pandas, seaborn, matplotlib, numpy, sklearn

## Цель проекта
Целью данного проекта является прогнозирование экономических параметров группы людей согласно их социально-демографическим характеристикам. Подобные задачи возникают в розничной торговле, финансах и других сферах коммерции и бизнеса.

## Описание проекта
<p>Набор данных содержит демографические и социально-экономические характеристики граждан США по данным переписи 1994 года. Обучающая выборка состоит из шестнадцати столбцов. Первый из них — идентификатор состава (точки данных), следующие четырнадцать — независимые переменные, шестнадцатый — целевая переменная, которую должна предсказать модель.</p><p>Тестовая выборка состоит из пятнадцати столбцов — так же, как и обучающую выборку, но без целевой переменной. Задача состоит в том, чтобы обучить классификационную модель, которая прогнозирует, превысит ли доход человека 50 000 долларов США или нет.</p><p>Используемая метрика – f1-оценка.</p>

## Что было сделано
Была произведена загрузка, проверка данных на пропуски и дубликаты. Объединены по смыслу некоторые значения категориальных признаков, для улучшения качества будущей модели в обучающей и тестовой выборках :

```
df_train['education'].replace(['Preschool','1st-4th','5th-6th',
                              '7th-8th','9th','10th','11th','12th'],
                              'dropout',inplace=True)
df_train['education'].replace(['Some-college','Assoc-acdm','Assoc-voc'],
                               'CommunityCollege',inplace=True)
df_train['education'].replace('Prof-school','Masters',inplace=True)

```
Проведена визуальная проверка выбросов в числовых признаках:
```
for column in num_columns:
    plt.figure()
    sns.boxplot(x_train[column])
    plt.title(column)
```
Выбросы замененны граничными значениями:
```
for column in num_columns:
    x_train.loc[x_train[column] > get_outliers(x_train[column])[1],
                                 column] = get_outliers(x_train[column])[1]
    x_train.loc[x_train[column] < get_outliers(x_train[column])[0],
                                  column] = get_outliers(x_train[column])[0]
    x_test.loc[x_test[column] > get_outliers(x_train[column])[1],
                                column] = get_outliers(x_train[column])[1]
    x_test.loc[x_test[column] < get_outliers(x_train[column])[0],
                                column] = get_outliers(x_train[column])[0]
```
В категриальных признаках знак "?" был заменен на значение "Other" с помощью SimpleImputer:
```
imp_const = SimpleImputer(missing_values='?', strategy='constant', fill_value='Other')
x_train[cat_columns] = imp_const.fit_transform(x_train[cat_columns])
x_test[cat_columns] = imp_const.transform(x_test[cat_columns])
```
Признаки доля которых меньше 5% также заменены на значение "Other":
```
for column in x_train[cat_columns]:
    serie = x_train[column].value_counts()
    serie = serie / serie.sum() * 100
    keep_cats = serie[serie > 5].index
    x_train[column] = np.where(x_train[column].isin(keep_cats), x_train[column], 'Other')
    x_test[column] = np.where(x_test[column].isin(keep_cats), x_test[column], 'Other')
```
Числовые признаки были стандартизированы:
```
scaler = StandardScaler()
x_train[num_columns] = scaler.fit_transform(x_train[num_columns])
x_test[num_columns] = scaler.transform(x_test[num_columns]) 
```
Категориальные признаки преобразованы с помощью One Hot Encoder:
```
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
x_train[ohe.get_feature_names_out()] = ohe.fit_transform(x_train[cat_columns])
x_test[ohe.get_feature_names_out()] = ohe.transform(df_test[cat_columns])
```
Был осуществлен подбор гиперпараметров нескольких моделей с помощью Randomized Search:
```
params = {'n_estimators' : [100, 300, 500],
          'max_depth': np.arange(2, 25, 2),
          'max_features': ['sqrt', 'log2', None],
          'min_samples_leaf': np.arange(1, 10, 1),
          'min_samples_split': np.arange(2, 20, 1),
          'class_weight': ['balanced', 'balanced_subsample', None]}

gs = RandomizedSearchCV(RandomForestClassifier(), params, cv=kf, scoring='f1_micro', n_jobs=-1)
gs.fit(x_train, y_train)

print('Лучшие параметры: ', gs.best_params_)
print('Лучший результат: ', gs.best_score_)
```
## Результат
Лучший результат показала модель случайного леса с f1-score равным 84%:
```
model = RandomForestClassifier(n_estimators=300, min_samples_split=14, min_samples_leaf = 3,
                               max_features ='sqrt', max_depth=20, class_weight=None)
model.fit(x_train, y_train)
```
Далее был создан и сохранен файл с предсказанием на тестовой выборке:
```
df_submission = pd.DataFrame(data= {
    'id': df_test['id'],
    'is_reach': test_pred
})
df_submission.head()
```


