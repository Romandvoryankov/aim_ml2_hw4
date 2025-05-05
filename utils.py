import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_df_info(df, thr=0.5):
    '''
    input:
    - df -- исходный датафрейм
    - thr -- доля одинаковых значений после которой мы считаем эти значения мусором

    returns:
    - pd.DataFrame со сводной информацией о датасете

    описание расчитываемых статистик:
    - индекс - все колонки входного датафрейма
    - тип данных
    - количество уникальных элементов (включая наны)
    - доля нанов в колонке
    - доля нулей в колонке
    - доля пустых строк в колонке
    - доля самого частовстречаемого элемента в колонке + сам этот элемент (исключая наны)
    - два разных примера содержимого колонки (исключая наны)        
    - `trash_score` колонки: max([суммарная доля нанов, нулей и пустых строк]
    '''
    # Сначала заполняем датасет по колонкам, затем разворачиваем его превращяя колонки в нужные строки
    df_info = pd.DataFrame()
    for feauture in df.columns.to_list():
        df_len = df[feauture].shape[0]
        null_count = df[feauture].isna().sum()
        zero_count = sum(df[feauture]==0)
        empty_string =sum(df[feauture]=='')
        sample_values = list(df.dropna()[feauture].unique()[:2])
        if(len(sample_values)<2):
            sample_values = sample_values + ['Нет второго значения']

        df_info[feauture] = [
            df[feauture].dtype.name,
            df[feauture].nunique(dropna=False),
            null_count/df_len,
            zero_count/df_len,
            empty_string/df_len,
            df[feauture].value_counts().index[0],
            df[feauture].value_counts(normalize=True).values[0],
            sample_values,
            max(
                [
                 df[df[feauture].isna() | df[feauture].isin(['', 0])].shape[0],
                 df[feauture].value_counts().values[0] if (df[feauture].value_counts(normalize=True).values[0] >= thr) else 0
                ]
               )/df_len
        ]
    df_info = df_info.T.reset_index()
    df_info.columns=['name', 'dtype', 'nunique', 'nan_prop', 'zero_prop', 'empty_str_prop', 'vc_max', 'vc_max_prop', 'example', 'trash_score']
    df_info = df_info.set_index('name')
    df_info[[
        'nan_prop' , 'zero_prop', 'empty_str_prop',
        'vc_max_prop', 'trash_score']] = df_info[[
                                    'nan_prop' , 'zero_prop', 'empty_str_prop',
                                    'vc_max_prop', 'trash_score']].astype(float).replace({0: -100}).round(3)
    return df_info


def my_beeswarm(df,
                features,
                shap_values,
                cat_features=[],
                cat_feature_threshold=0.001,
                top_k=13,
                figsize=(10, 6),
                dots=1000,
                lower_perc=1,
                upper_perc=99
                ):
    """
    Создает beeswarm-график для визуализации SHAP-значений с поддержкой числовых и категориальных признаков.

    Параметры:
    -----------
    df : pandas.DataFrame
        Входной датафрейм с признаками.
    features : list или array-like
        Список названий признаков для отображения.
    shap_values : shap.Explanation или аналогичный
        Объект SHAP-значений с атрибутом .values.
    cat_features : list, optional
        Список категориальных признаков (по умолчанию: []). Нечисловые признаки добавляются автоматически.
    cat_feature_threshold : float, optional
        Минимальная доля для отображения значений категориальных признаков (по умолчанию: 0.001).
    top_k : int, optional
        Количество топ-признаков для отображения, сортировка по среднему абсолютному SHAP (по умолчанию: 13).
    figsize : tuple, optional
        Размер графика в дюймах (по умолчанию: (10, 6)).
    dots : int, optional
        Максимальное количество точек на признак (по умолчанию: 1000).
    lower_perc : int, optional
        Нижний перцентиль для обрезки выбросов числовых признаков (по умолчанию: 1).
    upper_perc : int, optional
        Верхний перцентиль для обрезки выбросов числовых признаков (по умолчанию: 99).

    """
    from pandas.api.types import is_numeric_dtype

    features = np.array(features)
    # Добавляем к кастомному списку категориальных фичей все категориальные фичи из датасета
    cat_features = list(set(cat_features) | set(features[[not is_numeric_dtype(df[f]) for f in features]]))

    # Создаем массив словарей под каждую фичу с данными для отрисовки
    plot_data = []

    for i, feature_name in enumerate(features):
        feature_values = df[feature_name].to_numpy()
        shapley_values = shap_values.values[:, i]
        nan_mask = df[feature_name].isna().to_numpy()

        if feature_name not in cat_features:
            # Обрабатываем некатегориальные фичи
            plot_data.append({
              'cat': False,
              'feature_name': feature_name,
              'feature_values': feature_values[np.invert(nan_mask)],
              'shap_values': shapley_values[np.invert(nan_mask)],
              'nans_shap_values': shapley_values[nan_mask],
              'mean_abs_shap': np.abs(shapley_values).mean()
              })
        else:
            # Обрабатываем пары категориальная фича - значение
            for unique_value in np.unique(feature_values[np.invert(nan_mask)]):
                value_mask = feature_values == unique_value
                plot_data.append({
                  'cat': True,
                  'feature_name': feature_name + f' == {unique_value}',
                  'shap_values': shapley_values[value_mask],
                  'mean_abs_shap': np.abs(shapley_values[value_mask]).mean(),
                  'cat_prop': value_mask.sum()/feature_values.shape[0]
                  })
            # Отдельно добавляем нан для категориальных фич
            if nan_mask.sum() > 0:
                plot_data.append({
                  'cat': True,
                  'feature_name': feature_name + ' == Nan',
                  'shap_values': shapley_values[nan_mask],
                  'mean_abs_shap': np.abs(shapley_values[nan_mask]).mean()
                })

    # Сортируем по среднему значению shap_values
    plot_data.sort(key=lambda x: x['mean_abs_shap'])

    # Создаем отдельный массив словарей для того чтобы предобработать его перед отрисовкой
    prepared_plot_data = list(filter(lambda x: x.get('cat_prop', 1) > cat_feature_threshold, plot_data))

    # Возьмем топ к фич
    prepared_plot_data = prepared_plot_data[-top_k:]

    #  Семплируем точки для отрисовки
    for feature_data in prepared_plot_data:
      values_count = feature_data['shap_values'].shape[0]
      if values_count > dots:
          sample_mask = np.random.permutation(np.repeat([True, False], [dots, values_count-dots]))
          feature_data['shap_values'] = feature_data['shap_values'][sample_mask]
          if not feature_data['cat']:
              feature_data['feature_values'] = feature_data['feature_values'][sample_mask]

      if not feature_data['cat']:
          nans_count = feature_data['nans_shap_values'].shape[0]
          if nans_count > dots:
              nan_sample_mask = np.random.permutation(np.repeat([True, False], [dots, nans_count-dots]))
              feature_data['nans_shap_values'] = feature_data['nans_shap_values'][nan_sample_mask]

    # Убираем выбросы и приводим значение фичи в диапазон [0, 1] равномерного распределения
    for feature_data in prepared_plot_data:
        if not feature_data['cat']:
            q_low = np.percentile(feature_data['feature_values'], lower_perc)
            q_up = np.percentile(feature_data['feature_values'], upper_perc)
            mask = np.array((feature_data['feature_values'] >= q_low) & (feature_data['feature_values'] <= q_up))
            feature_data['feature_values'] = feature_data['feature_values'][mask]
            feature_data['shap_values'] = feature_data['shap_values'][mask]

            feature_data['feature_values'] = feature_data['feature_values'].argsort().argsort()/(feature_data['feature_values'].shape[0]-1)


    # Визуализация
    plt.figure(figsize=figsize)
    for i, feature_data in enumerate(prepared_plot_data):
        # Отрисовываем strip plot с jitter
        jitter = np.random.uniform(-0.2, 0.2, len(feature_data['shap_values'])) + i
        if feature_data['cat']:
          plt.scatter(feature_data['shap_values'],
            jitter,
            alpha=0.3,
            c='green',
            s=7
            )
        else:
          scatter = plt.scatter(feature_data['shap_values'],
            jitter,
            alpha=0.3,
            c=feature_data['feature_values'],
            cmap='coolwarm',
            s=7
            )
        # Отрисовываем strip plot для Nan-ов
        if not feature_data['cat'] and feature_data['nans_shap_values'].shape[0] > 0:
            plt.scatter(feature_data['nans_shap_values'],
                        [i] * len(feature_data['nans_shap_values']),
                        c='black',
                        marker='|',
                        s=5
                        )
    # plt.colorbar(scatter)

    # Настройки графика
    plt.yticks(range(len(prepared_plot_data)), [d['feature_name'] for d in prepared_plot_data])
    plt.title('Jitter Plot', fontsize=14)
    plt.xlabel('shap_values', fontsize=12)
    plt.ylabel('features', fontsize=12)
    # plt.legend()
    plt.axvline(x=0, color='gray', linestyle='-')
    plt.grid(axis='y', linestyle='-', alpha=0.4)

    plt.show()