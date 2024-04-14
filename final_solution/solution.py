import typing as tp
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import spacy
from joblib import load

clf = load('sentiment_model_finalv1.joblib')

data = pd.read_pickle("sentiment_texts.pickle")

data = data[data['SentimentScore'] != 0]
data = data.reset_index(drop=True)
X = data['MessageText']
y = data['SentimentScore']

data[data['SentimentScore']==0]

data = pd.read_pickle('sentiment_texts.pickle')
data = data[data['SentimentScore'] != 0]
data = data.reset_index(drop=True)
X = data['MessageText']
y = data['SentimentScore']

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Преобразование текстовых данных в матрицу TF-IDF

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

def classify_sentiment(post):
    # Preprocess the raw text

    # Transform the preprocessed text to the format understood by the model (TF-IDF)
    post_vectorized = tfidf_vectorizer.transform([post])

    # Predict the sentiment using the trained model
    sentiment = clf.predict(post_vectorized)

    return sentiment

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]

tfidf_vectorizer2 = TfidfVectorizer(ngram_range=(1, 2))

# Применение TF-IDF к текстовым данным
tfidf_matrix = tfidf_vectorizer2.fit_transform(X)

# Получение списка фичей и их значения TF-IDF для первого документа
feature_names = tfidf_vectorizer2.get_feature_names_out()

df = pd.read_excel("issuers.xlsx")
df = df.drop(['datetrackstart', 'datetrackend'], axis=1)

tfidf_vectorizer2 = TfidfVectorizer(ngram_range=(1, 2))

# Применение TF-IDF к текстовым данным
tfidf_matrix = tfidf_vectorizer2.fit_transform(X)

# Получение списка фичей и их значения TF-IDF для первого документа
feature_names = tfidf_vectorizer2.get_feature_names_out()

# # Предположим, что ваш датасет выглядит так:
df_3 = pd.DataFrame({
    'EMITENT_FULL_NAME': [i for i in df['EMITENT_FULL_NAME']],  # и т.д.
    'BGTicker': [i for i in df['BGTicker']],
    'OtherTicker': [i for i in df['OtherTicker']],
})

# Инициализация TfidfVectorizer
tfidf = TfidfVectorizer()

# Переводим названия компаний и тикеры в строковый формат
df_3 = pd.DataFrame({
    'EMITENT_FULL_NAME': [str(i) for i in df['EMITENT_FULL_NAME']],  # и т.д.
    'BGTicker': [str(i) for i in df['BGTicker']],
    'OtherTicker': [str(i) for i in df['OtherTicker']],# и т.д.
})

# Джоиним название и тикер в одну строку
df_3['combined'] = df_3['EMITENT_FULL_NAME'] + " " + df_3['BGTicker'] + " " + df_3['OtherTicker']

# Создаём матрицу TF-IDF
tfidf_matrix = tfidf.fit_transform(df_3['combined'])

nlp = spacy.load('en_core_web_sm')

def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """

    def extracted_entities(post):
        # Применяем модель к нашему тексту
        doc = nlp(post)

        text_entities = ""

        # Пробегаем по всем найденным сущностям в тексте post
        for entity in doc.ents:
            # Если сущность классифицирована как "ORG"
            if entity.label_ == "ORG":
                # Добавляем её в список найденных сущностей
                text_entities += '\n' + entity.text

        # Делим наш текст на строки и берем первое слово из каждой строки
        entities = [line.split(" ")[0] for line in text_entities.split('\n') if line]

        # Возвращаем список именованных сущностей (ORG)
        return entities

    def calculate_indices(entities, tfidf_matrix):
        # Этот список будет хранить индексы именованных сущностей
        indices = []

        # Проходимся по каждой именованной сущности
        for entity in entities:
            # Трансформируем именованную сущность в tf-idf вектор
            tfidf_entity = tfidf.transform([entity])

            # Вычисляем косинусное сходство между нашим вектором и матрицей tf-idf
            cosine_similarities = linear_kernel(tfidf_entity, tfidf_matrix).flatten()

            # Добавляем индекс в список, если сходство больше нуля
            indices += [i for i, score in enumerate(cosine_similarities) if score > 0]

        indices = list(set(indices))  # Преобразуем список в множество, чтобы убрать дубликаты
        return indices

    result = []

    # Проходимся по каждому сообщению в нашем наборе
    for post in data['MessageText']:
        # Пустой список, в который будем записывать результаты
        score = []

        # Классифицируем сентименты
        sentiment = classify_sentiment(post)

        # Извлекаем именованные сущности из поста
        entities = extracted_entities(post)

        # Вычисляем индексы именованных сущностей
        indices = calculate_indices(entities, tfidf_matrix)


        # Проходимся по каждому индексу и добавляем в score пару [индекс, последний элемент из списка sentiment]
        for i in indices:
            score.append([str(i + 1), str(sentiment[-1])])

        # Добавляем результат в общий список
        result.append(score)

    # Возвращаем итоговый список
    return result
    raise NotImplementedError
