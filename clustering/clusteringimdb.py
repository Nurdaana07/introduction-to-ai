import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Загружаем данные
movies_data = pd.read_csv('movies_overview.csv')
genres_data = pd.read_csv('movies_genres.csv')

# Преобразуем genre_ids из строки в список
movies_data['genre_ids'] = movies_data['genre_ids'].apply(ast.literal_eval)

# Выбираем только фильмы с одним жанром
movies_df = movies_data[movies_data['genre_ids'].apply(len) == 1].copy()
movies_df['genre_id'] = movies_df['genre_ids'].apply(lambda x: x[0])

# Объединяем с таблицей жанров
movies_df = movies_df.merge(genres_data, left_on='genre_id', right_on='id')
movies_df = movies_df[['title', 'overview', 'name']]
movies_df.rename(columns={'name': 'genre'}, inplace=True)

# Убираем пропуски в overview
movies_df = movies_df.dropna(subset=['overview'])

# # Диагностика: смотрим распределение жанров
# print("Распределение жанров в данных:")
# print(movies_df['genre'].value_counts())

# Преобразуем описания фильмов в числовые вектора (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(movies_df['overview'])

# Кодируем жанры
genres = movies_df['genre'].unique()
genre_mapping = {genre: idx for idx, genre in enumerate(genres)}
y = movies_df['genre'].map(genre_mapping)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель логистической регрессии
model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Балансируем классы
model.fit(X_train, y_train)

# Сохраняем модель, векторайзер и маппинг жанров
joblib.dump(model, 'movie_genre_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(genre_mapping, 'genre_mapping.joblib')
print("Модель, векторайзер и маппинг жанров сохранены в файлы.")

# Делаем предсказания
y_pred = model.predict(X_test)

# Оценка качества
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Получаем только те жанры, которые есть в y_test
unique_test_labels = np.unique(y_test)
test_genres = [genres[label] for label in unique_test_labels]

# # Добавляем диагностику
# print("\nЖанры в тестовой выборке (y_test):", test_genres)
# print("Уникальные предсказанные жанры (y_pred):", [genres[i] for i in np.unique(y_pred)])

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=test_genres, zero_division=0))

# Выводим примеры предсказаний
print("\nПримеры предсказаний (первые 5):")
inverse_genre_mapping = {idx: genre for genre, idx in genre_mapping.items()}
for i in range(min(5, len(y_test))):
    true_genre = inverse_genre_mapping[y_test.iloc[i]]
    pred_genre = inverse_genre_mapping[y_pred[i]]
    print(f"Фильм: {movies_df['title'].iloc[i]}, Истинный жанр: {true_genre}, Предсказанный жанр: {pred_genre}")

# Считаем f1-score для каждого жанра
f1_scores = f1_score(y_test, y_pred, average=None)

# Строим график
plt.figure(figsize=(10, 6))
plt.bar(test_genres, f1_scores, color='skyblue')
plt.title('F1-Score по жанрам')
plt.xlabel('Жанр')
plt.ylabel('F1-Score')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()