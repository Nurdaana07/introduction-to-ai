import joblib
import pandas as pd

# Загружаем сохранённые объекты
model = joblib.load('movie_genre_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
genre_mapping = joblib.load('genre_mapping.joblib')

# Обратный маппинг для перевода чисел в названия жанров
inverse_genre_mapping = {idx: genre for genre, idx in genre_mapping.items()}

# Функция для предсказания жанра
def predict_genre(description):
    X_new = vectorizer.transform([description])
    prediction = model.predict(X_new)
    predicted_genre = inverse_genre_mapping[prediction[0]]
    return predicted_genre

# Примеры с более длинными описаниями
new_description1 = ("A group of brave friends embark on a thrilling adventure to save the world "
                   "from an ancient evil force threatening to destroy everything they love.")
predicted_genre1 = predict_genre(new_description1)
print(f"Предсказанный жанр для описания '{new_description1}': {predicted_genre1}")

new_description2 = ("A dark and tragic tale of a lonely man seeking revenge after losing "
                   "everything he loved in a brutal betrayal, navigating a world of crime and despair.")
predicted_genre2 = predict_genre(new_description2)
print(f"Предсказанный жанр для описания '{new_description2}': {predicted_genre2}")