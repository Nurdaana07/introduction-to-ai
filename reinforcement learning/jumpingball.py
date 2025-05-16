import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Инициализация игры
pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Параметры мяча
BALL_RADIUS = 20
ball_x = 100
ball_y = HEIGHT - BALL_RADIUS
ball_vy = 0
GRAVITY = 0.5
JUMP_SPEED = -12

# Параметры препятствий
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 40
obstacle_speed = -3
obstacles = []
SPAWN_INTERVAL = 180  # Препятствия каждые 3 секунды

# Параметры Q-learning
ALPHA = 0.1  # Скорость обучения
GAMMA = 0.9  # Дисконтирование
EPSILON_MAX = 0.1  # Максимальная вероятность случайных действий
EPSILON = EPSILON_MAX  # Текущая вероятность случайных действий
EPSILON_DECAY = 0.995  # Уменьшение EPSILON после эпизода
MIN_EPSILON = 0.01  # Минимальное значение EPSILON
q_table = defaultdict(lambda: np.zeros(2))  # Q-таблица

# Дискретизация расстояния
DISTANCE_BINS = np.linspace(0, WIDTH, 10)

# Данные для графика
scores = []
episodes = []
episode_count = 0
EPISODE_LENGTH = 1200  # 10 секунд
frame_count = 0
MOVING_AVERAGE_WINDOW = 10  # Окно для скользящего среднего

# Функция: Преобразует расстояние в состояние
def get_state(distance):
    idx = np.digitize(distance, DISTANCE_BINS) - 1
    idx = max(0, min(len(DISTANCE_BINS)-2, idx))
    return idx

# Функция: Выбирает действие (0: не прыгать, 1: прыгать)
def choose_action(state):
    if random.random() < EPSILON:
        return random.randint(0, 1)
    return np.argmax(q_table[state])

# Функция: Обновляет Q-таблицу
def update_q_table(state, action, reward, next_state):
    best_next = np.argmax(q_table[next_state])
    q_table[state][action] += ALPHA * (
        reward + GAMMA * q_table[next_state][best_next] - q_table[state][action]
    )

# Функция: Сбрасывает игру
def reset_game():
    global ball_y, ball_vy, obstacles, frame_count
    ball_y = HEIGHT - BALL_RADIUS
    ball_vy = 0
    obstacles = []
    frame_count = 0
    return 0

# Основной игровой цикл
pygame.display.set_caption("Прыгающий Мяч")
plt.ion()
score = 0
state = None
action = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    frame_count += 1

    # Обновление мяча
    ball_vy += GRAVITY
    ball_y += ball_vy
    if ball_y > HEIGHT - BALL_RADIUS:
        ball_y = HEIGHT - BALL_RADIUS
        ball_vy = 0

    # Обновление препятствий
    if frame_count % SPAWN_INTERVAL == 0 or frame_count == 60:
        obstacles.append([WIDTH, HEIGHT - OBSTACLE_HEIGHT])
    for obs in obstacles[:]:
        obs[0] += obstacle_speed
        if obs[0] < -OBSTACLE_WIDTH:
            obstacles.remove(obs)
            score += 10

    # Находим расстояние до препятствия
    nearest_obs = WIDTH
    for obs in obstacles:
        if obs[0] > ball_x:
            nearest_obs = min(nearest_obs, obs[0] - ball_x)
    nearest_obs = max(0, min(WIDTH, nearest_obs))

    # Q-learning
    new_state = get_state(nearest_obs)
    reward = 1
    end_episode = False

    # Проверка столкновения
    for obs in obstacles:
        if (ball_x + BALL_RADIUS > obs[0] and
            ball_x - BALL_RADIUS < obs[0] + OBSTACLE_WIDTH and
            ball_y + BALL_RADIUS > HEIGHT - OBSTACLE_HEIGHT):
            reward = -100
            update_q_table(state, action, reward, new_state)
            end_episode = True
            break
    if frame_count >= EPISODE_LENGTH:
        end_episode = True

    # Завершение эпизода
    if end_episode:
        episode_count += 1
        scores.append(score)
        episodes.append(episode_count)
        # Уменьшаем EPSILON для меньшей случайности
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        # Показываем прогресс каждые 100 эпизодов
        if episode_count % 100 == 0:
            print(f"Эпизод: {episode_count}, Счет: {score}, EPSILON: {EPSILON:.3f}")
        plt.clf()
        # Рисуем график: сырой счёт и скользящее среднее
        plt.plot(episodes, scores, 'b-', alpha=0.3, label="Сырой счёт")
        if len(scores) >= MOVING_AVERAGE_WINDOW:
            moving_avg = [
                np.mean(scores[max(0, i-MOVING_AVERAGE_WINDOW+1):i+1])
                for i in range(len(scores))
            ]
            plt.plot(episodes, moving_avg, 'r-', label=f"Среднее ({MOVING_AVERAGE_WINDOW} эпизодов)")
        plt.xlabel("Эпизод")
        plt.ylabel("Счет")
        plt.title("Прогресс обучения")
        plt.grid(True)
        plt.legend()
        plt.pause(0.01)
        score = reset_game()
        new_state = get_state(WIDTH)

    # Обновление Q-таблицы и выбор действия
    if state is not None:
        update_q_table(state, action, reward, new_state)
    action = choose_action(new_state)
    state = new_state

    # Прыжок
    if action == 1 and ball_y >= HEIGHT - BALL_RADIUS:
        ball_vy = JUMP_SPEED

    # Отрисовка
    screen.fill(BLACK)
    pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), BALL_RADIUS)
    for obs in obstacles:
        pygame.draw.rect(screen, WHITE, (obs[0], obs[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
    font = pygame.font.SysFont(None, 36)
    # Показываем счет и номер эпизода на экране
    screen.blit(font.render(f"Счет: {score} | Эпизод: {episode_count}", True, WHITE), (10, 10))
    pygame.display.flip()
    clock.tick(60)

plt.close()
pygame.quit()