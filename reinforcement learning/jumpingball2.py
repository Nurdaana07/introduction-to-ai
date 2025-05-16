import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import imageio.v3 as iio
import os

pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
WHITE, RED, BLACK = (255, 255, 255), (255, 0, 0), (0, 0, 0)

BALL_RADIUS = 20
ball_x, ball_y = 100, HEIGHT - BALL_RADIUS
ball_vy = 0
GRAVITY, JUMP_SPEED = 0.5, -12

OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 40
obstacle_speed = -3
obstacles = []
SPAWN_INTERVAL = 180

ALPHA, GAMMA = 0.1, 0.9
EPSILON_MAX, EPSILON = 0.1, 0.1
EPSILON_DECAY, MIN_EPSILON = 0.995, 0.01
q_table = defaultdict(lambda: np.zeros(2))
DISTANCE_BINS = np.linspace(0, WIDTH, 10)

Q_TABLE_FILE = "q_table.pkl"
if os.path.exists(Q_TABLE_FILE):
    with open(Q_TABLE_FILE, 'rb') as f:
        q_table.update(pickle.load(f))
        print(f"Загружена Q-таблица из {Q_TABLE_FILE}")

scores, episodes, epsilon_values = [], [], []
episode_count, frame_count, score = 0, 0, 0
frames = []  # Для GIF

def handle_q_learning(distance, state, action, reward):
    idx = np.digitize(distance, DISTANCE_BINS) - 1
    new_state = max(0, min(len(DISTANCE_BINS)-2, idx))
    if state is not None:
        best_next = np.argmax(q_table[new_state])
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[new_state][best_next] - q_table[state][action]
        )
    new_action = random.randint(0, 1) if random.random() < EPSILON else np.argmax(q_table[new_state])
    return new_state, new_action

def draw_game():
    screen.fill(BLACK)
    pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), BALL_RADIUS)
    for obs in obstacles:
        pygame.draw.rect(screen, WHITE, (obs[0], obs[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
    screen.blit(pygame.font.SysFont(None, 36).render(
        f"Счет: {score} | Эпизод: {episode_count}", True, WHITE), (10, 10))
    pygame.display.flip()
    if frame_count % 5 == 0:  # Сохраняем каждый 5-й кадр
        frame = pygame.surfarray.array3d(screen).swapaxes(0, 1)
        frames.append(frame)

def update_plot():
    plt.clf()
    plt.plot(episodes, scores, 'b-', alpha=0.3, label="Сырой счёт")
    if len(scores) >= 10:
        moving_avg = [np.mean(scores[max(0, i-9):i+1]) for i in range(len(scores))]
        plt.plot(episodes, moving_avg, 'r-', label="Среднее (10 эпизодов)")
    plt.xlabel("Эпизод")
    plt.ylabel("Счет", color='b')
    ax2 = plt.gca().twinx()
    ax2.plot(episodes, epsilon_values, 'g--', label="EPSILON")
    ax2.set_ylabel("EPSILON", color='g')
    plt.title("Прогресс")
    plt.grid(True)
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.pause(0.01)

def reset_game():
    global ball_y, ball_vy, obstacles, frame_count, score
    ball_y, ball_vy, obstacles, frame_count, score = HEIGHT - BALL_RADIUS, 0, [], 0, 0

pygame.display.set_caption("Прыгающий Мяч")
plt.ion()
state, action = None, 0
running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame_count += 1
        ball_vy += GRAVITY
        ball_y += ball_vy
        if ball_y > HEIGHT - BALL_RADIUS:
            ball_y, ball_vy = HEIGHT - BALL_RADIUS, 0

        if frame_count % SPAWN_INTERVAL == 0 or frame_count == 60:
            obstacles.append([WIDTH, HEIGHT - OBSTACLE_HEIGHT])
        for obs in obstacles[:]:
            obs[0] += obstacle_speed
            if obs[0] < -OBSTACLE_WIDTH:
                obstacles.remove(obs)
                score += 10

        nearest_obs = WIDTH
        for obs in obstacles:
            if obs[0] > ball_x:
                nearest_obs = min(nearest_obs, obs[0] - ball_x)
        nearest_obs = max(0, min(WIDTH, nearest_obs))

        reward = 1
        state, action = handle_q_learning(nearest_obs, state, action, reward)
        end_episode = False

        for obs in obstacles:
            if (ball_x + BALL_RADIUS > obs[0] and
                ball_x - BALL_RADIUS < obs[0] + OBSTACLE_WIDTH and
                ball_y + BALL_RADIUS > HEIGHT - OBSTACLE_HEIGHT):
                reward = -100
                state, action = handle_q_learning(nearest_obs, state, action, reward)
                end_episode = True
                break

        if end_episode:
            episode_count += 1
            scores.append(score)
            episodes.append(episode_count)
            epsilon_values.append(EPSILON)
            EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
            if episode_count % 100 == 0:
                print(f"Эпизод: {episode_count}, Счет: {score}, EPSILON: {EPSILON:.3f}")
                with open(Q_TABLE_FILE, 'wb') as f:
                    pickle.dump(dict(q_table), f)
            update_plot()
            reset_game()
            state = None

        if action == 1 and ball_y >= HEIGHT - BALL_RADIUS:
            ball_vy = JUMP_SPEED

        draw_game()
        clock.tick(60)

finally:
    with open(Q_TABLE_FILE, 'wb') as f:
        pickle.dump(dict(q_table), f)
    if frames:
        iio.imwrite("learning_process.gif", frames, duration=1000/60, loop=0)
        print("GIF сохранён как learning_process.gif")
    print(f"Q-таблица сохранена")
    plt.close()
    pygame.quit()