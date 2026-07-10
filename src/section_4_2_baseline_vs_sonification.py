"""Section 4.2: comparison of the vector baseline and the sonification-based 2D-CNN model."""

# ============================================================
# РОЗДІЛ 4.2
# Порівняння векторної моделі та соніфікованої моделі
# NSL-KDD: binary classification (normal / attack)
# ============================================================

# ============================================================
# 1. ІМПОРТИ
# ============================================================
import os
import time
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.signal import stft

# ------------------------------------------------------------
# Шляхи репозиторію та директорія для збереження результатів
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(BASE_DIR, 'results', 'section_4_2')
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ============================================================
column_names_43 = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty_level"
]

train_path = os.path.join(BASE_DIR, 'data', 'nsl-kdd', 'KDDTrain+.txt')
test_path  = os.path.join(BASE_DIR, 'data', 'nsl-kdd', 'KDDTest+.txt')

missing_dataset_files = [path for path in (train_path, test_path) if not os.path.isfile(path)]
if missing_dataset_files:
    raise FileNotFoundError(
        "NSL-KDD files are missing. Place KDDTrain+.txt and KDDTest+.txt in "
        f"{os.path.join(BASE_DIR, 'data', 'nsl-kdd')}. Missing: {missing_dataset_files}"
    )

train_data = pd.read_csv(train_path, header=None, names=column_names_43)
test_data  = pd.read_csv(test_path,  header=None, names=column_names_43)

print("===== Перевірка розмірів датасету =====")
print("Train Data Shape:", train_data.shape)
print("Test Data Shape :", test_data.shape)

X_train_df = train_data.drop(['label','difficulty_level'], axis=1)
y_train_df = train_data['label']

X_test_df  = test_data.drop(['label','difficulty_level'], axis=1)
y_test_df  = test_data['label']

def label_to_binary(lbl):
    return 0 if lbl == 'normal' else 1

y_train_bin = np.array(y_train_df.apply(label_to_binary))
y_test_bin  = np.array(y_test_df.apply(label_to_binary))

print("\n===== Унікальні мітки у train =====")
print(np.unique(y_train_bin))

# ============================================================
# 3. ONE-HOT ENCODING + NORMALIZATION
# ============================================================
cat_cols = ['protocol_type','service','flag']

X_train_enc = pd.get_dummies(X_train_df, columns=cat_cols)
X_test_enc  = pd.get_dummies(X_test_df, columns=cat_cols)

X_train_enc, X_test_enc = X_train_enc.align(
    X_test_enc, join='left', axis=1, fill_value=0
)

print("\nПісля One-Hot: X_train_enc.shape =", X_train_enc.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled  = scaler.transform(X_test_enc)

print("X_train_scaled.shape =", X_train_scaled.shape)
print("X_test_scaled.shape  =", X_test_scaled.shape)

# ============================================================
# 4. СОНІФІКАЦІЯ: ПАРАМЕТРИ (НЕ ЗМІНЮЄМО)
# ============================================================
sample_rate = 2000
record_dur  = 0.4
samples_per_record = int(sample_rate * record_dur)

f_min, f_max = 200, 2000

def features_to_wave(features):
    """
    features: масив (F,) у [-1..1].
    Повертає wave з samples_per_record семплів.
    """
    F = len(features)
    wave = np.zeros(samples_per_record, dtype=float)

    samples_per_feat = samples_per_record // F
    remainder = samples_per_record - samples_per_feat * F

    idx = 0
    for i, x in enumerate(features):
        freq = f_min + x * (f_max - f_min)
        amp  = x
        for k in range(samples_per_feat):
            t_local = (k / sample_rate)
            wave[idx] = amp * np.sin(2 * np.pi * freq * t_local)
            idx += 1

    while idx < samples_per_record:
        wave[idx] = 0.0
        idx += 1

    return wave

def wave_to_stft_image(wave, sr=8000, nperseg=256, noverlap=128):
    """
    Обчислює STFT і повертає 2D-матрицю log(1+|Zxx|).
    """
    f, t, Zxx = stft(wave, fs=sr, nperseg=nperseg, noverlap=noverlap)
    spec = np.log1p(np.abs(Zxx))
    return spec

def stack_spectrograms(spec_list):
    N = len(spec_list)
    freq_bins = spec_list[0].shape[0]
    time_bins = spec_list[0].shape[1]
    arr = np.zeros((N, freq_bins, time_bins, 1), dtype=np.float32)
    for i in range(N):
        arr[i, :, :, 0] = spec_list[i]
    return arr

# ============================================================
# 5. ГЕНЕРАЦІЯ PCM + STFT
# ============================================================
print("\n=== Генерація PCM + STFT на TRAIN та TEST ===")

t0_sonif_prep = time.time()

train_stft = []
for i in range(X_train_scaled.shape[0]):
    features = X_train_scaled[i]
    wave = features_to_wave(features)
    spec = wave_to_stft_image(wave, sr=sample_rate, nperseg=256, noverlap=128)
    train_stft.append(spec)

test_stft = []
for i in range(X_test_scaled.shape[0]):
    features = X_test_scaled[i]
    wave = features_to_wave(features)
    spec = wave_to_stft_image(wave, sr=sample_rate, nperseg=256, noverlap=128)
    test_stft.append(spec)

train_stft = np.array(train_stft, dtype=object)
test_stft  = np.array(test_stft, dtype=object)

train_stft_4d = stack_spectrograms(train_stft)
test_stft_4d  = stack_spectrograms(test_stft)

t1_sonif_prep = time.time()
sonif_prep_time_sec = t1_sonif_prep - t0_sonif_prep

print("train_stft_4d.shape =", train_stft_4d.shape)
print("test_stft_4d.shape  =", test_stft_4d.shape)

y_train_final = y_train_bin
y_test_final  = y_test_bin

# ============================================================
# 6. ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================
def save_table(df, filename):
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Saved table: {path}")

def save_plot(filename):
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {path}")
    plt.show()

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }
    return metrics, cm

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['normal', 'attack'])
    plt.yticks(tick_marks, ['normal', 'attack'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

def build_confidence_distribution_df(y_true, y_pred, y_prob, bins):
    rows = []
    correct_mask = (y_true == y_pred)

    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        total = int(mask.sum())
        correct = int((mask & correct_mask).sum())
        incorrect = int((mask & (~correct_mask)).sum())

        rows.append({
            'Інтервал ймовірності': f'[{left:.1f}; {right:.1f}' + (']' if right == 1.0 else ')'),
            'Кількість прогнозів': total,
            'Правильні': correct,
            'Помилкові': incorrect
        })

    return pd.DataFrame(rows)

# ============================================================
# 7. ВЕКТОРНА МОДЕЛЬ (BASELINE)
# ============================================================
print("\n=== Побудова векторної baseline-моделі ===")

model_vector = tf.keras.Sequential([
    layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_vector.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_vector.summary()

t0_vector_train = time.time()

history_vector = model_vector.fit(
    X_train_scaled, y_train_final,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

t1_vector_train = time.time()
vector_train_time_sec = t1_vector_train - t0_vector_train

# Оцінка vector model
t0_vector_pred = time.time()
vector_test_loss, vector_test_acc = model_vector.evaluate(X_test_scaled, y_test_final, verbose=0)
vector_probs = model_vector.predict(X_test_scaled, verbose=0).reshape(-1)
vector_pred  = (vector_probs > 0.5).astype(int)
t1_vector_pred = time.time()
vector_infer_time_sec = t1_vector_pred - t0_vector_pred

vector_metrics, vector_cm = compute_metrics(y_test_final, vector_pred, vector_probs)

print("\n=== VECTOR MODEL RESULTS ===")
print(f"Test Loss: {vector_test_loss:.4f}")
print(f"Test Accuracy: {vector_test_acc:.4f}")
print(confusion_matrix(y_test_final, vector_pred))
print(classification_report(y_test_final, vector_pred, digits=4))

# ============================================================
# 8. СОНІФІКОВАНА 2D-CNN МОДЕЛЬ (ЯК У ТВОЄМУ СКРИПТІ)
# ============================================================
print("\n=== Побудова 2D-CNN ===")

model_2d = tf.keras.Sequential([
    layers.InputLayer(input_shape=(train_stft_4d.shape[1], train_stft_4d.shape[2], 1)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_2d.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_2d.summary()

t0_sonif_train = time.time()

history_2d = model_2d.fit(
    train_stft_4d, y_train_final,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

t1_sonif_train = time.time()
sonif_train_time_sec = t1_sonif_train - t0_sonif_train

# Оцінка sonified model
t0_sonif_pred = time.time()
test_loss, test_acc = model_2d.evaluate(test_stft_4d, y_test_final, verbose=0)
sonif_probs = model_2d.predict(test_stft_4d, verbose=0).reshape(-1)
sonif_pred  = (sonif_probs > 0.5).astype(int)
t1_sonif_pred = time.time()
sonif_infer_time_sec = t1_sonif_pred - t0_sonif_pred

sonif_metrics, sonif_cm = compute_metrics(y_test_final, sonif_pred, sonif_probs)

print("\n=== SONIFIED 2D-CNN RESULTS ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(confusion_matrix(y_test_final, sonif_pred))
print(classification_report(y_test_final, sonif_pred, digits=4))

# ============================================================
# 9. ТАБЛИЦЯ 4.6
# Основні параметри моделей
# ============================================================
table_4_6 = pd.DataFrame([
    ['Векторна модель', 'One-Hot + MinMaxScaler(-1,1)', X_train_scaled.shape[1], 'Dense(128)-Dense(64)-Dense(1)', 5, 128, 0.0005, 'Adam', 'binary_crossentropy'],
    ['Соніфікована модель', 'One-Hot + MinMaxScaler(-1,1) + PCM + STFT', f'{train_stft_4d.shape[1]}x{train_stft_4d.shape[2]}x1', 'Conv2D(16)-MaxPool-Conv2D(32)-GlobalMaxPool-Dense(32)-Dense(1)', 5, 128, 0.0005, 'Adam', 'binary_crossentropy']
], columns=[
    'Модель', 'Подання вхідних даних', 'Розмір входу', 'Архітектура',
    'Epochs', 'Batch size', 'Learning rate', 'Optimizer', 'Loss'
])

print("\nТаблиця 4.6")
print(table_4_6)
save_table(table_4_6, 'table_4_6_model_parameters.csv')

# ============================================================
# 10. ТАБЛИЦЯ 4.7
# Результати соніфікованої моделі
# ============================================================
table_4_7 = pd.DataFrame([{
    'Модель': 'Соніфікована 2D-CNN',
    'Accuracy': round(sonif_metrics['Accuracy'], 4),
    'Precision': round(sonif_metrics['Precision'], 4),
    'Recall': round(sonif_metrics['Recall'], 4),
    'F1-score': round(sonif_metrics['F1-score'], 4),
    'FPR': round(sonif_metrics['FPR'], 4),
    'FNR': round(sonif_metrics['FNR'], 4)
}])

print("\nТаблиця 4.7")
print(table_4_7)
save_table(table_4_7, 'table_4_7_sonified_results.csv')

# ============================================================
# 11. ТАБЛИЦЯ 4.8
# Порівняння векторної та соніфікованої моделей
# ============================================================
table_4_8 = pd.DataFrame([
    {
        'Модель': 'Векторна',
        'Accuracy': round(vector_metrics['Accuracy'], 4),
        'Precision': round(vector_metrics['Precision'], 4),
        'Recall': round(vector_metrics['Recall'], 4),
        'F1-score': round(vector_metrics['F1-score'], 4),
        'FPR': round(vector_metrics['FPR'], 4),
        'FNR': round(vector_metrics['FNR'], 4)
    },
    {
        'Модель': 'Соніфікована',
        'Accuracy': round(sonif_metrics['Accuracy'], 4),
        'Precision': round(sonif_metrics['Precision'], 4),
        'Recall': round(sonif_metrics['Recall'], 4),
        'F1-score': round(sonif_metrics['F1-score'], 4),
        'FPR': round(sonif_metrics['FPR'], 4),
        'FNR': round(sonif_metrics['FNR'], 4)
    }
])

print("\nТаблиця 4.8")
print(table_4_8)
save_table(table_4_8, 'table_4_8_vector_vs_sonified_metrics.csv')

# ============================================================
# 12. ТАБЛИЦЯ 4.9
# Порівняння часу навчання
# ============================================================
table_4_9 = pd.DataFrame([
    {
        'Модель': 'Векторна',
        'Час підготовки даних, с': round(0.0, 2),
        'Час навчання, с': round(vector_train_time_sec, 2),
        'Час інференсу на test, с': round(vector_infer_time_sec, 2),
        'Сумарний час, с': round(vector_train_time_sec + vector_infer_time_sec, 2)
    },
    {
        'Модель': 'Соніфікована',
        'Час підготовки даних, с': round(sonif_prep_time_sec, 2),
        'Час навчання, с': round(sonif_train_time_sec, 2),
        'Час інференсу на test, с': round(sonif_infer_time_sec, 2),
        'Сумарний час, с': round(sonif_prep_time_sec + sonif_train_time_sec + sonif_infer_time_sec, 2)
    }
])

print("\nТаблиця 4.9")
print(table_4_9)
save_table(table_4_9, 'table_4_9_training_time_comparison.csv')

# ============================================================
# 13. ТАБЛИЦЯ 4.10
# Розподіл помилок за рівнем впевненості
# ============================================================
bins = np.arange(0.0, 1.01, 0.1)

table_4_10_vector = build_confidence_distribution_df(
    y_test_final, vector_pred, vector_probs, bins
)
table_4_10_vector.insert(0, 'Модель', 'Векторна')

table_4_10_sonif = build_confidence_distribution_df(
    y_test_final, sonif_pred, sonif_probs, bins
)
table_4_10_sonif.insert(0, 'Модель', 'Соніфікована')

table_4_10 = pd.concat([table_4_10_vector, table_4_10_sonif], axis=0, ignore_index=True)

print("\nТаблиця 4.10")
print(table_4_10.head(20))
save_table(table_4_10, 'table_4_10_confidence_distribution.csv')

# ============================================================
# 14. РИСУНОК 4.6
# Приклад normal запису після соніфікації та спектрограма
# ============================================================
normal_idx = np.where(y_train_final == 0)[0][0]
normal_features = X_train_scaled[normal_idx]
normal_wave = features_to_wave(normal_features)
normal_spec = wave_to_stft_image(normal_wave, sr=sample_rate, nperseg=256, noverlap=128)
fN, tN, ZN = stft(normal_wave, fs=sample_rate, nperseg=256, noverlap=128)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(normal_wave)
plt.title('Рис. 4.6. Нормальний запис після соніфікації: хвильове представлення')
plt.xlabel('Номер семплу')
plt.ylabel('Амплітуда')

plt.subplot(2, 1, 2)
plt.pcolormesh(tN, fN, normal_spec, shading='auto')
plt.title('Рис. 4.6. Нормальний запис після соніфікації: спектрограма')
plt.xlabel('Час, с')
plt.ylabel('Частота, Гц')
plt.colorbar(label='log(amp)')

save_plot('fig_4_6_normal_wave_and_spectrogram.png')

# ============================================================
# 15. РИСУНОК 4.7
# Приклад attack запису після соніфікації та спектрограма
# ============================================================
attack_idx = np.where(y_train_final == 1)[0][0]
attack_features = X_train_scaled[attack_idx]
attack_wave = features_to_wave(attack_features)
attack_spec = wave_to_stft_image(attack_wave, sr=sample_rate, nperseg=256, noverlap=128)
fA, tA, ZA = stft(attack_wave, fs=sample_rate, nperseg=256, noverlap=128)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(attack_wave)
plt.title('Рис. 4.7. Атакувальний запис після соніфікації: хвильове представлення')
plt.xlabel('Номер семплу')
plt.ylabel('Амплітуда')

plt.subplot(2, 1, 2)
plt.pcolormesh(tA, fA, attack_spec, shading='auto')
plt.title('Рис. 4.7. Атакувальний запис після соніфікації: спектрограма')
plt.xlabel('Час, с')
plt.ylabel('Частота, Гц')
plt.colorbar(label='log(amp)')

save_plot('fig_4_7_attack_wave_and_spectrogram.png')

# ============================================================
# 16. РИСУНОК 4.8
# Подібність спектрограм для 5 записів одного типу атаки
# ============================================================
attack_label_counts = train_data[train_data['label'] != 'normal']['label'].value_counts()
chosen_attack_label = attack_label_counts.index[0]

same_attack_indices = train_data.index[train_data['label'] == chosen_attack_label].tolist()[:5]
same_attack_specs = []

for idx in same_attack_indices:
    feats = X_train_scaled[idx]
    wave = features_to_wave(feats)
    spec = wave_to_stft_image(wave, sr=sample_rate, nperseg=256, noverlap=128)
    same_attack_specs.append(spec)

plt.figure(figsize=(14, 10))
for i, spec in enumerate(same_attack_specs, start=1):
    plt.subplot(3, 2, i)
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(f'Запис {i}: {chosen_attack_label}')
    plt.xlabel('Часові фрейми')
    plt.ylabel('Частотні бін-и')

plt.suptitle('Рис. 4.8. Подібність спектрограм для 5 записів одного типу атаки', y=1.02)
save_plot('fig_4_8_similarity_of_5_same_attack_spectrograms.png')

# ============================================================
# 17. РИСУНОК 4.9
# Матриця невідповідностей соніфікованої моделі
# ============================================================
plot_confusion_matrix(sonif_cm, 'Рис. 4.9. Матриця невідповідностей соніфікованої моделі')
save_plot('fig_4_9_confusion_matrix_sonified.png')

# ============================================================
# 18. РИСУНОК 4.10
# Графіки навчання соніфікованої моделі
# ============================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_2d.history['loss'], label='Train Loss')
plt.plot(history_2d.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Рис. 4.10. Динаміка функції втрат соніфікованої моделі')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_2d.history['accuracy'], label='Train Acc')
plt.plot(history_2d.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Рис. 4.10. Динаміка точності соніфікованої моделі')
plt.legend()

save_plot('fig_4_10_sonified_training_curves.png')

# ============================================================
# 19. РИСУНОК 4.11
# Порівняння часу навчання
# ============================================================
plt.figure(figsize=(8, 5))
labels = ['Векторна', 'Соніфікована']
train_times = [vector_train_time_sec, sonif_train_time_sec]

plt.bar(labels, train_times)
plt.title('Рис. 4.11. Порівняння часу навчання моделей')
plt.ylabel('Час, с')
save_plot('fig_4_11_training_time_comparison.png')

# ============================================================
# 20. РИСУНОК 4.12
# Розподіл правильних і помилкових рішень за рівнем впевненості
# ============================================================
def plot_confidence_distribution(y_true, y_pred, y_prob, title, filename):
    bins_local = np.arange(0.0, 1.01, 0.1)
    correct_mask = (y_true == y_pred)

    correct_counts = []
    incorrect_counts = []

    for left, right in zip(bins_local[:-1], bins_local[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        correct_counts.append(int((mask & correct_mask).sum()))
        incorrect_counts.append(int((mask & (~correct_mask)).sum()))

    x = np.arange(len(correct_counts))
    xlabels = [f'{bins_local[i]:.1f}-{bins_local[i+1]:.1f}' for i in range(len(bins_local)-1)]

    plt.figure(figsize=(12, 5))
    plt.bar(x, correct_counts, label='Правильні')
    plt.bar(x, incorrect_counts, bottom=correct_counts, label='Помилкові')
    plt.xticks(x, xlabels, rotation=45)
    plt.xlabel('Інтервал прогнозованої ймовірності')
    plt.ylabel('Кількість прогнозів')
    plt.title(title)
    plt.legend()
    save_plot(filename)

plot_confidence_distribution(
    y_test_final, sonif_pred, sonif_probs,
    'Рис. 4.12. Розподіл правильних і помилкових рішень соніфікованої моделі за рівнем впевненості',
    'fig_4_12_sonified_confidence_distribution.png'
)

# ============================================================
# 21. ДОДАТКОВІ РИСУНКИ ДЛЯ ПОРІВНЯННЯ МОДЕЛЕЙ
# ============================================================
# 21.1 Матриця невідповідностей векторної моделі
plot_confusion_matrix(vector_cm, 'Додатковий рисунок. Матриця невідповідностей векторної моделі')
save_plot('fig_extra_vector_confusion_matrix.png')

# 21.2 Графіки навчання векторної моделі
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_vector.history['loss'], label='Train Loss')
plt.plot(history_vector.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Додатковий рисунок. Динаміка функції втрат векторної моделі')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_vector.history['accuracy'], label='Train Acc')
plt.plot(history_vector.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Додатковий рисунок. Динаміка точності векторної моделі')
plt.legend()

save_plot('fig_extra_vector_training_curves.png')

# 21.3 Розподіл впевненості для векторної моделі
plot_confidence_distribution(
    y_test_final, vector_pred, vector_probs,
    'Додатковий рисунок. Розподіл правильних і помилкових рішень векторної моделі за рівнем впевненості',
    'fig_extra_vector_confidence_distribution.png'
)

# 21.4 Порівняння метрик vector vs sonified
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
vector_vals = [vector_metrics[m] for m in metrics_names]
sonif_vals  = [sonif_metrics[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, vector_vals, width, label='Векторна')
plt.bar(x + width/2, sonif_vals, width, label='Соніфікована')
plt.xticks(x, metrics_names)
plt.ylim(0, 1)
plt.ylabel('Значення метрики')
plt.title('Додатковий рисунок. Порівняння основних метрик моделей')
plt.legend()
save_plot('fig_extra_vector_vs_sonified_metrics.png')

# ============================================================
# 22. ЗБЕРЕЖЕННЯ КЛАСИФІКАЦІЙНИХ ЗВІТІВ
# ============================================================
vector_report_text = classification_report(y_test_final, vector_pred, digits=4)
sonif_report_text  = classification_report(y_test_final, sonif_pred, digits=4)

with open(os.path.join(output_dir, 'vector_classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(vector_report_text)

with open(os.path.join(output_dir, 'sonified_classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(sonif_report_text)

# ============================================================
# 23. КОРОТКЕ ЗВЕДЕННЯ ДЛЯ ТЕКСТУ ДИСЕРТАЦІЇ
# ============================================================
summary = {
    'vector_model': {
        'accuracy': round(vector_metrics['Accuracy'], 4),
        'precision': round(vector_metrics['Precision'], 4),
        'recall': round(vector_metrics['Recall'], 4),
        'f1': round(vector_metrics['F1-score'], 4),
        'fpr': round(vector_metrics['FPR'], 4),
        'fnr': round(vector_metrics['FNR'], 4),
        'train_time_sec': round(vector_train_time_sec, 2),
        'infer_time_sec': round(vector_infer_time_sec, 2)
    },
    'sonified_model': {
        'accuracy': round(sonif_metrics['Accuracy'], 4),
        'precision': round(sonif_metrics['Precision'], 4),
        'recall': round(sonif_metrics['Recall'], 4),
        'f1': round(sonif_metrics['F1-score'], 4),
        'fpr': round(sonif_metrics['FPR'], 4),
        'fnr': round(sonif_metrics['FNR'], 4),
        'prep_time_sec': round(sonif_prep_time_sec, 2),
        'train_time_sec': round(sonif_train_time_sec, 2),
        'infer_time_sec': round(sonif_infer_time_sec, 2)
    },
    'chosen_attack_for_similarity_figure': chosen_attack_label
}

with open(os.path.join(output_dir, 'summary_4_2.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n=== ГОТОВО ===")
print("Результати збережені у:", output_dir)
print("\nОсновні таблиці:")
print("- table_4_6_model_parameters.csv")
print("- table_4_7_sonified_results.csv")
print("- table_4_8_vector_vs_sonified_metrics.csv")
print("- table_4_9_training_time_comparison.csv")
print("- table_4_10_confidence_distribution.csv")

print("\nОсновні рисунки:")
print("- fig_4_6_normal_wave_and_spectrogram.png")
print("- fig_4_7_attack_wave_and_spectrogram.png")
print("- fig_4_8_similarity_of_5_same_attack_spectrograms.png")
print("- fig_4_9_confusion_matrix_sonified.png")
print("- fig_4_10_sonified_training_curves.png")
print("- fig_4_11_training_time_comparison.png")
print("- fig_4_12_sonified_confidence_distribution.png")

print("\nДодаткові рисунки:")
print("- fig_extra_vector_confusion_matrix.png")
print("- fig_extra_vector_training_curves.png")
print("- fig_extra_vector_confidence_distribution.png")
print("- fig_extra_vector_vs_sonified_metrics.png")
