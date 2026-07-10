# @title
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
# Налаштування директорії для збереження
# ------------------------------------------------------------
output_dir = '/content/nsl_kdd_section_4_2'
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

train_path = '/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt'
test_path  = '/content/drive/MyDrive/nsl-kdd/KDDTest+.txt'

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










































# @title
# ============================================================
# РОЗДІЛ 4.3
# Дослідження методу балансування: Baseline vs SMOTENC vs SPAS
# (Binary IDS: normal vs attack, + per-attack recall на test)
# Соніфікація (PCM->STFT) + 2D-CNN як у "стабільному" коді
# ============================================================

import os, time, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, optimizers

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

from scipy.signal import stft

# SMOTENC (репер)
from imblearn.over_sampling import SMOTENC, RandomOverSampler

# ------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------
train_path = '/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt'
test_path  = '/content/drive/MyDrive/nsl-kdd/KDDTest+.txt'

OUT_DIR = '/content/nsl_kdd_section_4_3'
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE_VAL = 0.1

# Для швидких тестів (None = full train)
MAX_TRAIN_SAMPLES_FOR_DEBUG = None   # напр 30000 або None

# Які режими запускати:
RUN_MODES = ["baseline", "smotenc", "spas"]  # можна лишити один/два

# Цільові атаки для oversampling (як у тебе)
TARGET_ATTACKS = [
 'snmpgetattack','snmpguess','phf','xsnoop','ps','sendmail','xterm',
 'buffer_overflow','xlock','loadmodule','udpstorm','imap','worm',
 'sqlattack','perl','mailbomb','processtable','rootkit','guess_passwd',
 'multihop','warezmaster','back','named'
]

# Обмеження/політика oversampling
MAX_MULTIPLIER = 5
MIN_TARGET_COUNT = 1000   # мінімальний target після oversampling для класу (cap by max_attack_count)

# Параметри соніфікації (стабільні як у твоєму коді)
SON_CFG = {
    "sample_rate": 2000,
    "record_dur": 0.4,
    "f_min": 200,
    "f_max": 2000,
    "nperseg": 256,
    "noverlap": 128,
    "epochs": 5,
    "batch_size": 128,
    "lr": 0.0005,
    "threshold": 0.5
}

# Параметри SPAS
SPAS_CFG = {
    "q_low": 0.05,          # нижній процентиль
    "q_high": 0.95,         # верхній процентиль
    "alpha_min": 0.2,       # інтерполяція між 2 зразками
    "alpha_max": 0.8,
    "noise_sigma": 0.03,    # гаус шум по числових фічах
    "max_tries": 50,        # спроби згенерувати валідний зразок (після clamp)
    "random_state": RANDOM_STATE
}

# Для ілюстрацій bounds: який тип атаки і які 8 ознак показувати
DEMO_ATTACK_FOR_BOUNDS = "guess_passwd"
DEMO_FEATURES_8 = ["src_bytes","dst_bytes","count","srv_count","serror_rate","srv_serror_rate","same_srv_rate","dst_host_count"]

# Для ілюстрацій "реальний vs синтетичний"
DEMO_ATTACK_FOR_SYNTH = "guess_passwd"
N_SYNTH_EXAMPLES = 3

# Збережемо конфіги
with open(os.path.join(OUT_DIR, "son_cfg.json"), "w", encoding="utf-8") as f:
    json.dump(SON_CFG, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, "spas_cfg.json"), "w", encoding="utf-8") as f:
    json.dump(SPAS_CFG, f, ensure_ascii=False, indent=2)

print("Saved configs:", OUT_DIR)

# ------------------------------------------------------------
# 2) COLUMNS
# ------------------------------------------------------------
cols = [
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

CAT_COLS = ['protocol_type','service','flag']

# ------------------------------------------------------------
# 3) HELPERS
# ------------------------------------------------------------
def label_to_binary(lbl):
    return 0 if lbl == 'normal' else 1

def save_table(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved table:", path)

def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved fig:", path)
    plt.show()

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred)
    return float(acc), float(p), float(r), float(f1), y_pred, cm

def plot_confidence_bins(y_true, y_prob, title, fig_name):
    y_pred = (y_prob >= 0.5).astype(int)
    correct = (y_pred == y_true)

    bins = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, len(bins)-2)

    good = np.zeros(len(bins)-1, dtype=int)
    bad  = np.zeros(len(bins)-1, dtype=int)
    for i in range(len(y_prob)):
        (good if correct[i] else bad)[bin_ids[i]] += 1

    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    x = np.arange(len(labels))

    plt.figure(figsize=(11,4))
    plt.bar(x, good, label="correct")
    plt.bar(x, bad, bottom=good, label="incorrect")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.xlabel("Predicted probability bin")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    save_fig(fig_name)

def per_attack_recall_table(y_test_text, y_true_bin, y_pred_bin):
    """
    Пер-атака recall на test: TP/(TP+FN) серед зразків даного true attack label.
    Для normal не рахуємо.
    """
    df = pd.DataFrame({
        "true_text": pd.Series(y_test_text).reset_index(drop=True),
        "y_true": np.array(y_true_bin).flatten(),
        "y_pred": np.array(y_pred_bin).flatten()
    })
    attacks = df[df["y_true"] == 1].copy()
    if len(attacks) == 0:
        return pd.DataFrame(columns=["attack_label","support","TP","FN","recall"])

    g = attacks.groupby("true_text")
    out = g.apply(lambda x: pd.Series({
        "support": int(len(x)),
        "TP": int((x["y_pred"] == 1).sum()),
        "FN": int((x["y_pred"] == 0).sum()),
        "recall": float((x["y_pred"] == 1).sum() / max(1, len(x)))
    })).reset_index().rename(columns={"true_text":"attack_label"}).sort_values("support", ascending=False)

    return out

# ------------------------------------------------------------
# 4) SONIFICATION PIPELINE (as in your stable code)
# ------------------------------------------------------------
sample_rate = SON_CFG["sample_rate"]
record_dur = SON_CFG["record_dur"]
samples_per_record = int(sample_rate * record_dur)
f_min, f_max = SON_CFG["f_min"], SON_CFG["f_max"]

def features_to_wave(features_scaled):
    # EXACT style: freq = f_min + x*(f_max-f_min), amp=x  (x in [-1,1])
    F = len(features_scaled)
    wave = np.zeros(samples_per_record, dtype=np.float32)
    spp = max(1, samples_per_record // F)
    idx = 0
    for x in features_scaled:
        freq = f_min + x * (f_max - f_min)
        amp = x
        for s in range(spp):
            if idx >= samples_per_record:
                break
            t_local = s / sample_rate
            wave[idx] = amp * np.sin(2*np.pi*freq*t_local)
            idx += 1
    return wave

def wave_to_stft_image(wave):
    f, t, Zxx = stft(wave, fs=sample_rate, nperseg=SON_CFG["nperseg"], noverlap=SON_CFG["noverlap"])
    spec = np.log1p(np.abs(Zxx)).astype(np.float32)
    return spec

def gen_specs_from_X(X_scaled):
    specs = []
    for i in range(X_scaled.shape[0]):
        specs.append(wave_to_stft_image(features_to_wave(X_scaled[i])))
    return np.stack(specs, axis=0)

def build_2dcnn(input_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(16, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.GlobalMaxPooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=SON_CFG["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ------------------------------------------------------------
# 5) LOAD + SPLIT
# ------------------------------------------------------------
train_df = pd.read_csv(train_path, header=None, names=cols)
test_df  = pd.read_csv(test_path,  header=None, names=cols)

if MAX_TRAIN_SAMPLES_FOR_DEBUG is not None:
    train_df = train_df.sample(n=MAX_TRAIN_SAMPLES_FOR_DEBUG, random_state=RANDOM_STATE).reset_index(drop=True)

print("Train:", train_df.shape, "Test:", test_df.shape)

X_all = train_df.drop(["label","difficulty_level"], axis=1).copy()
y_all_text = train_df["label"].copy()

X_test = test_df.drop(["label","difficulty_level"], axis=1).copy()
y_test_text = test_df["label"].copy()
y_test_bin  = y_test_text.apply(label_to_binary).values

y_all_bin = y_all_text.apply(label_to_binary).values

X_train_orig, X_val_orig, y_train_text_orig, y_val_text = train_test_split(
    X_all, y_all_text, test_size=TEST_SIZE_VAL, stratify=y_all_bin, random_state=RANDOM_STATE
)
y_val_bin = y_val_text.apply(label_to_binary).values

print("Train split:", X_train_orig.shape, "Val split:", X_val_orig.shape)

# ------------------------------------------------------------
# 6) TABLE/FIG: класовий розподіл до балансування
# ------------------------------------------------------------
def count_attacks(series):
    s = pd.Series(series)
    attacks = s[s != "normal"].value_counts().reset_index()
    attacks.columns = ["attack_label","count"]
    return attacks

before_counts = y_train_text_orig.value_counts().reset_index()
before_counts.columns = ["label","count"]
save_table(before_counts, "table_4_3_counts_before.csv")

top_before = count_attacks(y_train_text_orig).head(20)
plt.figure(figsize=(10,5))
plt.barh(top_before["attack_label"][::-1], top_before["count"][::-1])
plt.title("Top attack types in train (before balancing)")
plt.xlabel("Count")
save_fig("fig_4_3_top_attacks_before.png")

# ------------------------------------------------------------
# 7) OVERSAMPLING HELPERS (SMOTENC + SPAS)
# ------------------------------------------------------------
def make_sampling_strategy(y_train_text, targets, max_multiplier, min_target_count):
    counts = y_train_text.value_counts()
    attack_counts = counts[counts.index != "normal"]

    if len(attack_counts) == 0:
        return {}, attack_counts

    available = set(attack_counts.index.tolist())
    targets_set = set(targets) & available if targets is not None else available

    max_attack_count = int(attack_counts.max())
    strategy = {}

    for lbl, c in attack_counts.items():
        if lbl not in targets_set:
            continue
        base_target = min(int(c * max_multiplier), max_attack_count)
        if base_target < min_target_count:
            base_target = min(max_attack_count, int(min_target_count))
        if base_target > c:
            strategy[lbl] = base_target

    return strategy, attack_counts

def apply_smotenc(X_train_df, y_train_text, strategy):
    if len(strategy) == 0:
        return X_train_df.copy().reset_index(drop=True), y_train_text.reset_index(drop=True), "none"

    X_enc = X_train_df.copy().reset_index(drop=True)
    label_encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        label_encoders[col] = le

    X_sm = X_enc.values
    y_sm = y_train_text.values

    # safe k_neighbors
    counts = y_train_text.value_counts()
    min_c = min([counts[lbl] for lbl in strategy.keys()])
    k_neighbors = min(5, max(1, min_c - 1)) if min_c > 1 else 1

    try:
        cat_idx = [X_enc.columns.get_loc(c) for c in CAT_COLS]
        smote = SMOTENC(categorical_features=cat_idx, sampling_strategy=strategy,
                        random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_sm, y_sm)
        used = "SMOTENC"
    except Exception as e:
        print("SMOTENC failed:", e, "-> fallback ROS")
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=RANDOM_STATE)
        X_res, y_res = ros.fit_resample(X_sm, y_sm)
        used = "RandomOverSampler"

    X_res_df = pd.DataFrame(X_res, columns=X_enc.columns)
    for col in CAT_COLS:
        le = label_encoders[col]
        X_res_df[col] = X_res_df[col].round().astype(int).clip(0, len(le.classes_)-1)
        X_res_df[col] = le.inverse_transform(X_res_df[col].astype(int))

    return X_res_df.reset_index(drop=True), pd.Series(y_res).reset_index(drop=True), used

# ----- SPAS (Signature-Preserving Adaptive Sampling) -----
def spas_generate(X_train_df, y_train_text, strategy, q_low, q_high, alpha_min, alpha_max, noise_sigma, max_tries, seed):
    """
    SPAS ідея (практична реалізація для NSL-KDD):
    - працюємо у вихідному просторі до get_dummies (є категоріальні колонки)
    - для числових фіч: генеруємо через інтерполяцію між 2 реальними зразками + шум
    - після цього "clamp" у процентильні межі (signature-preserving bounds)
    - категоріальні поля (protocol/service/flag) копіюємо з одного з батьків (збереження сигнатури)
    """
    rng = np.random.default_rng(seed)

    if len(strategy) == 0:
        return X_train_df.copy().reset_index(drop=True), y_train_text.reset_index(drop=True)

    # розділяємо numeric/cat
    num_cols = [c for c in X_train_df.columns if c not in CAT_COLS]

    X_out = X_train_df.copy().reset_index(drop=True)
    y_out = y_train_text.copy().reset_index(drop=True)

    # precompute bounds per class (on numeric)
    bounds = {}
    for lbl, target_n in strategy.items():
        cls_idx = np.where(y_train_text.values == lbl)[0]
        if len(cls_idx) < 2:
            continue
        Xc = X_train_df.iloc[cls_idx][num_cols]
        lo = Xc.quantile(q_low)
        hi = Xc.quantile(q_high)
        bounds[lbl] = (lo, hi, cls_idx)

    synth_rows = []
    synth_labels = []

    for lbl, target_n in strategy.items():
        cur_n = int((y_out == lbl).sum())
        need = int(target_n - cur_n)
        if need <= 0:
            continue
        if lbl not in bounds:
            continue

        lo, hi, cls_idx = bounds[lbl]

        for _ in range(need):
            # pick two parents
            i1, i2 = rng.choice(cls_idx, size=2, replace=False)
            p1 = X_train_df.iloc[i1]
            p2 = X_train_df.iloc[i2]

            # categorical signature
            child = {}
            for c in CAT_COLS:
                child[c] = p1[c] if rng.random() < 0.5 else p2[c]

            # numeric: interpolate + noise + clamp
            x1 = p1[num_cols].values.astype(np.float32)
            x2 = p2[num_cols].values.astype(np.float32)

            alpha = rng.uniform(alpha_min, alpha_max)
            xn = alpha * x1 + (1 - alpha) * x2
            xn = xn + rng.normal(0.0, noise_sigma, size=xn.shape).astype(np.float32)

            # clamp to bounds
            xn = np.maximum(xn, lo.values.astype(np.float32))
            xn = np.minimum(xn, hi.values.astype(np.float32))

            for j, c in enumerate(num_cols):
                child[c] = xn[j]

            synth_rows.append(child)
            synth_labels.append(lbl)

    if len(synth_rows) > 0:
        X_syn = pd.DataFrame(synth_rows, columns=CAT_COLS + num_cols)  # order
        y_syn = pd.Series(synth_labels)

        X_out = pd.concat([X_out, X_syn], axis=0).reset_index(drop=True)
        y_out = pd.concat([y_out, y_syn], axis=0).reset_index(drop=True)

    return X_out, y_out

# ------------------------------------------------------------
# 8) COMMON ENCODING FUNCTION (align columns)
# ------------------------------------------------------------
def encode_and_scale(X_train_df, X_val_df, X_test_df):
    X_all = pd.get_dummies(pd.concat([X_train_df, X_val_df, X_test_df], axis=0), columns=CAT_COLS)
    n_tr = len(X_train_df); n_va = len(X_val_df)
    X_tr = X_all.iloc[:n_tr].reset_index(drop=True)
    X_va = X_all.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
    X_te = X_all.iloc[n_tr+n_va:].reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(-1,1))
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_va_s, X_te_s, X_tr.columns

# ------------------------------------------------------------
# 9) RUN EXPERIMENTS
# ------------------------------------------------------------
results_rows = []
per_attack_tables = {}

# sampling strategy (однакова логіка для smotenc і spas)
strategy, attack_counts = make_sampling_strategy(
    y_train_text_orig, TARGET_ATTACKS, MAX_MULTIPLIER, MIN_TARGET_COUNT
)

# Таблиця planned strategy
if len(strategy) > 0:
    plan_df = pd.DataFrame([
        [lbl, int(attack_counts[lbl]), int(strategy[lbl]), float(strategy[lbl] / max(1, int(attack_counts[lbl])))]
        for lbl in strategy.keys()
    ], columns=["attack_label","current","target","multiplier"])
else:
    plan_df = pd.DataFrame(columns=["attack_label","current","target","multiplier"])
save_table(plan_df, "table_4_3_sampling_plan.csv")

# bounds table example for DEMO_ATTACK_FOR_BOUNDS (8 features)
def make_bounds_table_for_attack(X_train_df, y_train_text, attack_label, feat_list, q_low, q_high):
    df_cls = X_train_df[y_train_text.values == attack_label]
    if len(df_cls) == 0:
        return pd.DataFrame(columns=["feature","q_low","q_high"])
    rows = []
    for f in feat_list:
        rows.append([f, float(df_cls[f].quantile(q_low)), float(df_cls[f].quantile(q_high))])
    return pd.DataFrame(rows, columns=["feature","q_low","q_high"])

bounds_tbl = make_bounds_table_for_attack(
    X_train_orig, y_train_text_orig, DEMO_ATTACK_FOR_BOUNDS, DEMO_FEATURES_8,
    SPAS_CFG["q_low"], SPAS_CFG["q_high"]
)
save_table(bounds_tbl, "table_4_3_bounds_example_8_features.csv")

# Plot bounds bars
if len(bounds_tbl) > 0:
    plt.figure(figsize=(10,4))
    x = np.arange(len(bounds_tbl))
    plt.errorbar(x, (bounds_tbl["q_low"]+bounds_tbl["q_high"])/2,
                 yerr=(bounds_tbl["q_high"]-bounds_tbl["q_low"])/2, fmt='o')
    plt.xticks(x, bounds_tbl["feature"], rotation=30, ha="right")
    plt.title(f"Percentile bounds for {DEMO_ATTACK_FOR_BOUNDS} (q={SPAS_CFG['q_low']}-{SPAS_CFG['q_high']})")
    plt.ylabel("Feature value")
    save_fig("fig_4_3_bounds_example.png")

for mode in RUN_MODES:
    print("\n==============================")
    print("RUN MODE:", mode)
    print("==============================")

    # 9.1 choose training set variant
    if mode == "baseline":
        X_train_use = X_train_orig.copy().reset_index(drop=True)
        y_train_use = y_train_text_orig.reset_index(drop=True)
        used_bal = "none"

    elif mode == "smotenc":
        X_train_use, y_train_use, used_bal = apply_smotenc(X_train_orig, y_train_text_orig, strategy)

    elif mode == "spas":
        X_train_use, y_train_use = spas_generate(
            X_train_orig, y_train_text_orig, strategy,
            q_low=SPAS_CFG["q_low"], q_high=SPAS_CFG["q_high"],
            alpha_min=SPAS_CFG["alpha_min"], alpha_max=SPAS_CFG["alpha_max"],
            noise_sigma=SPAS_CFG["noise_sigma"],
            max_tries=SPAS_CFG["max_tries"],
            seed=SPAS_CFG["random_state"]
        )
        used_bal = "SPAS"

    else:
        raise ValueError("Unknown mode")

    # 9.2 tables: counts after
    after_counts = y_train_use.value_counts().reset_index()
    after_counts.columns = ["label","count"]
    save_table(after_counts, f"table_4_3_counts_after_{mode}.csv")

    top_after = count_attacks(y_train_use).head(20)
    plt.figure(figsize=(10,5))
    plt.barh(top_after["attack_label"][::-1], top_after["count"][::-1])
    plt.title(f"Top attack types in train (after balancing) — {mode}")
    plt.xlabel("Count")
    save_fig(f"fig_4_3_top_attacks_after_{mode}.png")

    # 9.3 Encode + scale (align columns)
    X_tr_s, X_va_s, X_te_s, feat_cols = encode_and_scale(X_train_use, X_val_orig, X_test)

    y_tr_bin = np.array([label_to_binary(x) for x in y_train_use])
    y_va_bin = np.array([label_to_binary(x) for x in y_val_text])
    # y_test_bin already computed

    # 9.4 Sonification -> STFT (heavy)
    print("Generating STFT (train/val/test) ...")
    t0 = time.time()
    tr_spec = gen_specs_from_X(X_tr_s)
    va_spec = gen_specs_from_X(X_va_s)
    te_spec = gen_specs_from_X(X_te_s)
    stft_time = time.time() - t0
    print("STFT shapes:", tr_spec.shape, va_spec.shape, te_spec.shape, " time:", stft_time)

    tr_4d = tr_spec[..., np.newaxis]
    va_4d = va_spec[..., np.newaxis]
    te_4d = te_spec[..., np.newaxis]

    # 9.5 Model train/eval
    model = build_2dcnn(tr_4d.shape[1:])
    t0 = time.time()
    hist = model.fit(
        tr_4d, y_tr_bin,
        validation_data=(va_4d, y_va_bin),
        epochs=SON_CFG["epochs"],
        batch_size=SON_CFG["batch_size"],
        verbose=1
    )
    train_time = time.time() - t0

    probs = model.predict(te_4d, verbose=0).flatten()
    acc, prec, rec, f1, pred, cm = compute_metrics(y_test_bin, probs, SON_CFG["threshold"])

    print(f"[{mode}] Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print("CM:\n", cm)

    # 9.6 Save plots: learning curves (для 4.3 можна 1 раз або для кожного режиму)
    # loss
    plt.figure(figsize=(7,4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title(f"{mode}: loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    save_fig(f"fig_4_3_{mode}_history_loss.png")

    # acc
    plt.figure(figsize=(7,4))
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val")
    plt.title(f"{mode}: accuracy")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    save_fig(f"fig_4_3_{mode}_history_acc.png")

    # confusion matrix
    plt.figure(figsize=(4.5,4.5))
    plt.imshow(cm)
    plt.title(f"{mode}: confusion matrix")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.xticks([0,1], ["normal","attack"])
    plt.yticks([0,1], ["normal","attack"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    save_fig(f"fig_4_3_{mode}_confusion.png")

    # confidence bins
    plot_confidence_bins(
        y_test_bin, probs,
        f"{mode}: correct/incorrect by confidence",
        f"fig_4_3_{mode}_confidence_bins.png"
    )

    # per-attack recall table
    per_attack = per_attack_recall_table(y_test_text, y_test_bin, pred)
    per_attack_tables[mode] = per_attack
    save_table(per_attack, f"table_4_3_per_attack_recall_{mode}.csv")

    # store results row
    results_rows.append([
        mode, used_bal,
        len(X_train_use), stft_time, train_time,
        acc, prec, rec, f1,
        int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    ])

# ------------------------------------------------------------
# 10) FINAL COMPARISON TABLES
# ------------------------------------------------------------
res_df = pd.DataFrame(results_rows, columns=[
    "mode","balancing_method",
    "train_samples","stft_time_sec","train_time_sec",
    "accuracy","precision","recall","f1",
    "TN","FP","FN","TP"
])
save_table(res_df, "table_4_3_final_metrics_comparison.csv")
print(res_df)

# ------------------------------------------------------------
# 11) PER-ATTACK IMPROVEMENT TABLE (SPAS vs SMOTENC vs baseline)
# ------------------------------------------------------------
if ("baseline" in per_attack_tables) and ("spas" in per_attack_tables):
    base = per_attack_tables["baseline"].set_index("attack_label")
    spas = per_attack_tables["spas"].set_index("attack_label")
    smt  = per_attack_tables["smotenc"].set_index("attack_label") if "smotenc" in per_attack_tables else None

    # align
    common = base.index.intersection(spas.index)
    rows = []
    for lbl in common:
        sup = int(base.loc[lbl, "support"])
        r0 = float(base.loc[lbl, "recall"])
        rS = float(spas.loc[lbl, "recall"])
        rM = float(smt.loc[lbl, "recall"]) if smt is not None and lbl in smt.index else np.nan
        rows.append([lbl, sup, r0, rM, rS, (rS - r0)])

    imp = pd.DataFrame(rows, columns=[
        "attack_label","support",
        "recall_baseline","recall_smotenc","recall_spas","delta_spas_minus_base"
    ]).sort_values("delta_spas_minus_base", ascending=False)

    save_table(imp, "table_4_3_per_attack_improvement_spas.csv")

    # plot top-20 improvements
    top20 = imp.head(20).copy()
    plt.figure(figsize=(10,5))
    plt.barh(top20["attack_label"][::-1], top20["delta_spas_minus_base"][::-1])
    plt.title("Top-20 recall improvements: SPAS vs baseline (per-attack)")
    plt.xlabel("Δ recall")
    save_fig("fig_4_3_top20_recall_improvement_spas.png")

# ------------------------------------------------------------
# 12) SYNTHETIC EXAMPLES: Real vs SMOTENC vs SPAS (for 1 attack label)
# ------------------------------------------------------------
# Це легка демонстрація: беремо кілька зразків DEMO_ATTACK_FOR_SYNTH із train_orig,
# генеруємо кілька синтетичних SPAS і, якщо є SMOTENC — показуємо 1-2 синтетики.
# (Візуалізуємо спектрограми)
print("\nGenerating synthetic examples for visualization...")

# базові зразки
idx_cls = np.where(y_train_text_orig.values == DEMO_ATTACK_FOR_SYNTH)[0]
if len(idx_cls) >= 2:
    pick = idx_cls[:N_SYNTH_EXAMPLES]
    # закодуємо базовий (для спектрограми треба шлях через encode_and_scale)
    # зробимо маленький батч: (real + few SPAS synth)
    X_base = X_train_orig.iloc[pick].copy().reset_index(drop=True)
    y_base = y_train_text_orig.iloc[pick].copy().reset_index(drop=True)

    # згенеруємо SPAS синтетику тільки для цього класу: strategy на 1 клас
    tmp_counts = int((y_base == DEMO_ATTACK_FOR_SYNTH).sum())
    tmp_strategy = {DEMO_ATTACK_FOR_SYNTH: tmp_counts + N_SYNTH_EXAMPLES}
    X_spas_v, y_spas_v = spas_generate(
        X_train_orig, y_train_text_orig, tmp_strategy,
        q_low=SPAS_CFG["q_low"], q_high=SPAS_CFG["q_high"],
        alpha_min=SPAS_CFG["alpha_min"], alpha_max=SPAS_CFG["alpha_max"],
        noise_sigma=SPAS_CFG["noise_sigma"],
        max_tries=SPAS_CFG["max_tries"],
        seed=SPAS_CFG["random_state"]
    )
    # витягнемо лише нові (останнє)
    X_spas_new = X_spas_v.iloc[-N_SYNTH_EXAMPLES:].copy().reset_index(drop=True)

    # SMOTENC synthetic (опційно)
    X_smote_new = None
    try:
        tmp_strategy2, _ = make_sampling_strategy(y_train_text_orig, [DEMO_ATTACK_FOR_SYNTH], 2, tmp_counts + N_SYNTH_EXAMPLES)
        X_smote_all, y_smote_all, used = apply_smotenc(X_train_orig, y_train_text_orig, tmp_strategy2)
        if used != "none":
            # виберемо кілька останніх з цим label (евристика)
            sm_idx = np.where(y_smote_all.values == DEMO_ATTACK_FOR_SYNTH)[0]
            if len(sm_idx) > tmp_counts:
                X_smote_new = X_smote_all.iloc[sm_idx[-N_SYNTH_EXAMPLES:]].copy().reset_index(drop=True)
    except Exception as e:
        print("SMOTENC viz skip:", e)

    # сформуємо набір для енкодінгу: real + spas + smote
    blocks = [X_base]
    labels = ["Real"] * len(X_base)

    blocks.append(X_spas_new)
    labels += ["SPAS"] * len(X_spas_new)

    if X_smote_new is not None:
        blocks.append(X_smote_new)
        labels += ["SMOTENC"] * len(X_smote_new)

    X_viz = pd.concat(blocks, axis=0).reset_index(drop=True)

    # encode+scale using val/test as empty placeholders (hack)
    X_tr_s, _, _, _ = encode_and_scale(X_viz, X_viz.iloc[:0], X_viz.iloc[:0])
    # spectrograms
    specs = gen_specs_from_X(X_tr_s)

    # plot
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(specs[i], aspect='auto', origin='lower')
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Freq")
    plt.suptitle(f"Spectrograms: Real vs SPAS vs SMOTENC ({DEMO_ATTACK_FOR_SYNTH})")
    save_fig("fig_4_3_real_vs_synth_spectrograms.png")

else:
    print("Not enough samples for DEMO_ATTACK_FOR_SYNTH to visualize.")

print("\n=== DONE 4.3 ===")
print("Artifacts saved in:", OUT_DIR)






































# @title
# ============================================================
# РОЗДІЛ 4.4
# SPAS + ErrorBoost + τ-zone + additional model + selective override
# Binary IDS: normal vs attack
# Соніфікація + STFT + 2D-CNN
# ============================================================

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from scipy.signal import stft

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
train_path = '/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt'
test_path  = '/content/drive/MyDrive/nsl-kdd/KDDTest+.txt'

OUT_DIR = '/content/nsl_kdd_section_4_4'
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE_VAL = 0.1
MAX_TRAIN_SAMPLES_FOR_DEBUG = None   # для дебагу можна поставити число

# ---- SPAS config ----
TARGET_ATTACKS = [
 'snmpgetattack','snmpguess','phf','xsnoop','ps','sendmail','xterm',
 'buffer_overflow','xlock','loadmodule','udpstorm','imap','worm',
 'sqlattack','perl','mailbomb','processtable','rootkit','guess_passwd',
 'multihop','warezmaster','back','named'
]

MAX_MULTIPLIER = 5
MIN_TARGET_COUNT = 1000

SPAS_CFG = {
    "q_low": 0.05,
    "q_high": 0.95,
    "alpha_min": 0.2,
    "alpha_max": 0.8,
    "noise_sigma": 0.03,
    "max_tries": 50,
    "random_state": RANDOM_STATE
}

# ---- ErrorBoost config ----
ERRORBOOST_CFG = {
    "use_error_boost": True,
    "multiply_errored_examples_by": 20,
    "exclude_normal": True,
    "use_only_attack_labels": True,
    "top_k_labels": 50,
    "min_err_count": 10
}

# ---- τ-zone + confidence gating ----
TAU_CFG = {
    "tau_l_grid": np.round(np.arange(0.20, 0.41, 0.05), 2),   # baseline prob lower bound candidates
    "tau_u_grid": np.round(np.arange(0.60, 0.81, 0.05), 2),   # baseline prob upper bound candidates
    "gate_thr_lo_grid": np.round(np.arange(0.30, 0.51, 0.02), 2),
    "gate_thr_hi_grid": np.round(np.arange(0.50, 0.71, 0.02), 2),
    "threshold_base": 0.5,
    "threshold_boost": 0.5
}

# ---- Sonification config ----
SON_CFG = {
    "sample_rate": 2000,
    "record_dur": 0.4,
    "f_min": 200,
    "f_max": 2000,
    "nperseg": 256,
    "noverlap": 128,
    "epochs_base": 5,
    "epochs_boost": 5,
    "batch_size": 128,
    "lr": 0.0005
}

# ---- Demo for plots ----
DEMO_ATTACK_LABEL = "guess_passwd"

# save configs
with open(os.path.join(OUT_DIR, 'spas_cfg.json'), 'w', encoding='utf-8') as f:
    json.dump(SPAS_CFG, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, 'errorboost_cfg.json'), 'w', encoding='utf-8') as f:
    json.dump(ERRORBOOST_CFG, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, 'tau_cfg.json'), 'w', encoding='utf-8') as f:
    json.dump({k: list(v) if isinstance(v, np.ndarray) else v for k, v in TAU_CFG.items()}, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, 'son_cfg.json'), 'w', encoding='utf-8') as f:
    json.dump(SON_CFG, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# 2. COLUMNS
# ------------------------------------------------------------
column_names = [
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

CAT_COLS = ['protocol_type', 'service', 'flag']

# ------------------------------------------------------------
# 3. HELPERS
# ------------------------------------------------------------
def label_to_binary(lbl):
    return 0 if lbl == 'normal' else 1

def save_table(df, filename):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Saved table: {path}")

def save_fig(filename):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved fig: {path}")
    plt.show()

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    return acc, float(p), float(r), float(f1), y_pred, cm

def plot_history(history, prefix, title):
    plt.figure(figsize=(7,4))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title}: loss')
    plt.legend()
    save_fig(f'{prefix}_loss.png')

    plt.figure(figsize=(7,4))
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title}: accuracy')
    plt.legend()
    save_fig(f'{prefix}_accuracy.png')

def plot_confusion(cm, title, filename):
    plt.figure(figsize=(4.5,4.5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0,1], ['normal','attack'])
    plt.yticks([0,1], ['normal','attack'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    save_fig(filename)

def plot_confidence_bins(y_true, y_prob, title, filename):
    y_pred = (y_prob >= 0.5).astype(int)
    correct = (y_pred == y_true)

    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, len(bins)-2)

    correct_counts = np.zeros(len(bins)-1, dtype=int)
    wrong_counts = np.zeros(len(bins)-1, dtype=int)

    for i in range(len(y_prob)):
        if correct[i]:
            correct_counts[bin_ids[i]] += 1
        else:
            wrong_counts[bin_ids[i]] += 1

    labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    x = np.arange(len(labels))

    plt.figure(figsize=(11,4))
    plt.bar(x, correct_counts, label='correct')
    plt.bar(x, wrong_counts, bottom=correct_counts, label='incorrect')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.xlabel('Predicted probability interval')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    save_fig(filename)

def per_attack_recall_table(y_test_text, y_true_bin, y_pred_bin):
    df = pd.DataFrame({
        'true_text': pd.Series(y_test_text).reset_index(drop=True),
        'y_true': np.array(y_true_bin).flatten(),
        'y_pred': np.array(y_pred_bin).flatten()
    })
    df_attack = df[df['y_true'] == 1].copy()
    if len(df_attack) == 0:
        return pd.DataFrame(columns=['attack_label','support','TP','FN','recall'])

    out = df_attack.groupby('true_text').apply(
        lambda g: pd.Series({
            'support': int(len(g)),
            'TP': int((g['y_pred'] == 1).sum()),
            'FN': int((g['y_pred'] == 0).sum()),
            'recall': float((g['y_pred'] == 1).sum() / len(g))
        })
    ).reset_index().rename(columns={'true_text':'attack_label'}).sort_values('support', ascending=False)

    return out

def make_sampling_strategy(y_train_text, targets, max_multiplier, min_target_count):
    counts = y_train_text.value_counts()
    attack_counts = counts[counts.index != 'normal']

    if len(attack_counts) == 0:
        return {}, attack_counts

    available = set(attack_counts.index.tolist())
    targets_set = set(targets) & available if targets is not None else available

    max_attack_count = int(attack_counts.max())
    strategy = {}

    for lbl, c in attack_counts.items():
        if lbl not in targets_set:
            continue
        base_target = min(int(c * max_multiplier), max_attack_count)
        if base_target < min_target_count:
            base_target = min(max_attack_count, int(min_target_count))
        if base_target > c:
            strategy[lbl] = base_target

    return strategy, attack_counts

# ------------------------------------------------------------
# 4. SPAS
# ------------------------------------------------------------
def spas_generate(
    X_train_df, y_train_text, strategy,
    q_low, q_high, alpha_min, alpha_max, noise_sigma, max_tries, seed
):
    rng = np.random.default_rng(seed)

    if len(strategy) == 0:
        return X_train_df.copy().reset_index(drop=True), y_train_text.reset_index(drop=True)

    num_cols = [c for c in X_train_df.columns if c not in CAT_COLS]

    X_out = X_train_df.copy().reset_index(drop=True)
    y_out = y_train_text.copy().reset_index(drop=True)

    bounds = {}
    for lbl, target_n in strategy.items():
        cls_idx = np.where(y_train_text.values == lbl)[0]
        if len(cls_idx) < 2:
            continue
        Xc = X_train_df.iloc[cls_idx][num_cols]
        lo = Xc.quantile(q_low)
        hi = Xc.quantile(q_high)
        bounds[lbl] = (lo, hi, cls_idx)

    synth_rows = []
    synth_labels = []

    for lbl, target_n in strategy.items():
        cur_n = int((y_out == lbl).sum())
        need = int(target_n - cur_n)
        if need <= 0:
            continue
        if lbl not in bounds:
            continue

        lo, hi, cls_idx = bounds[lbl]

        for _ in range(need):
            i1, i2 = rng.choice(cls_idx, size=2, replace=False)
            p1 = X_train_df.iloc[i1]
            p2 = X_train_df.iloc[i2]

            child = {}
            for c in CAT_COLS:
                child[c] = p1[c] if rng.random() < 0.5 else p2[c]

            x1 = p1[num_cols].values.astype(np.float32)
            x2 = p2[num_cols].values.astype(np.float32)

            alpha = rng.uniform(alpha_min, alpha_max)
            xn = alpha * x1 + (1 - alpha) * x2
            xn = xn + rng.normal(0.0, noise_sigma, size=xn.shape).astype(np.float32)

            xn = np.maximum(xn, lo.values.astype(np.float32))
            xn = np.minimum(xn, hi.values.astype(np.float32))

            for j, c in enumerate(num_cols):
                child[c] = xn[j]

            synth_rows.append(child)
            synth_labels.append(lbl)

    if len(synth_rows) > 0:
        num_cols = [c for c in X_train_df.columns if c not in CAT_COLS]
        ordered_cols = CAT_COLS + num_cols
        X_syn = pd.DataFrame(synth_rows)[ordered_cols]
        y_syn = pd.Series(synth_labels)

        X_out = pd.concat([X_out, X_syn], axis=0).reset_index(drop=True)
        y_out = pd.concat([y_out, y_syn], axis=0).reset_index(drop=True)

    return X_out, y_out

# ------------------------------------------------------------
# 5. ENCODING + SONIFICATION
# ------------------------------------------------------------
def encode_and_scale(X_train_df, X_val_df, X_test_df):
    X_all = pd.get_dummies(pd.concat([X_train_df, X_val_df, X_test_df], axis=0), columns=CAT_COLS)

    n_train = len(X_train_df)
    n_val = len(X_val_df)

    X_train_enc = X_all.iloc[:n_train].reset_index(drop=True)
    X_val_enc   = X_all.iloc[n_train:n_train+n_val].reset_index(drop=True)
    X_test_enc  = X_all.iloc[n_train+n_val:].reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(-1,1))
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_val_scaled   = scaler.transform(X_val_enc)
    X_test_scaled  = scaler.transform(X_test_enc)

    return X_train_scaled, X_val_scaled, X_test_scaled, X_train_enc.columns

sample_rate = SON_CFG["sample_rate"]
record_dur = SON_CFG["record_dur"]
samples_per_record = int(sample_rate * record_dur)
f_min = SON_CFG["f_min"]
f_max = SON_CFG["f_max"]

def features_to_wave(features):
    F = len(features)
    wave = np.zeros(samples_per_record, dtype=np.float32)
    samples_per_feat = max(1, samples_per_record // F)
    idx = 0
    for x in features:
        freq = f_min + x * (f_max - f_min)
        amp = x
        for s in range(samples_per_feat):
            if idx >= samples_per_record:
                break
            t_local = s / sample_rate
            wave[idx] = amp * np.sin(2*np.pi*freq*t_local)
            idx += 1
    if idx < samples_per_record:
        wave[idx:] = 0.0
    return wave

def wave_to_stft_image(wave):
    f, t, Zxx = stft(
        wave,
        fs=sample_rate,
        nperseg=SON_CFG["nperseg"],
        noverlap=SON_CFG["noverlap"]
    )
    spec = np.log1p(np.abs(Zxx)).astype(np.float32)
    return spec

def gen_specs_from_X(X_scaled):
    specs = []
    for i in range(X_scaled.shape[0]):
        specs.append(wave_to_stft_image(features_to_wave(X_scaled[i])))
    return np.stack(specs, axis=0)

def build_2dcnn(input_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.GlobalMaxPooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=SON_CFG["lr"]),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ------------------------------------------------------------
# 6. LOAD DATA
# ------------------------------------------------------------
train_data = pd.read_csv(train_path, header=None, names=column_names).reset_index(drop=True)
test_data  = pd.read_csv(test_path, header=None, names=column_names).reset_index(drop=True)

train_data["row_id"] = np.arange(len(train_data))
test_data["row_id"]  = np.arange(len(test_data))

if MAX_TRAIN_SAMPLES_FOR_DEBUG is not None:
    train_data = train_data.iloc[:MAX_TRAIN_SAMPLES_FOR_DEBUG].reset_index(drop=True)

print("Train shape:", train_data.shape, "Test shape:", test_data.shape)

X_all = train_data.drop(['label','difficulty_level'], axis=1).copy()
y_all_text = train_data['label'].copy()
row_id_all = train_data["row_id"].copy()

X_test_df = test_data.drop(['label','difficulty_level'], axis=1).copy()
y_test_text = test_data['label'].copy()
row_id_test = test_data["row_id"].copy()
y_test_bin  = y_test_text.apply(label_to_binary).values

y_all_bin = y_all_text.apply(label_to_binary).values

X_train_orig, X_val_orig, y_train_text_orig, y_val_text, row_id_train_orig, row_id_val = train_test_split(
    X_all, y_all_text, row_id_all,
    test_size=TEST_SIZE_VAL,
    stratify=y_all_bin,
    random_state=RANDOM_STATE
)

y_val_bin = y_val_text.apply(label_to_binary).values

print("Train_orig:", X_train_orig.shape, "Val_orig:", X_val_orig.shape)

# ------------------------------------------------------------
# 7. BASELINE SPAS TRAIN SET
# ------------------------------------------------------------
base_strategy, base_attack_counts = make_sampling_strategy(
    y_train_text_orig, TARGET_ATTACKS, MAX_MULTIPLIER, MIN_TARGET_COUNT
)

table_strategy = pd.DataFrame([
    [lbl, int(base_attack_counts[lbl]), int(base_strategy[lbl]), round(base_strategy[lbl]/base_attack_counts[lbl], 2)]
    for lbl in base_strategy.keys()
], columns=['attack_label','current_count','target_count','multiplier'])

save_table(table_strategy, 'table_4_4_spas_sampling_plan.csv')

X_train_spas, y_train_spas = spas_generate(
    X_train_orig, y_train_text_orig, base_strategy,
    q_low=SPAS_CFG["q_low"],
    q_high=SPAS_CFG["q_high"],
    alpha_min=SPAS_CFG["alpha_min"],
    alpha_max=SPAS_CFG["alpha_max"],
    noise_sigma=SPAS_CFG["noise_sigma"],
    max_tries=SPAS_CFG["max_tries"],
    seed=SPAS_CFG["random_state"]
)

print("SPAS train shape:", X_train_spas.shape)

before_counts = y_train_text_orig.value_counts().reset_index()
before_counts.columns = ['label','count']
after_counts = y_train_spas.value_counts().reset_index()
after_counts.columns = ['label','count']

save_table(before_counts, 'table_4_4_counts_before_spas.csv')
save_table(after_counts, 'table_4_4_counts_after_spas.csv')

# plot before/after for target attacks
top_before = before_counts[before_counts['label'] != 'normal'].head(20)
top_after = after_counts[after_counts['label'] != 'normal'].head(20)

plt.figure(figsize=(10,5))
plt.barh(top_before['label'][::-1], top_before['count'][::-1])
plt.title('Train attack distribution before SPAS')
plt.xlabel('Count')
save_fig('fig_4_4_top_attacks_before_spas.png')

plt.figure(figsize=(10,5))
plt.barh(top_after['label'][::-1], top_after['count'][::-1])
plt.title('Train attack distribution after SPAS')
plt.xlabel('Count')
save_fig('fig_4_4_top_attacks_after_spas.png')

# ------------------------------------------------------------
# 8. BASELINE MODEL (trained on SPAS-balanced train)
# ------------------------------------------------------------
X_train_scaled_base, X_val_scaled_base, X_test_scaled_base, feat_cols_base = encode_and_scale(
    X_train_spas, X_val_orig, X_test_df
)

y_train_bin_base = np.array([label_to_binary(x) for x in y_train_spas])

print("Generating STFT for baseline model...")
t0 = time.time()
train_stft_base = gen_specs_from_X(X_train_scaled_base)
val_stft_base   = gen_specs_from_X(X_val_scaled_base)
test_stft_base  = gen_specs_from_X(X_test_scaled_base)
stft_time_base = time.time() - t0

train_4d_base = train_stft_base[..., np.newaxis]
val_4d_base   = val_stft_base[..., np.newaxis]
test_4d_base  = test_stft_base[..., np.newaxis]

baseline_model = build_2dcnn(train_4d_base.shape[1:])
baseline_model.summary()

t0 = time.time()
history_base = baseline_model.fit(
    train_4d_base, y_train_bin_base,
    validation_data=(val_4d_base, y_val_bin),
    epochs=SON_CFG["epochs_base"],
    batch_size=SON_CFG["batch_size"],
    verbose=1
)
train_time_base = time.time() - t0

base_prob_train = baseline_model.predict(train_4d_base, verbose=0).flatten()
base_prob_val   = baseline_model.predict(val_4d_base, verbose=0).flatten()
base_prob_test  = baseline_model.predict(test_4d_base, verbose=0).flatten()

base_acc, base_prec, base_rec, base_f1, base_pred_test, cm_base_test = compute_metrics(
    y_test_bin, base_prob_test, TAU_CFG["threshold_base"]
)

print("\n=== BASELINE SPAS MODEL (FULL TEST) ===")
print(f"Acc={base_acc:.4f} P={base_prec:.4f} R={base_rec:.4f} F1={base_f1:.4f}")
print("Confusion matrix:\n", cm_base_test)
print(classification_report(y_test_bin, base_pred_test, digits=4))

plot_history(history_base, 'fig_4_4_base_history', 'Baseline SPAS model')
plot_confusion(cm_base_test, 'Baseline SPAS model: confusion matrix', 'fig_4_4_base_confusion.png')
plot_confidence_bins(
    y_test_bin, base_prob_test,
    'Baseline SPAS model: correct / incorrect by confidence',
    'fig_4_4_base_confidence_bins.png'
)

# baseline passports
base_pass_train = pd.DataFrame({
    'row_id': np.arange(len(X_train_spas)),
    'y_true': y_train_bin_base,
    'p1': base_prob_train,
    'y_pred': (base_prob_train >= TAU_CFG["threshold_base"]).astype(int),
    'true_text': pd.Series(y_train_spas).reset_index(drop=True)
})
base_pass_val = pd.DataFrame({
    'row_id': row_id_val.reset_index(drop=True).astype(int),
    'y_true': y_val_bin,
    'p1': base_prob_val,
    'y_pred': (base_prob_val >= TAU_CFG["threshold_base"]).astype(int),
    'true_text': pd.Series(y_val_text).reset_index(drop=True)
})
base_pass_test = pd.DataFrame({
    'row_id': row_id_test.reset_index(drop=True).astype(int),
    'y_true': y_test_bin,
    'p1': base_prob_test,
    'y_pred': (base_prob_test >= TAU_CFG["threshold_base"]).astype(int),
    'true_text': pd.Series(y_test_text).reset_index(drop=True)
})

save_table(base_pass_val[['row_id','y_true','p1','y_pred','true_text']], 'passport_base_val.csv')
save_table(base_pass_test[['row_id','y_true','p1','y_pred','true_text']], 'passport_base_test.csv')

# ------------------------------------------------------------
# 9. ERROR ANALYSIS: labels to boost
# ------------------------------------------------------------
df_err = pd.concat([base_pass_train, base_pass_val], axis=0, ignore_index=True)
df_err['is_error'] = (df_err['y_true'].astype(int) != df_err['y_pred'].astype(int)).astype(int)

err_counts = (df_err[df_err['is_error'] == 1]
              .groupby('true_text')
              .size()
              .sort_values(ascending=False))

if ERRORBOOST_CFG["exclude_normal"] and 'normal' in err_counts.index:
    err_counts = err_counts.drop('normal')

err_counts = err_counts[err_counts >= ERRORBOOST_CFG["min_err_count"]]
labels_to_boost = err_counts.head(ERRORBOOST_CFG["top_k_labels"]).index.astype(str).tolist()

if ERRORBOOST_CFG["use_only_attack_labels"]:
    labels_to_boost = [x for x in labels_to_boost if x != 'normal']

table_top_errors = err_counts.reset_index()
table_top_errors.columns = ['attack_label','error_count']
save_table(table_top_errors, 'table_4_4_top_errored_labels.csv')

plt.figure(figsize=(10,5))
top_err = table_top_errors.head(20)
plt.barh(top_err['attack_label'][::-1], top_err['error_count'][::-1])
plt.title('Top errored attack labels (train+val) for baseline SPAS model')
plt.xlabel('Error count')
save_fig('fig_4_4_top_errored_labels.png')

print("Labels to boost:", labels_to_boost[:20], "... total:", len(labels_to_boost))

# ------------------------------------------------------------
# 10. ERRORBOOST TRAIN SET
# ------------------------------------------------------------
# Починаємо від SPAS-balanced train і ще раз посилюємо проблемні підкласи
spas_counts = y_train_spas.value_counts()
spas_attack_counts = spas_counts[spas_counts.index != 'normal']

boost_strategy = {}
if len(spas_attack_counts) > 0:
    max_attack_count_boost = int(spas_attack_counts.max())
    for lbl in labels_to_boost:
        cur = int(spas_counts.get(lbl, 0))
        if cur <= 0:
            continue
        tgt = min(int(cur * ERRORBOOST_CFG["multiply_errored_examples_by"]), max_attack_count_boost)
        if tgt > cur:
            boost_strategy[lbl] = tgt

table_boost_plan = pd.DataFrame([
    [lbl, int(spas_counts[lbl]), int(boost_strategy[lbl]), round(boost_strategy[lbl] / spas_counts[lbl], 2)]
    for lbl in boost_strategy.keys()
], columns=['attack_label','current_count','target_count','multiplier'])
save_table(table_boost_plan, 'table_4_4_errorboost_sampling_plan.csv')

# ErrorBoost = ще один SPAS, але тільки по помилкових labels
X_train_boost, y_train_boost = spas_generate(
    X_train_spas, y_train_spas, boost_strategy,
    q_low=SPAS_CFG["q_low"],
    q_high=SPAS_CFG["q_high"],
    alpha_min=SPAS_CFG["alpha_min"],
    alpha_max=SPAS_CFG["alpha_max"],
    noise_sigma=SPAS_CFG["noise_sigma"],
    max_tries=SPAS_CFG["max_tries"],
    seed=SPAS_CFG["random_state"] + 1
)

boost_counts = y_train_boost.value_counts().reset_index()
boost_counts.columns = ['label','count']
save_table(boost_counts, 'table_4_4_counts_after_errorboost.csv')

# Compare SPAS train vs ErrorBoost train
cmp_rows = []
for lbl in sorted(set(y_train_spas.unique()) | set(y_train_boost.unique())):
    cmp_rows.append([
        lbl,
        int((y_train_spas == lbl).sum()),
        int((y_train_boost == lbl).sum())
    ])
table_cmp_boost = pd.DataFrame(cmp_rows, columns=['label','count_after_spas','count_after_errorboost'])
save_table(table_cmp_boost, 'table_4_4_compare_spas_vs_errorboost_counts.csv')

# plot boosted labels
top_boosted = table_cmp_boost.copy()
top_boosted['delta'] = top_boosted['count_after_errorboost'] - top_boosted['count_after_spas']
top_boosted = top_boosted[top_boosted['delta'] > 0].sort_values('delta', ascending=False).head(20)

plt.figure(figsize=(10,5))
plt.barh(top_boosted['label'][::-1], top_boosted['delta'][::-1])
plt.title('Increase in train samples after ErrorBoost')
plt.xlabel('Additional samples')
save_fig('fig_4_4_errorboost_delta_counts.png')

# ------------------------------------------------------------
# 11. BOOST MODEL
# ------------------------------------------------------------
X_train_scaled_boost, X_val_scaled_boost, X_test_scaled_boost, feat_cols_boost = encode_and_scale(
    X_train_boost, X_val_orig, X_test_df
)

y_train_bin_boost = np.array([label_to_binary(x) for x in y_train_boost])

print("Generating STFT for boost model...")
t0 = time.time()
train_stft_boost = gen_specs_from_X(X_train_scaled_boost)
val_stft_boost   = gen_specs_from_X(X_val_scaled_boost)
test_stft_boost  = gen_specs_from_X(X_test_scaled_boost)
stft_time_boost = time.time() - t0

train_4d_boost = train_stft_boost[..., np.newaxis]
val_4d_boost   = val_stft_boost[..., np.newaxis]
test_4d_boost  = test_stft_boost[..., np.newaxis]

boost_model = build_2dcnn(train_4d_boost.shape[1:])
boost_model.summary()

t0 = time.time()
history_boost = boost_model.fit(
    train_4d_boost, y_train_bin_boost,
    validation_data=(val_4d_boost, y_val_bin),
    epochs=SON_CFG["epochs_boost"],
    batch_size=SON_CFG["batch_size"],
    verbose=1
)
train_time_boost = time.time() - t0

boost_prob_val = boost_model.predict(val_4d_boost, verbose=0).flatten()
boost_prob_test = boost_model.predict(test_4d_boost, verbose=0).flatten()

boost_acc_full, boost_prec_full, boost_rec_full, boost_f1_full, boost_pred_test, cm_boost_test = compute_metrics(
    y_test_bin, boost_prob_test, TAU_CFG["threshold_boost"]
)

print("\n=== BOOST MODEL (FULL TEST) ===")
print(f"Acc={boost_acc_full:.4f} P={boost_prec_full:.4f} R={boost_rec_full:.4f} F1={boost_f1_full:.4f}")
print("Confusion matrix:\n", cm_boost_test)
print(classification_report(y_test_bin, boost_pred_test, digits=4))

plot_history(history_boost, 'fig_4_4_boost_history', 'Boost model')
plot_confusion(cm_boost_test, 'Boost model: confusion matrix', 'fig_4_4_boost_confusion.png')
plot_confidence_bins(
    y_test_bin, boost_prob_test,
    'Boost model: correct / incorrect by confidence',
    'fig_4_4_boost_confidence_bins.png'
)

# ------------------------------------------------------------
# 12. AUTO PICK τ-ZONE + GATING ON VALIDATION
# ------------------------------------------------------------
base_val_df = base_pass_val[['row_id','y_true','p1','y_pred','true_text']].copy()
base_val_df['p1_boost'] = boost_prob_val.astype(float)

def apply_confidence_gating(df, in_tau_mask, thr_lo, thr_hi):
    y_base = df['y_pred'].astype(int).values.copy()
    p2 = df['p1_boost'].astype(float).values

    use_hi = in_tau_mask & (p2 >= thr_hi)
    use_lo = in_tau_mask & (p2 <= thr_lo)
    used_boost = use_hi | use_lo

    y_mix = y_base.copy()
    y_mix[use_hi] = 1
    y_mix[use_lo] = 0
    return y_mix, used_boost

best = None
grid_rows = []

for tau_l in TAU_CFG["tau_l_grid"]:
    for tau_u in TAU_CFG["tau_u_grid"]:
        if tau_l >= tau_u:
            continue

        in_tau_val = ((base_val_df['p1'] >= tau_l) & (base_val_df['p1'] <= tau_u)).values
        tau_size = int(in_tau_val.sum())
        if tau_size == 0:
            continue

        for thr_lo in TAU_CFG["gate_thr_lo_grid"]:
            for thr_hi in TAU_CFG["gate_thr_hi_grid"]:
                if thr_lo >= thr_hi:
                    continue

                y_mix_val, used_boost_mask_val = apply_confidence_gating(base_val_df, in_tau_val, thr_lo, thr_hi)

                y_true_tau = base_val_df.loc[in_tau_val, 'y_true'].astype(int).values
                y_base_tau = base_val_df.loc[in_tau_val, 'y_pred'].astype(int).values
                y_mix_tau  = y_mix_val[in_tau_val]

                err0 = int((y_true_tau != y_base_tau).sum())
                err1 = int((y_true_tau != y_mix_tau).sum())

                fp0 = int(((y_true_tau == 0) & (y_base_tau == 1)).sum())
                fn0 = int(((y_true_tau == 1) & (y_base_tau == 0)).sum())
                fp1 = int(((y_true_tau == 0) & (y_mix_tau == 1)).sum())
                fn1 = int(((y_true_tau == 1) & (y_mix_tau == 0)).sum())

                used_frac = float(used_boost_mask_val[in_tau_val].mean())
                objective = err1 + 0.01 * used_frac * len(y_true_tau)

                grid_rows.append([
                    tau_l, tau_u, thr_lo, thr_hi, tau_size,
                    err0, err1, fp0, fn0, fp1, fn1,
                    used_frac, objective
                ])

                cand = (
                    objective, err1, used_frac,
                    tau_l, tau_u, thr_lo, thr_hi,
                    tau_size, err0, err1, fp0, fn0, fp1, fn1
                )
                if (best is None) or (cand[0] < best[0]):
                    best = cand

grid_df = pd.DataFrame(grid_rows, columns=[
    'tau_l','tau_u','thr_lo','thr_hi','tau_size',
    'err_base','err_mix','fp_base','fn_base','fp_mix','fn_mix',
    'used_frac','objective'
])
save_table(grid_df, 'table_4_4_tau_gating_grid_search.csv')

(
    best_obj, best_err1, best_used_frac,
    TAU_L, TAU_U, GATE_THR_LO, GATE_THR_HI,
    tau_size_best, err0_best, err1_best, fp0_best, fn0_best, fp1_best, fn1_best
) = best

print("\n=== BEST VAL SETTINGS ===")
print(f"TAU_L={TAU_L:.2f}, TAU_U={TAU_U:.2f}")
print(f"GATE_THR_LO={GATE_THR_LO:.2f}, GATE_THR_HI={GATE_THR_HI:.2f}")
print(f"VAL tau_size={tau_size_best}, err {err0_best}->{err1_best}, used_frac={best_used_frac:.4f}")

table_best = pd.DataFrame([[
    TAU_L, TAU_U, GATE_THR_LO, GATE_THR_HI,
    tau_size_best, err0_best, err1_best, fp0_best, fn0_best, fp1_best, fn1_best, best_used_frac
]], columns=[
    'tau_l','tau_u','gate_thr_lo','gate_thr_hi',
    'tau_size_val','err_base_val','err_mix_val',
    'fp_base_val','fn_base_val','fp_mix_val','fn_mix_val','used_frac_val'
])
save_table(table_best, 'table_4_4_best_tau_gating_params.csv')

# plot probability distribution and tau zone on val
plt.figure(figsize=(10,4))
plt.hist(base_val_df['p1'], bins=40)
plt.axvline(TAU_L, linestyle='--')
plt.axvline(TAU_U, linestyle='--')
plt.title('Validation baseline probability distribution with selected τ-zone')
plt.xlabel('Baseline predicted probability')
plt.ylabel('Count')
save_fig('fig_4_4_val_tau_zone_distribution.png')

# top attack types in tau-zone on val
in_tau_val_best = ((base_val_df['p1'] >= TAU_L) & (base_val_df['p1'] <= TAU_U))
tau_attack_counts_val = base_val_df.loc[in_tau_val_best & (base_val_df['true_text'] != 'normal'), 'true_text'].value_counts().reset_index()
tau_attack_counts_val.columns = ['attack_label','count']
save_table(tau_attack_counts_val, 'table_4_4_top_attack_types_in_tau_val.csv')

plt.figure(figsize=(10,5))
top_tau_val = tau_attack_counts_val.head(20)
plt.barh(top_tau_val['attack_label'][::-1], top_tau_val['count'][::-1])
plt.title('Top attack types in τ-zone on validation set')
plt.xlabel('Count')
save_fig('fig_4_4_top_attacks_in_tau_val.png')

# ------------------------------------------------------------
# 13. APPLY MIX ON TEST
# ------------------------------------------------------------
mix_test_df = base_pass_test[['row_id','y_true','p1','y_pred','true_text']].copy()
mix_test_df['p1_boost'] = boost_prob_test.astype(float)

in_tau_test = ((mix_test_df['p1'] >= TAU_L) & (mix_test_df['p1'] <= TAU_U)).values
print("τ-zone size (test):", int(in_tau_test.sum()))

y_pred_mix, used_boost_mask = apply_confidence_gating(
    mix_test_df, in_tau_test, GATE_THR_LO, GATE_THR_HI
)

mix_test_df['used_boost'] = used_boost_mask.astype(int)
mix_test_df['y_pred_mix'] = y_pred_mix.astype(int)

# evaluate mixed
mix_cm = confusion_matrix(mix_test_df['y_true'], mix_test_df['y_pred_mix'])
mix_acc = float((mix_test_df['y_true'].values == mix_test_df['y_pred_mix'].values).mean())
mix_p, mix_r, mix_f1, _ = precision_recall_fscore_support(
    mix_test_df['y_true'].values,
    mix_test_df['y_pred_mix'].values,
    average='binary',
    zero_division=0
)

print("\n=== MIXED MODEL (BASE + BOOST IN τ-ZONE) FULL TEST ===")
print(f"Acc={mix_acc:.4f} P={mix_p:.4f} R={mix_r:.4f} F1={mix_f1:.4f}")
print("Confusion matrix:\n", mix_cm)
print(classification_report(mix_test_df['y_true'].values, mix_test_df['y_pred_mix'].values, digits=4))

plot_confusion(mix_cm, 'Mixed model: confusion matrix', 'fig_4_4_mix_confusion.png')

# compare base vs mixed errors in tau-zone
y_true_tau = mix_test_df.loc[in_tau_test, 'y_true'].astype(int).values
y_base_tau = mix_test_df.loc[in_tau_test, 'y_pred'].astype(int).values
y_mix_tau  = mix_test_df.loc[in_tau_test, 'y_pred_mix'].astype(int).values

fp0 = int(((y_true_tau == 0) & (y_base_tau == 1)).sum())
fn0 = int(((y_true_tau == 1) & (y_base_tau == 0)).sum())
fp1 = int(((y_true_tau == 0) & (y_mix_tau == 1)).sum())
fn1 = int(((y_true_tau == 1) & (y_mix_tau == 0)).sum())

used_n = int(mix_test_df.loc[in_tau_test, 'used_boost'].sum())
tau_n = int(in_tau_test.sum())

table_tau_delta = pd.DataFrame([[
    tau_n, used_n, used_n / max(1, tau_n),
    fp0, fn0, fp1, fn1,
    (fp1 - fp0), (fn1 - fn0),
    (fp1 + fn1) - (fp0 + fn0)
]], columns=[
    'tau_zone_size_test','used_boost_count','used_boost_fraction',
    'fp_base_tau','fn_base_tau','fp_mix_tau','fn_mix_tau',
    'delta_fp','delta_fn','delta_total_errors'
])
save_table(table_tau_delta, 'table_4_4_tau_zone_delta_test.csv')

# plot baseline and boost probabilities in tau-zone
plt.figure(figsize=(10,4))
plt.hist(mix_test_df.loc[in_tau_test, 'p1'], bins=30, alpha=0.7, label='baseline p1 in tau')
plt.hist(mix_test_df.loc[in_tau_test, 'p1_boost'], bins=30, alpha=0.7, label='boost p1 in tau')
plt.title('Baseline and boost probability distributions in τ-zone (test)')
plt.xlabel('Predicted probability')
plt.ylabel('Count')
plt.legend()
save_fig('fig_4_4_tau_zone_probabilities_test.png')

# top attacks in tau-zone on test
tau_attack_counts_test = mix_test_df.loc[in_tau_test & (mix_test_df['true_text'] != 'normal'), 'true_text'].value_counts().reset_index()
tau_attack_counts_test.columns = ['attack_label','count']
save_table(tau_attack_counts_test, 'table_4_4_top_attack_types_in_tau_test.csv')

plt.figure(figsize=(10,5))
top_tau_test = tau_attack_counts_test.head(20)
plt.barh(top_tau_test['attack_label'][::-1], top_tau_test['count'][::-1])
plt.title('Top attack types in τ-zone on test set')
plt.xlabel('Count')
save_fig('fig_4_4_top_attacks_in_tau_test.png')

# ------------------------------------------------------------
# 14. PER-ATTACK COMPARISON
# ------------------------------------------------------------
per_attack_base = per_attack_recall_table(y_test_text, y_test_bin, base_pred_test)
per_attack_boost = per_attack_recall_table(y_test_text, y_test_bin, boost_pred_test)
per_attack_mix = per_attack_recall_table(y_test_text, y_test_bin, mix_test_df['y_pred_mix'].values)

save_table(per_attack_base, 'table_4_4_per_attack_base.csv')
save_table(per_attack_boost, 'table_4_4_per_attack_boost.csv')
save_table(per_attack_mix, 'table_4_4_per_attack_mix.csv')

# comparison table
cmp_attack = per_attack_base[['attack_label','support','recall']].rename(columns={'recall':'recall_base'})
cmp_attack = cmp_attack.merge(
    per_attack_boost[['attack_label','recall']].rename(columns={'recall':'recall_boost'}),
    on='attack_label', how='left'
)
cmp_attack = cmp_attack.merge(
    per_attack_mix[['attack_label','recall']].rename(columns={'recall':'recall_mix'}),
    on='attack_label', how='left'
)

cmp_attack['delta_boost_minus_base'] = cmp_attack['recall_boost'] - cmp_attack['recall_base']
cmp_attack['delta_mix_minus_base'] = cmp_attack['recall_mix'] - cmp_attack['recall_base']

save_table(cmp_attack, 'table_4_4_per_attack_comparison.csv')

top_improved = cmp_attack.sort_values('delta_mix_minus_base', ascending=False).head(20)

plt.figure(figsize=(10,5))
plt.barh(top_improved['attack_label'][::-1], top_improved['delta_mix_minus_base'][::-1])
plt.title('Top attack recall improvements: mixed model vs baseline')
plt.xlabel('Δ recall')
save_fig('fig_4_4_top_recall_improvements_mix.png')

# ------------------------------------------------------------
# 15. FINAL SUMMARY TABLE
# ------------------------------------------------------------
table_final = pd.DataFrame([
    ['Baseline SPAS model', len(X_train_spas), stft_time_base, train_time_base, base_acc, base_prec, base_rec, base_f1,
     int(cm_base_test[0,0]), int(cm_base_test[0,1]), int(cm_base_test[1,0]), int(cm_base_test[1,1])],
    ['Boost model', len(X_train_boost), stft_time_boost, train_time_boost, boost_acc_full, boost_prec_full, boost_rec_full, boost_f1_full,
     int(cm_boost_test[0,0]), int(cm_boost_test[0,1]), int(cm_boost_test[1,0]), int(cm_boost_test[1,1])],
    ['Mixed model (selective override)', len(X_train_boost), stft_time_boost, train_time_boost, mix_acc, mix_p, mix_r, mix_f1,
     int(mix_cm[0,0]), int(mix_cm[0,1]), int(mix_cm[1,0]), int(mix_cm[1,1])]
], columns=[
    'model','train_samples','stft_time_sec','train_time_sec',
    'accuracy','precision','recall','f1',
    'TN','FP','FN','TP'
])

save_table(table_final, 'table_4_4_final_comparison.csv')
print(table_final)

# ------------------------------------------------------------
# 16. SHORT TEXT SUMMARY
# ------------------------------------------------------------
summary_lines = []
summary_lines.append("4.4 Summary")
summary_lines.append("")
summary_lines.append(f"Baseline SPAS model: Accuracy={base_acc:.4f}, Precision={base_prec:.4f}, Recall={base_rec:.4f}, F1={base_f1:.4f}")
summary_lines.append(f"Boost model: Accuracy={boost_acc_full:.4f}, Precision={boost_prec_full:.4f}, Recall={boost_rec_full:.4f}, F1={boost_f1_full:.4f}")
summary_lines.append(f"Mixed model: Accuracy={mix_acc:.4f}, Precision={mix_p:.4f}, Recall={mix_r:.4f}, F1={mix_f1:.4f}")
summary_lines.append("")
summary_lines.append(f"Selected τ-zone: [{TAU_L:.2f}, {TAU_U:.2f}]")
summary_lines.append(f"Selected gating thresholds: lo={GATE_THR_LO:.2f}, hi={GATE_THR_HI:.2f}")
summary_lines.append(f"τ-zone size on test: {tau_n}")
summary_lines.append(f"Rows overridden by boost in τ-zone: {used_n} ({used_n/max(1,tau_n):.4f})")

summary_path = os.path.join(OUT_DIR, 'summary_4_4.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"Saved summary: {summary_path}")
print("\n=== ГОТОВО 4.4 ===")
print("Усі результати збережені в:", OUT_DIR)