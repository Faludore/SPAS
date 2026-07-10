"""Section 4.3: baseline, SMOTENC, and SPAS balancing experiments for NSL-KDD."""

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, 'data', 'nsl-kdd', 'KDDTrain+.txt')
test_path  = os.path.join(BASE_DIR, 'data', 'nsl-kdd', 'KDDTest+.txt')

missing_dataset_files = [path for path in (train_path, test_path) if not os.path.isfile(path)]
if missing_dataset_files:
    raise FileNotFoundError(
        "NSL-KDD files are missing. Place KDDTrain+.txt and KDDTest+.txt in "
        f"{os.path.join(BASE_DIR, 'data', 'nsl-kdd')}. Missing: {missing_dataset_files}"
    )

OUT_DIR = os.path.join(BASE_DIR, 'results', 'section_4_3')
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
