"""
╔══════════════════════════════════════════════════════════════════╗
║       KASC Crowd Management System — Complete Training Script    ║
║       King Abdullah Sports City | UQU-DS-2025-M07               ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    python train_complete.py

WHAT IT DOES:
    1. Loads stadium gate flow data
    2. Engineers 30 features
    3. Trains Congestion Classifier  → saves classifier.pkl
    4. Trains Processing Time Regressor → saves regressor.pkl
    5. Evaluates both models with charts
    6. Runs Dijkstra routing on all 9 sample tickets

REQUIREMENTS:
    pip install -r requirements.txt
"""

import os
import sys
import pickle
import warnings
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, r2_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

DATA_FILE   = 'data/stadium_gate_flow.csv'
MODELS_DIR  = 'models'
PLOTS_DIR   = 'plots'

FEATURES = [
    'Hour', 'Minute', 'Time_Minutes', 'Is_Peak_Hour', 'Hour_Sin', 'Hour_Cos',
    'Outer_Gate', 'Gate_Capacity', 'Gate_Utilization',
    'Open_Lanes', 'Security_Staff_Count', 'Staff_Per_Lane',
    'Incident_Flag', 'Delay_Minutes', 'Incident_Severity',
    'Temperature_C', 'Humidity_%', 'Heat_Index', 'Discomfort_Score',
    'Total_Attendance', 'Expected_Attendance', 'Attendance_Ratio',
    'Entry_Rate', 'Exit_Rate', 'Queue_Length_Est',
    'Match_Type_Enc', 'Zone_Enc', 'Fan_Side_Enc', 'Security_Tier_Enc',
    'Match_Importance'
]

SAMPLE_TICKETS = {
    'TKT001': {'gate': 1, 'section': 'S124', 'congestion': 'High'},
    'TKT002': {'gate': 2, 'section': 'S118', 'congestion': 'Medium'},
    'TKT003': {'gate': 3, 'section': 'S313', 'congestion': 'Low'},
    'TKT004': {'gate': 4, 'section': 'S309', 'congestion': 'High'},
    'TKT005': {'gate': 5, 'section': 'S503', 'congestion': 'Medium'},
    'TKT006': {'gate': 6, 'section': 'S541', 'congestion': 'Low'},
    'TKT123': {'gate': 1, 'section': 'S123', 'congestion': 'High'},
    'TKT456': {'gate': 3, 'section': 'S315', 'congestion': 'Low'},
    'TKT789': {'gate': 2, 'section': 'S116', 'congestion': 'Medium'},
}

STADIUM_GRAPH = {
    'G1':  {'edges': {'CP1': 2}},
    'G2':  {'edges': {'CP2': 2}},
    'G3':  {'edges': {'CP3': 2}},
    'G4':  {'edges': {'CP4': 2}},
    'G5':  {'edges': {'CP5': 2}},
    'G6':  {'edges': {'CP6': 2}},
    'CP1': {'edges': {'C_N': 1}},
    'CP2': {'edges': {'C_N': 1, 'C_E': 1}},
    'CP3': {'edges': {'C_E': 1}},
    'CP4': {'edges': {'C_S': 1}},
    'CP5': {'edges': {'C_S': 1, 'C_W': 1}},
    'CP6': {'edges': {'C_W': 1}},
    'C_N': {'edges': {'Z1': 1, 'Z2': 2}},
    'C_E': {'edges': {'Z2': 1, 'Z3': 2}},
    'C_S': {'edges': {'Z3': 1, 'Z4': 2}},
    'C_W': {'edges': {'Z4': 1, 'Z5': 2, 'Z6': 2}},
    'Z1':  {'edges': {'S124': 1, 'S123': 1}},
    'Z2':  {'edges': {'S116': 1, 'S118': 1}},
    'Z3':  {'edges': {'S309': 1, 'S313': 1}},
    'Z4':  {'edges': {'S315': 1}},
    'Z5':  {'edges': {'S503': 1}},
    'Z6':  {'edges': {'S541': 1}},
    'S124': {'edges': {}}, 'S123': {'edges': {}},
    'S116': {'edges': {}}, 'S118': {'edges': {}},
    'S309': {'edges': {}}, 'S313': {'edges': {}},
    'S315': {'edges': {}}, 'S503': {'edges': {}},
    'S541': {'edges': {}},
}


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def banner(title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


def make_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_data():
    banner('STEP 1 — Load Data')

    if not os.path.exists(DATA_FILE):
        print(f'\n❌  File not found: {DATA_FILE}')
        print('    Make sure your CSV is at:  data/stadium_gate_flow_KASC_master_batch2_v3.csv')
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f'✅  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns')
    print(f'\n📊  Congestion_Level distribution:')
    print(df['Congestion_Level'].value_counts().to_string())
    print(f'\n⏱️   Processing_Time_s stats:')
    print(df['Processing_Time_s'].describe().round(2).to_string())
    return df


# ─────────────────────────────────────────────────────────────────
#  STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def engineer_features(df):
    banner('STEP 2 — Feature Engineering (30 Features)')
    df = df.copy()

    # Temporal (6)
    df['Hour']         = df['Time_Bin'].str.split(':').str[0].astype(int)
    df['Minute']       = df['Time_Bin'].str.split(':').str[1].astype(int)
    df['Time_Minutes'] = df['Hour'] * 60 + df['Minute']
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)
    df['Hour_Sin']     = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos']     = np.cos(2 * np.pi * df['Hour'] / 24)

    # Spatial (3)
    gate_cap = {1: 500, 2: 450, 3: 480, 4: 520, 5: 460, 6: 490}
    df['Gate_Capacity']    = df['Outer_Gate'].map(gate_cap).fillna(480)
    df['Gate_Utilization'] = df['Total_Attendance'] / (df['Gate_Capacity'] * df['Open_Lanes'] + 1)

    # Operational (6)
    df['Staff_Per_Lane']   = df['Security_Staff_Count'] / (df['Open_Lanes'] + 1)
    df['Incident_Severity'] = df['Incident_Flag'] * df['Delay_Minutes']

    # Environmental (4)
    df['Discomfort_Score'] = (df['Temperature_C'] - 20) * 0.5 + (df['Humidity_%'] - 50) * 0.3

    # Attendance (3)
    df['Attendance_Ratio'] = df['Total_Attendance'] / (df['Expected_Attendance'] + 1)

    # Encode categoricals (4)
    le = LabelEncoder()
    df['Match_Type_Enc']    = le.fit_transform(df['Match_Type'].astype(str))
    df['Zone_Enc']          = le.fit_transform(df['Zone'].astype(str))
    df['Fan_Side_Enc']      = le.fit_transform(df['Fan_Side'].astype(str))
    df['Security_Tier_Enc'] = le.fit_transform(df['Security_Tier'].astype(str))

    print(f'✅  Feature engineering complete — {len(FEATURES)} features ready')
    return df


# ─────────────────────────────────────────────────────────────────
#  STEP 3 — TRAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────

def train_classifier(df):
    banner('STEP 3 — Train Congestion Classifier')

    X = df[FEATURES].fillna(0)
    le_cong = LabelEncoder()
    y = le_cong.fit_transform(df['Congestion_Level'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_w = np.array([weights[c] for c in y_train])

    print('   Training GradientBoostingClassifier (this may take ~1 min)...')
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.08, subsample=0.8, random_state=42
    )
    clf.fit(X_train_s, y_train, sample_weight=sample_w)

    y_pred   = clf.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f'\n✅  Classifier Results:')
    print(f'    Accuracy  : {accuracy*100:.2f}%')
    print(f'    Macro-F1  : {macro_f1*100:.2f}%')
    print(f'\n    Per-Class Report:')
    print(classification_report(y_test, y_pred, target_names=le_cong.classes_))

    return clf, scaler, le_cong, X_test_s, y_test, y_pred, X_train


# ─────────────────────────────────────────────────────────────────
#  STEP 4 — TRAIN REGRESSOR
# ─────────────────────────────────────────────────────────────────

def train_regressor(df, scaler):
    banner('STEP 4 — Train Processing Time Regressor')

    X = df[FEATURES].fillna(0)
    y = df['Processing_Time_s']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print('   Training GradientBoostingRegressor (this may take ~1 min)...')
    reg = GradientBoostingRegressor(
        n_estimators=300, max_depth=5,
        learning_rate=0.08, subsample=0.8, random_state=42
    )
    reg.fit(X_train_s, np.log1p(y_train))

    y_pred = np.expm1(reg.predict(X_test_s))
    y_pred = np.maximum(y_pred, 0)

    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mask = y_test > 0
    wmape = np.sum(np.abs(y_test[mask] - y_pred[mask])) / np.sum(y_test[mask]) * 100

    print(f'\n✅  Regressor Results:')
    print(f'    MAE   : {mae:.2f} seconds')
    print(f'    WMAPE : {wmape:.2f}%')
    print(f'    R²    : {r2:.3f}')

    return reg, X_test_s, y_test, y_pred


# ─────────────────────────────────────────────────────────────────
#  STEP 5 — SAVE MODELS
# ─────────────────────────────────────────────────────────────────

def save_models(clf, reg, scaler, le_cong):
    banner('STEP 5 — Save Models')

    pickle.dump(clf,     open(f'{MODELS_DIR}/classifier.pkl',    'wb'))
    pickle.dump(reg,     open(f'{MODELS_DIR}/regressor.pkl',     'wb'))
    pickle.dump(scaler,  open(f'{MODELS_DIR}/scaler.pkl',        'wb'))
    pickle.dump(le_cong, open(f'{MODELS_DIR}/label_encoder.pkl', 'wb'))

    print(f'✅  Models saved to /{MODELS_DIR}/:')
    print(f'    classifier.pkl')
    print(f'    regressor.pkl')
    print(f'    scaler.pkl')
    print(f'    label_encoder.pkl')


# ─────────────────────────────────────────────────────────────────
#  STEP 6 — EVALUATION CHARTS
# ─────────────────────────────────────────────────────────────────

def plot_evaluation(clf, le_cong, X_test_clf, y_test_clf, y_pred_clf,
                    y_test_reg, y_pred_reg, X_train):
    banner('STEP 6 — Generate Evaluation Charts')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('KASC Model Evaluation', fontsize=14, fontweight='bold')

    # Confusion Matrix
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=le_cong.classes_, yticklabels=le_cong.classes_)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    axes[0].set_title(f'Confusion Matrix  (Accuracy: {acc*100:.2f}%)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Feature Importance
    importances = pd.Series(clf.feature_importances_, index=FEATURES)
    importances.sort_values().tail(15).plot(kind='barh', ax=axes[1], color='#3498db')
    axes[1].set_title('Top 15 Feature Importances')
    axes[1].set_xlabel('Importance Score')

    # Predicted vs Actual
    axes[2].scatter(y_test_reg, y_pred_reg, alpha=0.3, s=10, color='#9b59b6')
    mv = max(y_test_reg.max(), y_pred_reg.max())
    axes[2].plot([0, mv], [0, mv], 'r--', lw=2, label='Perfect Fit')
    mask = y_test_reg > 0
    wmape = np.sum(np.abs(y_test_reg[mask] - y_pred_reg[mask])) / np.sum(y_test_reg[mask]) * 100
    r2    = r2_score(y_test_reg, y_pred_reg)
    axes[2].set_title(f'Regressor: Actual vs Predicted\nWMAPE: {wmape:.2f}%  R²: {r2:.3f}')
    axes[2].set_xlabel('Actual (s)')
    axes[2].set_ylabel('Predicted (s)')
    axes[2].legend()

    plt.tight_layout()
    out = f'{PLOTS_DIR}/model_evaluation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅  Chart saved → {out}')


# ─────────────────────────────────────────────────────────────────
#  STEP 7 — DIJKSTRA ROUTING
# ─────────────────────────────────────────────────────────────────

def dijkstra(graph, start, end):
    distances = {n: float('inf') for n in graph}
    distances[start] = 0
    previous = {n: None for n in graph}
    visited  = []
    heap     = [(0, start)]

    while heap:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.append(node)
        if node == end:
            break
        for neighbor, weight in graph[node]['edges'].items():
            new_cost = cost + weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                previous[neighbor]  = node
                heapq.heappush(heap, (new_cost, neighbor))

    path, cur = [], end
    while cur:
        path.insert(0, cur)
        cur = previous[cur]

    return {
        'path': path,
        'cost': distances[end],
        'nodes_visited': len(visited),
        'total_nodes':   len(graph),
    }


def run_routing():
    banner('STEP 7 — Dijkstra Routing System')

    emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    print(f'\n  {"Ticket":<8}  {"Congestion":<8}  {"Path":<45}  {"Cost":>4}  Nodes')
    print(f'  {"─"*8}  {"─"*8}  {"─"*45}  {"─"*4}  {"─"*5}')

    for tid, info in SAMPLE_TICKETS.items():
        start  = f"G{info['gate']}"
        end    = info['section']
        result = dijkstra(STADIUM_GRAPH, start, end)
        path   = ' → '.join(result['path'])
        e      = emoji.get(info['congestion'], '⚪')
        print(f'  {tid:<8}  {e} {info["congestion"]:<6}  {path:<45}  '
              f'{result["cost"]:>4}  {result["nodes_visited"]}/{result["total_nodes"]}')

    print(f'\n✅  All {len(SAMPLE_TICKETS)} tickets routed successfully')


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print('\n' + '═'*60)
    print('  🏟️  KASC Crowd Management System')
    print('  King Abdullah Sports City | UQU-DS-2025-M07')
    print('═'*60)

    make_dirs()

    df               = load_data()
    df               = engineer_features(df)
    clf, scaler, le_cong, X_test_clf, y_test_clf, y_pred_clf, X_train = train_classifier(df)
    reg, X_test_reg, y_test_reg, y_pred_reg                            = train_regressor(df, scaler)

    save_models(clf, reg, scaler, le_cong)
    plot_evaluation(clf, le_cong, X_test_clf, y_test_clf, y_pred_clf,
                    y_test_reg, y_pred_reg, X_train)
    run_routing()

    print('\n' + '═'*60)
    print('  ✅  TRAINING COMPLETE')
    print(f'  📁  Models  → /{MODELS_DIR}/')
    print(f'  📊  Charts  → /{PLOTS_DIR}/')
    print('═'*60 + '\n')


if __name__ == '__main__':
    main()
