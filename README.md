# 🏟️ KASC Crowd Management System
**King Abdullah Sports City — Predictive Crowd Management**

> BSc Data Science Graduation Project | Umm Al-Qura University | 2025–2026  
> Project ID: UQU-DS-2025-M07

---

## 📌 Overview

A production-grade ML system for predictive crowd management at King Abdullah Sports City Stadium in Jeddah, Saudi Arabia. The system combines **data augmentation**, **machine learning**, and **graph-based routing** to manage crowd flow, predict congestion, and guide fans to their seats — even when real-time sensor data is incomplete.

---

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| Congestion Classifier Accuracy | **97.82%** |
| Congestion Classifier Macro-F1 | **97.03%** |
| Processing Time WMAPE | **10.09%** |
| Routing Pipeline Accuracy | **97.62%** |
| Inference Latency | **2.49 ms** |

---

## 🤖 ML Models

### Model 1 — Congestion Classifier
- **Algorithm**: Gradient Boosting Classifier
- **Target**: `Congestion_Level` (High / Medium / Low)
- **Features**: 30 engineered features
- **Accuracy**: 97.82% | **Macro-F1**: 97.03%

### Model 2 — Processing Time Regressor
- **Algorithm**: Gradient Boosting Regressor (with log1p transform)
- **Target**: `Processing_Time_s`
- **MAE**: 1.16 seconds | **WMAPE**: 10.09% | **R²**: 0.571

---

## 🗺️ Routing System

Implements **Dijkstra's algorithm** on a 28-node directed stadium graph:

```
Outer Gate → Security Checkpoint → Concourse → Zone → Section
```

- 6 Outer Gates (G1–G6)
- 6 Checkpoints (CP1–CP6)
- 4 Concourses (North, East, South, West)
- 6 Zones (Z1–Z6)
- 9 Sections (124, 123, 116, 118, 309, 313, 315, 503, 541)

Edge weights are dynamically updated using predicted congestion probabilities and observed gate scan delays.

---

## 📁 Project Structure

```
KASC-Crowd-Management/
├── data/
│   ├── stadium_gate_flow.csv        ← Training data (5,040 rows × 26 cols)
│   ├── tickets_KASC_part1.csv       ← Ticket data part 1
│   ├── tickets_KASC_part2.csv       ← Ticket data part 2
│   └── tickets_KASC_part3.csv       ← Ticket data part 3
├── plots/
│   ├── eda_plots.png                ← EDA visualizations
│   └── model_evaluation.png         ← Model evaluation charts
├── models/
│   ├── classifier.pkl               ← Trained congestion classifier
│   ├── regressor.pkl                ← Trained processing time regressor
│   ├── scaler.pkl                   ← Feature scaler
│   └── label_encoder.pkl            ← Congestion label encoder
├── dashboard/
│   └── dashboard_dijkstra.html      ← FINAL dashboard (5 tabs + Dijkstra) ⭐
├── train_complete.py                ← Full training script
├── KASC_Google_Colab.ipynb          ← Google Colab notebook
├── requirements.txt
└── README.md
```



---

## 🚀 How to Run

### Option 1 — Local Python

```bash
# 1. Clone the repo
git clone https://github.com/BandarAlsarwani/KASC-Crowd-Management.git
cd KASC-Crowd-Management

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models
python train_complete.py

# 4. Open the dashboard
open dashboard/dashboard_dijkstra.html
```

### Option 2 — Google Colab
1. Upload `KASC_Google_Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Upload `data/stadium_gate_flow.csv`
3. Click **Runtime → Run all**
4. Download the trained `.pkl` model files

### Option 3 — Dashboard Only (No Training)
1. Download `dashboard/dashboard_dijkstra.html`
2. Double-click to open in any browser
3. Navigate the 5 tabs
4. Test with sample ticket IDs: `TKT001`, `TKT002`, `TKT003`

---

## 🧪 Sample Tickets for Testing

| Ticket ID | Gate | Section | Congestion |
|-----------|------|---------|------------|
| TKT001 | 1 | 124 | 🔴 High |
| TKT002 | 2 | 118 | 🟡 Medium |
| TKT003 | 3 | 313 | 🟢 Low |
| TKT004 | 4 | 309 | 🔴 High |
| TKT005 | 5 | 503 | 🟡 Medium |
| TKT006 | 6 | 541 | 🟢 Low |

---

## 🔧 Feature Engineering (30 Features)

| Category | Features |
|----------|----------|
| Temporal (6) | Hour, Minute, Time_Minutes, Is_Peak_Hour, Hour_Sin, Hour_Cos |
| Spatial (3) | Outer_Gate, Gate_Capacity, Gate_Utilization |
| Operational (6) | Open_Lanes, Security_Staff_Count, Staff_Per_Lane, Incident_Flag, Delay_Minutes, Incident_Severity |
| Environmental (4) | Temperature_C, Humidity_%, Heat_Index, Discomfort_Score |
| Attendance (3) | Total_Attendance, Expected_Attendance, Attendance_Ratio |
| Flow (3) | Entry_Rate, Exit_Rate, Queue_Length_Est |
| Encoded (4) | Match_Type_Enc, Zone_Enc, Fan_Side_Enc, Security_Tier_Enc |
| Match (1) | Match_Importance |

---

## 👥 Team

| Name | Student ID |
|------|------------|
| Bandar Abdullah Alsarwani | 444006001 |
| Hamzah Khaled Sherbini | 44411249 |
| Anas Mohammed Alahmdi | 444002854 |
| Ahmed Nawaf Almufrraji | 44411780 |

**Supervisor**: Dr. Mohammed Halawani — mkhalawani@uqu.edu.sa  
**Department**: Data Science, College of Computing, Umm Al-Qura University

---

## 📄 License

This project is the intellectual property of Umm Al-Qura University and the respective supervisor, as declared in the Intellectual Property Right Declaration.
