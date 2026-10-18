# 🧠 Mental Health Twin: Deep Learning-Based Multi-Task Digital Twin for Student Mental Well-Being

A pure **Machine Learning / Deep Learning Research Pipeline** (Python, PyTorch, Scikit-learn, Pandas, Matplotlib) predicting **Stress, Anxiety, Depression, and Academic Burnout** from longitudinal smartphone sensing streams (Dartmouth StudentLife Dataset).

---

## 🌟 The Core Novelty: Inactivity Disambiguation Engine
*(Distinguishing **Healthy Digital Detox** from **Depressive Bed-Bound Withdrawal**)*

### The Blind Spot in All Prior ML Literature:
Conventional models trained on the StudentLife dataset learn a naive linear assumption:
$$\text{High Screen Time} \implies \text{High Stress} \quad \text{and} \quad \text{Low Screen Time} \implies \text{Healthy / Calm Student}$$

**The Fatal Failure Mode:** When a student enters severe depression or acute burnout, they often shut down—spending $16+$ hours in bed in a dark room without touching their phone. Traditional models observe **0 screen time** and falsely predict **0% depression**, creating dangerous clinical false negatives.

### Our Solution:
Our **Inactivity Disambiguation Engine** (`inactivity_engine.py`) cross-references periods of smartphone silence with passive ambient sensors:
* **Healthy Digital Detox:** Screen is off, BUT ambient light shows daylight, accelerometer shows walking movement ($>5,000$ steps), and location indicates campus mobility. $\rightarrow$ **Low Risk (Detox)**.
* **Depressive Bed-Bound Withdrawal:** Screen is off, BUT room is pitch-dark at 2:00 PM ($0\text{ lux}$), accelerometer is motionless on a mattress, and audio shows zero human speech. $\rightarrow$ **Critical Depressive Alert ($85\%+$)**.

---

## 📂 Project Architecture (Pure Python ML Pipeline)

```
mental_health_twin/
│
├── data_pipeline.py          # Generates & cleans StudentLife 30-day time series (30 x 10) & static traits
├── inactivity_engine.py      # CORE NOVELTY: Disambiguates Digital Detox vs. Depressive Withdrawal
├── model.py                  # Hybrid Multi-Branch Multi-Task DNN + BiLSTM Architecture
├── train.py                  # Multi-task training loop & loss convergence plotting
├── evaluate.py               # Computes RMSE, MAE, R², Pearson r & benchmarks the Novelty
├── explainability.py         # Generates publication-ready SHAP-style attribution plots (.png)
├── simulate_twin.py          # Terminal-based Digital Twin console runner
├── run_pipeline.py           # Master script executing all steps in order
│
├── models/                   # Directory where trained weights (.npz / .pth) are saved
├── results/                  # Generated evaluation metrics (CSV) and publication plots (PNG)
│   ├── training_loss_curve.png
│   ├── feature_importance_waterfall.png
│   ├── detox_vs_withdrawal_comparison.png
│   └── evaluation_metrics.csv
└── requirements.txt          # Pure ML dependencies (numpy, pandas, scikit-learn, matplotlib)
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

### 2. Run the Entire Pipeline in One Command
```bash
python run_pipeline.py
```

### 3. Or Run Individual Modules:

* **Train the Model & Save Checkpoints:**
  ```bash
  python train.py
  ```
* **Evaluate Accuracy & Benchmark Novelty:**
  ```bash
  python evaluate.py
  ```
* **Generate Publication Figures:**
  ```bash
  python explainability.py
  ```
* **Simulate Individual Student Twins in Terminal:**
  ```bash
  # Test the Depressive Bed-Bound Withdrawal case:
  python simulate_twin.py --archetype depressive_bed_bound_withdrawal

  # Test the Healthy Digital Detox case:
  python simulate_twin.py --archetype healthy_digital_detox

  # Test the Cramming Burnout case:
  python simulate_twin.py --archetype cramming_burnout
  ```

---

## 📊 Sample Terminal Output from `simulate_twin.py`

```text
===========================================================================
🧠 MENTAL HEALTH DIGITAL TWIN: TERMINAL SIMULATION ENGINE
===========================================================================

[PROFILE]: Student ID: STU_0101 | Archetype: DEPRESSIVE_BED_BOUND_WITHDRAWAL
Demographics: Age 21 | Gender: Female | Year: 3 | GPA: 3.42 | Baseline Flourishing: 32/56

[LONGITUDINAL SENSING BIOMARKERS (Last 7 Days)]:
  • Daily Sleep Average       : 4.8 hours/day
  • Daily Screen Time Average : 0.7 hours/day (Phone Idle)
  • Daily Phone Unlocks       : 8 unlocks/day
  • Physical Activity Steps   : 350 steps/day (Immobile)
  • Daytime Darkness Exposure : 6.2 hours in dark room (8 AM - 8 PM)
  • Social Conversation Voice : 0.1 hours/day
  • Academic Deadline Surge   : 65% intensity

---------------------------------------------------------------------------
🔮 DIGITAL TWIN MULTI-TASK RISK PREDICTIONS
---------------------------------------------------------------------------
  1. Stress Risk          :  62.4%  [ELEVATED RISK]
  2. Anxiety Risk         :  54.1%  [ELEVATED RISK]
  3. Depression Risk      :  87.2%  [CRITICAL RISK]
  4. Academic Burnout     :  79.8%  [CRITICAL RISK]

---------------------------------------------------------------------------
🔍 CORE NOVELTY: INACTIVITY DISAMBIGUATION ENGINE REPORT
---------------------------------------------------------------------------
  • Detected Inactivity State : DEPRESSIVE_BED_BOUND_WITHDRAWAL
  • Compound Withdrawal Index : 0.825 / 1.00
  • Darkness Sub-Score        : 1.00
  • Physical Immobility       : 0.88
  • Social Voice Silence      : 0.97
  • Clinical Rationale        : CRITICAL: Phone idle (0.7h screen, 8 unlocks),
                                but ambient sensors reveal 6.2h daytime darkness,
                                near-zero movement (350 steps), and zero voice audio.

---------------------------------------------------------------------------
⚡ SCIENTIFIC COMPARISON: Traditional ML vs Our Digital Twin
---------------------------------------------------------------------------
  • Traditional ML Prediction (Screen Time only) : 16.4% Depression
  • Our Digital Twin Prediction (Context-Aware)  : 87.2% Depression
  ⚠️ ALERT: TRADITIONAL MODEL PRODUCED A CRITICAL FALSE NEGATIVE!
     Because screen time was near-zero, traditional ML assumed the student was fine.
     Our Digital Twin detected dark-room bed confinement and corrected the alert to 87.2%.
===========================================================================
```
