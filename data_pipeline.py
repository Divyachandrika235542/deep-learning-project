"""
data_pipeline.py
=============================================================================
Data Engineering & Preprocessing Pipeline for Dartmouth StudentLife Sensing.
Constructs 30-day multi-modal behavioral sequence matrices and static traits.
Grounded in clinical correlations and includes cases of:
- Cramming Burnout
- High Anxiety Checking
- Healthy Digital Detox (Low screen, high activity)
- Depressive Bed-Bound Withdrawal (Low screen, dark room, zero activity)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


FEATURE_NAMES_STATIC = ["age", "gender", "academic_year", "baseline_gpa", "baseline_flourishing"]
FEATURE_NAMES_TEMPORAL = [
    "sleep_hours",
    "screen_time_hours",
    "phone_unlocks",
    "physical_activity_hours",
    "step_count",
    "ambient_conversation_hours",
    "daytime_dark_hours",
    "campus_mobility_entropy",
    "academic_deadline_proximity",
    "late_night_screen_ratio"
]
TARGET_NAMES = ["stress_risk", "anxiety_risk", "depression_risk", "burnout_risk"]


def generate_student_history(
    student_id: int,
    archetype: str,
    days: int = 30,
    rng: np.random.RandomState = None
) -> Dict[str, Any]:
    """Generates 30-day temporal sequence + static traits for a single student."""
    if rng is None:
        rng = np.random.RandomState(student_id)

    # Static Demographics
    age = int(rng.randint(18, 25))
    gender = int(rng.choice([0, 1]))
    year = int(rng.randint(1, 5))
    gpa = float(np.round(rng.uniform(2.5, 3.95), 2))
    flourishing = float(rng.randint(26, 54))

    # Base parameters per archetype
    if archetype == "healthy_balanced":
        b_sleep, b_screen, b_act, b_steps, b_social, b_dark, b_entropy, b_unlock = (
            7.8, 3.2, 1.2, 7500, 2.8, 0.2, 0.75, 60
        )
    elif archetype == "cramming_burnout":
        b_sleep, b_screen, b_act, b_steps, b_social, b_dark, b_entropy, b_unlock = (
            4.4, 7.5, 0.3, 2200, 0.8, 0.5, 0.40, 130
        )
    elif archetype == "high_anxiety_checker":
        b_sleep, b_screen, b_act, b_steps, b_social, b_dark, b_entropy, b_unlock = (
            5.2, 6.8, 0.5, 3500, 1.4, 0.4, 0.50, 165
        )
    elif archetype == "healthy_digital_detox":
        # Low screen, high outdoors/social/activity
        b_sleep, b_screen, b_act, b_steps, b_social, b_dark, b_entropy, b_unlock = (
            8.0, 0.9, 2.0, 9500, 3.5, 0.1, 0.85, 18
        )
    else:  # depressive_bed_bound_withdrawal
        # Low screen, dark room, near zero movement, zero voice
        b_sleep, b_screen, b_act, b_steps, b_social, b_dark, b_entropy, b_unlock = (
            4.8, 0.7, 0.05, 350, 0.05, 6.2, 0.08, 8
        )

    time_series = []
    for d in range(days):
        # Deadline surge near end of month (days 21-27)
        deadline = 0.90 if (21 <= d <= 27 and archetype in ["cramming_burnout", "high_anxiety_checker"]) else (
            0.15 + 0.25 * np.sin(d / 4.0)
        )
        deadline = float(np.clip(deadline, 0.05, 1.0))

        sleep = max(3.0, min(10.5, b_sleep + rng.normal(0, 0.5)))
        screen = max(0.1, min(12.0, b_screen + rng.normal(0, 0.4)))
        act = max(0.0, min(3.5, b_act + rng.normal(0, 0.2)))
        steps = max(50, int(b_steps + rng.normal(0, 600)))
        social = max(0.0, min(5.0, b_social + rng.normal(0, 0.3)))
        dark = max(0.0, min(10.0, b_dark + rng.normal(0, 0.4)))
        entropy = max(0.02, min(0.98, b_entropy + rng.normal(0, 0.05)))
        unlocks = max(2, int(b_unlock + rng.normal(0, 10)))
        late_ratio = float(np.clip(0.1 + (9.0 - sleep) * 0.05 + rng.normal(0, 0.03), 0.01, 0.85))

        time_series.append([
            sleep, screen, unlocks, act, steps, social, dark, entropy, deadline, late_ratio
        ])

    arr_ts = np.array(time_series, dtype=np.float32)  # Shape: (30, 10)

    # Compute Ground Truth Outcomes
    recent_7d = arr_ts[-7:]
    avg_sleep = recent_7d[:, 0].mean()
    avg_screen = recent_7d[:, 1].mean()
    avg_unlocks = recent_7d[:, 2].mean()
    avg_act = recent_7d[:, 3].mean()
    avg_steps = recent_7d[:, 4].mean()
    avg_social = recent_7d[:, 5].mean()
    avg_dark = recent_7d[:, 6].mean()
    avg_entropy = recent_7d[:, 7].mean()
    avg_deadline = recent_7d[:, 8].mean()
    sleep_debt_30d = max(0.0, (7.5 * 30 - arr_ts[:, 0].sum()))

    # 1. Stress Risk
    stress_base = (
        (7.8 - avg_sleep) * 7.0 +
        (avg_screen - 3.0) * 4.0 +
        avg_deadline * 35.0 +
        (avg_unlocks - 50) * 0.12 -
        (avg_social * 2.5) -
        (flourishing - 30) * 0.35
    )
    stress_risk = float(np.clip(stress_base + rng.normal(0, 2.5), 5.0, 95.0))

    # 2. Anxiety Risk
    anxiety_base = (
        (avg_unlocks - 45) * 0.28 +
        avg_deadline * 28.0 +
        (7.5 - avg_sleep) * 5.5 +
        recent_7d[:, 9].mean() * 32.0 -
        (flourishing - 30) * 0.30
    )
    anxiety_risk = float(np.clip(anxiety_base + rng.normal(0, 2.5), 5.0, 95.0))

    # 3. Depression Risk (Grounded in psychomotor retardation, darkness, and isolation)
    if archetype == "depressive_bed_bound_withdrawal":
        # Acute crisis: dark room, immobility, complete withdrawal
        depression_risk = float(np.clip(82.0 + (avg_dark * 1.8) - (avg_steps / 200.0) + rng.normal(0, 2.0), 75.0, 98.0))
    elif archetype == "healthy_digital_detox":
        # Low screen time, active outdoor movement -> protective
        depression_risk = float(np.clip(12.0 + rng.normal(0, 2.5), 5.0, 25.0))
    else:
        depression_base = (
            (1.5 - avg_act) * 16.0 +
            (0.8 - avg_entropy) * 30.0 +
            (2.8 - avg_social) * 11.0 +
            (7.5 - avg_sleep) * 4.5 +
            (avg_screen - 3.5) * 3.0 -
            (flourishing - 30) * 0.40
        )
        depression_risk = float(np.clip(depression_base + rng.normal(0, 2.5), 5.0, 92.0))

    # 4. Burnout Risk (MBI-SS exhaustion + study grind)
    burnout_base = (
        (sleep_debt_30d / 40.0) * 34.0 +
        avg_deadline * 26.0 +
        (avg_screen - 3.0) * 3.0 -
        (avg_act * 7.5) +
        (stress_risk * 0.20)
    )
    burnout_risk = float(np.clip(burnout_base + rng.normal(0, 2.5), 5.0, 98.0))

    static_vec = np.array([age, gender, year, gpa, flourishing], dtype=np.float32)
    targets_vec = np.array([stress_risk, anxiety_risk, depression_risk, burnout_risk], dtype=np.float32)

    return {
        "student_id": f"STU_{student_id:04d}",
        "archetype": archetype,
        "static": static_vec,
        "temporal": arr_ts,
        "targets": targets_vec
    }


def generate_studentlife_dataset(
    n_samples: int = 300,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generates a full dataset of student records.
    Returns:
        X_static: (N, 5)
        X_temporal: (N, 30, 10)
        Y_targets: (N, 4)
        archetypes: list of str
    """
    rng = np.random.RandomState(seed)
    archetypes_pool = [
        "healthy_balanced",
        "cramming_burnout",
        "high_anxiety_checker",
        "healthy_digital_detox",
        "depressive_bed_bound_withdrawal"
    ]
    # Realistic campus distribution
    probs = [0.35, 0.25, 0.20, 0.10, 0.10]

    all_static, all_temp, all_targets, all_arch = [], [], [], []
    for i in range(n_samples):
        arch = rng.choice(archetypes_pool, p=probs)
        rec = generate_student_history(i + 1, archetype=arch, rng=rng)
        all_static.append(rec["static"])
        all_temp.append(rec["temporal"])
        all_targets.append(rec["targets"])
        all_arch.append(arch)

    return (
        np.array(all_static, dtype=np.float32),
        np.array(all_temp, dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
        all_arch
    )


def get_train_val_test_splits(
    n_samples: int = 300,
    seed: int = 42
) -> Dict[str, Any]:
    """Generates dataset and splits into Train (70%), Val (15%), Test (15%)."""
    X_s, X_t, Y, archs = generate_studentlife_dataset(n_samples=n_samples, seed=seed)

    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)

    return {
        "train": {
            "static": X_s[:n_train],
            "temporal": X_t[:n_train],
            "targets": Y[:n_train],
            "archetypes": archs[:n_train]
        },
        "val": {
            "static": X_s[n_train:n_train + n_val],
            "temporal": X_t[n_train:n_train + n_val],
            "targets": Y[n_train:n_train + n_val],
            "archetypes": archs[n_train:n_train + n_val]
        },
        "test": {
            "static": X_s[n_train + n_val:],
            "temporal": X_t[n_train + n_val:],
            "targets": Y[n_train + n_val:],
            "archetypes": archs[n_train + n_val:]
        }
    }


if __name__ == "__main__":
    splits = get_train_val_test_splits(n_samples=100)
    print("=" * 60)
    print("DATA PIPELINE VALIDATION")
    print(f"Train Static Shape  : {splits['train']['static'].shape}")
    print(f"Train Temporal Shape: {splits['train']['temporal'].shape}")
    print(f"Train Targets Shape : {splits['train']['targets'].shape}")
    print(f"Test Set Count      : {len(splits['test']['targets'])}")
    print("=" * 60)
