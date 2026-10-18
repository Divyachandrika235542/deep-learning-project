"""
simulate_twin.py
=============================================================================
Terminal-Based Digital Twin Interactive Simulation Runner.
-----------------------------------------------------------------------------
Simulates an individual student's mental health twin and executes
the Inactivity Disambiguation Engine in real-time.
Outputs formatted clinical reports directly to the console.
Usage:
    python simulate_twin.py --archetype depressive_bed_bound_withdrawal
    python simulate_twin.py --archetype healthy_digital_detox
    python simulate_twin.py --archetype cramming_burnout
=============================================================================
"""

import sys
import argparse
from data_pipeline import generate_student_history
from model import MentalHealthTwinPredictor


def run_digital_twin_simulation(archetype: str = "depressive_bed_bound_withdrawal"):
    print("\n" + "=" * 75)
    print("[TWIN] MENTAL HEALTH DIGITAL TWIN: TERMINAL SIMULATION ENGINE")
    print("=" * 75)

    # 1. Generate Student Behavioral Stream
    record = generate_student_history(student_id=101, archetype=archetype)
    static = record["static"]
    temporal = record["temporal"]
    ground_truth = record["targets"]

    gender_str = "Male" if static[1] == 1 else "Female"
    print(f"\n[PROFILE]: Student ID: {record['student_id']} | Archetype: {archetype.upper()}")
    print(f"Demographics: Age {int(static[0])} | Gender: {gender_str} | Year: {int(static[2])} | GPA: {static[3]:.2f} | Baseline Flourishing: {static[4]:.0f}/56")

    # 2. Extract recent 7-day behavioral summary
    recent_7d = temporal[-7:]
    print("\n[LONGITUDINAL SENSING BIOMARKERS (Last 7 Days)]:")
    print(f"  * Daily Sleep Average       : {recent_7d[:, 0].mean():.1f} hours/day")
    print(f"  * Daily Screen Time Average : {recent_7d[:, 1].mean():.1f} hours/day")
    print(f"  * Daily Phone Unlocks       : {int(recent_7d[:, 2].mean())} unlocks/day")
    print(f"  * Physical Activity Steps   : {int(recent_7d[:, 4].mean()):,} steps/day")
    print(f"  * Daytime Darkness Exposure : {recent_7d[:, 6].mean():.1f} hours in dark room (8 AM - 8 PM)")
    print(f"  * Social Conversation Voice : {recent_7d[:, 5].mean():.1f} hours/day")
    print(f"  * Academic Deadline Surge   : {recent_7d[:, 8].mean() * 100:.0f}% intensity")

    # 3. Execute Digital Twin Inference
    predictor = MentalHealthTwinPredictor(seed=42)
    result = predictor.predict(static, temporal, enable_novelty=True)
    preds = result["predictions"]
    novelty = result["novelty_analysis"]

    print("\n" + "-" * 75)
    print("[PREDICTIONS] DIGITAL TWIN MULTI-TASK RISK PREDICTIONS")
    print("-" * 75)
    print(f"  1. Stress Risk          : {preds['stress_risk']:>5.1f}%  [{get_risk_tag(preds['stress_risk'])}]")
    print(f"  2. Anxiety Risk         : {preds['anxiety_risk']:>5.1f}%  [{get_risk_tag(preds['anxiety_risk'])}]")
    print(f"  3. Depression Risk      : {preds['depression_risk']:>5.1f}%  [{get_risk_tag(preds['depression_risk'])}]")
    print(f"  4. Academic Burnout     : {preds['burnout_risk']:>5.1f}%  [{get_risk_tag(preds['burnout_risk'])}]")

    print("\n" + "-" * 75)
    print("[NOVELTY ENGINE] INACTIVITY DISAMBIGUATION REPORT")
    print("-" * 75)
    print(f"  * Detected Inactivity State : {novelty['state']}")
    print(f"  * Compound Withdrawal Index : {novelty['withdrawal_index']} / 1.00")
    print(f"  * Darkness Sub-Score        : {novelty['sub_scores']['daytime_darkness']}")
    print(f"  * Physical Immobility       : {novelty['sub_scores']['immobility']}")
    print(f"  * Social Voice Silence      : {novelty['sub_scores']['social_silence']}")
    print(f"  * Clinical Rationale        : {novelty['clinical_rationale']}")

    # 4. Compare Traditional Model vs Our Twin
    print("\n" + "-" * 75)
    print("[COMPARISON] Traditional ML vs Our Digital Twin")
    print("-" * 75)
    trad_dep = result["traditional_depression_prediction"]
    our_dep = preds["depression_risk"]
    print(f"  * Traditional ML Prediction (Screen Time only) : {trad_dep:.1f}% Depression")
    print(f"  * Our Digital Twin Prediction (Context-Aware)  : {our_dep:.1f}% Depression")
    if result["false_negative_prevented"]:
        print("  [ALERT] TRADITIONAL MODEL PRODUCED A CRITICAL FALSE NEGATIVE!")
        print("     Because screen time was near-zero, traditional ML assumed the student was fine.")
        print(f"     Our Digital Twin detected dark-room bed confinement and corrected the alert to {our_dep:.1f}%.")
    else:
        print("  [OK] Concordant Analysis: Context sensors confirm normal/healthy lifestyle dynamics.")

    print("\n" + "=" * 75)
    print("[RECOMMENDATIONS] PERSONALIZED LIFESTYLE ACTION RECOMMENDATIONS")
    print("=" * 75)
    if archetype == "depressive_bed_bound_withdrawal":
        print("  1. Urgent: Immediate in-person wellness check-in recommended by campus health advisor.")
        print("  2. Environmental: Open room blinds to increase natural circadian lux exposure.")
        print("  3. Behavioral: Gentle 10-minute daylight walk to counter psychomotor retardation.")
    elif archetype == "healthy_digital_detox":
        print("  1. Maintain current outdoor activities and healthy screen boundaries.")
        print("  2. Continue consistent sleep-wake circadian rhythm.")
    else:
        print("  1. Shift 1.5 hours from late-night screen browsing into restorative sleep.")
        print("  2. Break assignment cramming into 45-minute focused blocks with 10-minute screen breaks.")
    print("=" * 75 + "\n")


def get_risk_tag(score: float) -> str:
    if score >= 75.0:
        return "CRITICAL RISK"
    elif score >= 50.0:
        return "ELEVATED RISK"
    elif score >= 30.0:
        return "MODERATE RISK"
    return "LOW / HEALTHY"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Mental Health Digital Twin")
    parser.add_argument(
        "--archetype",
        type=str,
        default="depressive_bed_bound_withdrawal",
        choices=[
            "depressive_bed_bound_withdrawal",
            "healthy_digital_detox",
            "cramming_burnout",
            "high_anxiety_checker",
            "healthy_balanced"
        ],
        help="Select student behavioral profile to simulate"
    )
    args = parser.parse_args()
    run_digital_twin_simulation(args.archetype)
