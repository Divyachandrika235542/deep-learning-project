"""
inactivity_engine.py
=============================================================================
CORE NOVELTY: Inactivity Disambiguation Engine
Distinguishes between "Healthy Digital Detox" and "Depressive Bed-Bound Withdrawal".
-----------------------------------------------------------------------------
Solves the "Inactivity Paradox" in StudentLife ML literature:
Traditional models assume: Low Screen Time = Low Stress / Happy Student.
In reality: Severely depressed students often stay in bed in the dark for 16h,
leaving their phone untouched. Traditional models produce false negatives.
This engine cross-references phone silence with ambient light, accelerometer,
and audio sensors to decode the true psychological context of inactivity.
=============================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple


class InactivityDisambiguationEngine:
    """
    Decodes periods of zero/low smartphone interaction by fusing
    passive environmental context sensors.
    """
    def __init__(
        self,
        idle_threshold_hours: float = 4.0,
        dark_lux_threshold: float = 10.0,
        immobile_step_threshold: int = 50
    ):
        self.idle_threshold_hours = idle_threshold_hours
        self.dark_lux_threshold = dark_lux_threshold
        self.immobile_step_threshold = immobile_step_threshold

    def evaluate_inactivity_context(
        self,
        screen_time_hours: float,
        phone_unlocks: int,
        daytime_dark_hours: float,
        step_count: int,
        ambient_conversation_hours: float,
        campus_mobility_entropy: float
    ) -> Dict[str, Any]:
        """
        Analyzes a single day's sensory stream to determine the nature of inactivity.

        Returns:
            - state: 'ACTIVE_USAGE', 'HEALTHY_DIGITAL_DETOX', or 'DEPRESSIVE_BED_BOUND_WITHDRAWAL'
            - withdrawal_index: float in [0.0, 1.0]
            - risk_adjustment_factor: float multiplier to correct depression risk
            - clinical_rationale: string explaining the diagnosis
        """
        # If screen usage is normal or high, student is actively engaging with phone
        if screen_time_hours > self.idle_threshold_hours or phone_unlocks > 40:
            return {
                "state": "ACTIVE_USAGE",
                "withdrawal_index": 0.05,
                "risk_adjustment_factor": 1.0,
                "clinical_rationale": "Normal active device engagement. Standard multi-task behavioral analysis applies."
            }

        # --- Phone is Idle / Low Usage: Enter Disambiguation ---
        # 1. Darkness component: Hours spent in dark room during daylight hours (8 AM - 8 PM)
        darkness_score = min(1.0, daytime_dark_hours / 6.0)

        # 2. Immobility component: Lack of physical steps (bed-confinement)
        immobility_score = max(0.0, 1.0 - (step_count / 3000.0))

        # 3. Social silence component: Zero human conversation audio detected
        social_silence_score = max(0.0, 1.0 - (ambient_conversation_hours / 1.5))

        # 4. Campus confinement: Zero mobility across campus buildings (low location entropy)
        confinement_score = max(0.0, 1.0 - (campus_mobility_entropy / 0.6))

        # Compute compound Withdrawal Index (W_t)
        # Weights grounded in clinical DSM-5 criteria: psychomotor retardation + isolation
        w_t = (
            0.35 * darkness_score +
            0.30 * immobility_score +
            0.20 * social_silence_score +
            0.15 * confinement_score
        )
        w_t = float(np.clip(w_t, 0.0, 1.0))

        # Classification boundary
        if w_t >= 0.60:
            state = "DEPRESSIVE_BED_BOUND_WITHDRAWAL"
            # Traditional models would output low depression due to low screen time.
            # We apply a positive risk correction factor to reflect the acute crisis.
            risk_adjustment_factor = 1.0 + (w_t * 1.5)  # Can scale depression risk up by up to +150%
            rationale = (
                f"CRITICAL: Phone idle ({screen_time_hours:.1f}h screen, {phone_unlocks} unlocks), "
                f"but ambient sensors reveal {daytime_dark_hours:.1f}h daytime darkness, "
                f"near-zero movement ({step_count} steps), and no social voice audio. "
                f"Indicates severe bed-bound depressive isolation / psychomotor retardation."
            )
        elif w_t <= 0.28:
            state = "HEALTHY_DIGITAL_DETOX"
            risk_adjustment_factor = 0.70  # Protective factor: lowers depression/stress risk
            rationale = (
                f"PROTECTIVE: Low phone usage ({screen_time_hours:.1f}h) accompanied by daylight, "
                f"active physical movement ({step_count} steps), and campus mobility. "
                f"Indicates healthy offline engagement and digital detox."
            )
        else:
            state = "MILD_INACTIVITY"
            risk_adjustment_factor = 1.0
            rationale = "Moderate inactivity with mixed environmental cues. Baseline risk maintained."

        return {
            "state": state,
            "withdrawal_index": round(w_t, 3),
            "risk_adjustment_factor": round(risk_adjustment_factor, 2),
            "clinical_rationale": rationale,
            "sub_scores": {
                "daytime_darkness": round(darkness_score, 2),
                "immobility": round(immobility_score, 2),
                "social_silence": round(social_silence_score, 2),
                "dorm_confinement": round(confinement_score, 2)
            }
        }


# Quick self-test demonstration
if __name__ == "__main__":
    engine = InactivityDisambiguationEngine()
    print("=" * 70)
    print("TESTING INACTIVITY DISAMBIGUATION ENGINE (NOVELTY MODULE)")
    print("=" * 70)

    # Student 1: Healthy Digital Detox (Studying / Outdoors without phone)
    res_detox = engine.evaluate_inactivity_context(
        screen_time_hours=0.8,
        phone_unlocks=12,
        daytime_dark_hours=0.2,
        step_count=6200,
        ambient_conversation_hours=2.5,
        campus_mobility_entropy=0.75
    )
    print("\nCase 1: Student on Outdoor / Study Digital Detox")
    print(f"  State: {res_detox['state']}")
    print(f"  Withdrawal Index: {res_detox['withdrawal_index']}")
    print(f"  Risk Adjustment: {res_detox['risk_adjustment_factor']}x")
    print(f"  Rationale: {res_detox['clinical_rationale']}")

    # Student 2: Depressive Bed-Bound Withdrawal (Staying in dark dorm bed all day)
    res_withdraw = engine.evaluate_inactivity_context(
        screen_time_hours=0.4,
        phone_unlocks=5,
        daytime_dark_hours=6.5,
        step_count=80,
        ambient_conversation_hours=0.0,
        campus_mobility_entropy=0.05
    )
    print("\nCase 2: Student in Acute Depressive Bed-Bound Isolation")
    print(f"  State: {res_withdraw['state']}")
    print(f"  Withdrawal Index: {res_withdraw['withdrawal_index']}")
    print(f"  Risk Adjustment: {res_withdraw['risk_adjustment_factor']}x")
    print(f"  Rationale: {res_withdraw['clinical_rationale']}")
    print("=" * 70)
