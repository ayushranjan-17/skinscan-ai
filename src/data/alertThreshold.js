// Must match BIOPSY_ALERT_THRESHOLD in ml_pipeline/risk_stratification.py (scripts/verify_demo.py checks this).
export const BIOPSY_ALERT_THRESHOLD = 0.175;
// Lower edge of the "moderate risk" band, kept at 0.48 of the alert level.
export const MODERATE_RISK_THRESHOLD = Math.round(BIOPSY_ALERT_THRESHOLD * 0.48 * 1000) / 1000;
export const ALERT_PERCENT = Math.round(BIOPSY_ALERT_THRESHOLD * 1000) / 10;
