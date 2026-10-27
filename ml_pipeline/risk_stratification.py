"""
4-Stage Clinical Risk Stratification, DermaSensor Spectral Scoring,
and Explicit Binary Cancer Verification Engine.
Replaces biopsy-dependent TNM staging with non-invasive visual risk triage.
"""

def compute_cancer_verification(probability_dict: dict | None = None, is_valid_skin_upload: bool = True) -> dict:
    """
    Computes explicit binary cancer verification and risk tiers from 7-class HAM10000 probabilities.
    
    Malignant Group:
      - 'mel': Melanoma (Invasive / in situ)
      - 'bcc': Basal Cell Carcinoma
      - 'akiec': Actinic Keratosis / Bowen's Intraepithelial Carcinoma
    
    Benign Group:
      - 'nv': Melanocytic Nevus
      - 'bkl': Benign Keratosis (Seborrheic / Solar Lentigo)
      - 'df': Dermatofibroma
      - 'vasc': Vascular Lesion (Hemangioma / Angiokeratoma)
      - 'wound': Acute Resolving Cutaneous Trauma / Scab (if present)
      
    Medically Sensitive Triage Thresholds:
      - Malignant Probability >= 0.25 (25%): Triggers Suspected Malignant / Urgent Clinical Biopsy
      - Malignant Probability 0.12 to 0.25: Triggers Moderate Risk / Dermoscopic Surveillance (3-6 mo)
      - Malignant Probability < 0.12: Triggers Likely Benign / Reassuring Annual Check
    """
    if not probability_dict or not isinstance(probability_dict, dict):
        probability_dict = {'mel': 0.0, 'bcc': 0.0, 'akiec': 0.0, 'nv': 1.0, 'bkl': 0.0, 'df': 0.0, 'vasc': 0.0}

    # Extract probabilities
    p_mel = float(probability_dict.get('mel', 0.0))
    p_bcc = float(probability_dict.get('bcc', 0.0))
    p_akiec = float(probability_dict.get('akiec', 0.0))

    p_nv = float(probability_dict.get('nv', 0.0))
    p_bkl = float(probability_dict.get('bkl', 0.0))
    p_df = float(probability_dict.get('df', 0.0))
    p_vasc = float(probability_dict.get('vasc', 0.0))
    p_wound = float(probability_dict.get('wound', 0.0))

    raw_malignant = p_mel + p_bcc + p_akiec
    raw_benign = p_nv + p_bkl + p_df + p_vasc + p_wound

    total = raw_malignant + raw_benign
    if total > 0:
        mal_prob = raw_malignant / total
        ben_prob = raw_benign / total
    else:
        mal_prob = 0.0
        ben_prob = 1.0

    mal_prob = round(float(mal_prob), 4)
    ben_prob = round(float(ben_prob), 4)

    if not is_valid_skin_upload:
        is_suspected = False
        verdict = "Invalid Skin Upload / Verification Failed"
        tier = "Uncertain / Out-of-Distribution"
    elif mal_prob >= 0.25:
        is_suspected = True
        verdict = "Suspected Malignant / Clinical Biopsy Recommended"
        tier = "High Malignancy Risk"
    elif mal_prob >= 0.12:
        is_suspected = False
        verdict = "Moderate Risk / Monitor Closely"
        tier = "Moderate Risk"
    else:
        is_suspected = False
        verdict = "Likely Benign"
        tier = "Low / Reassuring"

    class_breakdown = {
        'mel': round(p_mel, 4),
        'bcc': round(p_bcc, 4),
        'akiec': round(p_akiec, 4),
        'nv': round(p_nv, 4),
        'bkl': round(p_bkl, 4),
        'df': round(p_df, 4),
        'vasc': round(p_vasc, 4),
    }
    if 'wound' in probability_dict:
        class_breakdown['wound'] = round(p_wound, 4)

    return {
        "is_cancer_suspected": is_suspected,
        "verdict_label": verdict,
        "malignant_probability": mal_prob,
        "benign_probability": ben_prob,
        "risk_tier": tier,
        "class_breakdown": class_breakdown
    }


def map_to_4_tier_risk(predicted_code, probability_dict=None, patient_metadata=None, is_valid_skin_upload=True):
    """
    Maps 7 HAM10000 disease categories to actionable clinical risk tiers:
    - Stage 1: Benign (Melanocytic Nevi 'nv', Vascular 'vasc', Dermatofibroma 'df', Cutaneous Wound 'wound')
    - Stage 2: Atypical (Benign Keratosis-like 'bkl')
    - Stage 3: Pre-Malignant (Actinic Keratosis / Bowen's 'akiec')
    - Stage 4: Malignant (Melanoma 'mel', Basal Cell Carcinoma 'bcc')
    """
    code = predicted_code.lower()
    
    # 1. Base Class Mapping
    if code in ['mel', 'bcc']:
        stage_num = 4
        stage_name = 'Stage 4: Malignant'
        risk_level = 'Malignant'
        urgency = 'Critical'
        action = 'Urgent Excisional Biopsy & Oncology Specialist Referral'
        base_spectral_score = 9.2 if code == 'mel' else 8.5
    elif code in ['akiec']:
        stage_num = 3
        stage_name = 'Stage 3: Pre-Malignant'
        risk_level = 'Pre-Malignant'
        urgency = 'High'
        action = 'Prompt Evaluation & Targeted Cryotherapy / Topical 5-FU'
        base_spectral_score = 6.8
    elif code in ['bkl']:
        stage_num = 2
        stage_name = 'Stage 2: Atypical'
        risk_level = 'Atypical / Indeterminate'
        urgency = 'Moderate'
        action = 'Short-Term Dermoscopic Observation (3-6 Months)'
        base_spectral_score = 4.2
    elif code in ['wound']:
        stage_num = 1
        stage_name = 'Stage 1: Benign'
        risk_level = 'Benign (Non-Cancerous)'
        urgency = 'Reassuring'
        action = 'Normal healing skin wound / scab. Zero oncological malignancy. Conservative wound hygiene care.'
        base_spectral_score = 1.5
    else: # nv, vasc, df
        stage_num = 1
        stage_name = 'Stage 1: Benign'
        risk_level = 'Benign'
        urgency = 'Reassuring'
        action = 'Reassuring. Routine Annual Skin Surveillance'
        base_spectral_score = 1.8 if code == 'nv' else (1.3 if code == 'vasc' else 2.0)
        
    # 2. Continuous Probability-Weighted Spectral Score (DermaSensor 1.0 - 10.0)
    class_weights = {
        'mel': 9.6,
        'bcc': 8.6,
        'akiec': 6.8,
        'bkl': 4.2,
        'df': 2.0,
        'nv': 1.8,
        'vasc': 1.3,
        'wound': 1.2,
    }

    if probability_dict and isinstance(probability_dict, dict) and len(probability_dict) > 0:
        weighted_score = sum(
            float(probability_dict.get(c, 0.0)) * weight
            for c, weight in class_weights.items()
        )
    else:
        weighted_score = class_weights.get(code, 2.0)

    # 3. Patient Clinical Context Calibration
    spectral_score = weighted_score
    if patient_metadata:
        age = patient_metadata.get('age', 50)
        if age > 60:
            spectral_score += 0.3
        elif age < 30:
            spectral_score -= 0.3
            
        # Only elevate for personal history if not an acute wound
        if patient_metadata.get('history_of_melanoma', False) and code != 'wound':
            spectral_score += 0.8
            
        loc = str(patient_metadata.get('localization', '')).lower()
        if loc in ['face', 'scalp', 'back'] and code != 'wound':
            spectral_score += 0.2

    # Clamp score to [1.0, 10.0]
    spectral_score = round(max(1.0, min(10.0, spectral_score)), 1)
    
    # 4. DermaSensor Binary Clinical Triage Recommendation
    dermasensor_result = 'INVESTIGATE FURTHER' if spectral_score >= 5.5 else 'MONITOR'
    
    # Ensure risk stage strictly matches calibrated score
    if spectral_score >= 7.5 and stage_num < 4:
        stage_num = 4
        stage_name = 'Stage 4: Malignant'
        risk_level = 'Malignant'
        urgency = 'Critical'
        action = 'Urgent Excisional Biopsy & Oncology Specialist Referral'
    elif spectral_score >= 5.5 and stage_num < 3 and code not in ['wound']:
        stage_num = 3
        stage_name = 'Stage 3: Pre-Malignant'
        risk_level = 'Pre-Malignant'
        urgency = 'High'
        action = 'Prompt Evaluation & Targeted Cryotherapy / Topical 5-FU'
    
    cancer_verif = compute_cancer_verification(probability_dict, is_valid_skin_upload=is_valid_skin_upload)

    return {
        'predicted_class': code,
        'risk_stage': stage_name,
        'risk_stage_num': stage_num,
        'risk_level': risk_level,
        'urgency_level': urgency,
        'action_needed': action,
        'spectral_score': spectral_score,
        'dermasensor_assessment': dermasensor_result,
        'cancer_verification': cancer_verif,
    }


if __name__ == '__main__':
    test_probs = {'mel': 0.35, 'bcc': 0.05, 'akiec': 0.02, 'nv': 0.50, 'bkl': 0.05, 'df': 0.02, 'vasc': 0.01}
    test_res = map_to_4_tier_risk(
        'mel',
        probability_dict=test_probs,
        patient_metadata={'age': 65, 'history_of_melanoma': True, 'localization': 'back'}
    )
    print("Risk Stratification Test Result:", test_res)
    assert test_res['risk_stage_num'] == 4
    assert test_res['cancer_verification']['is_cancer_suspected'] is True
    assert test_res['cancer_verification']['malignant_probability'] == 0.42
    print("Risk Stratification and Cancer Verification verified successfully!")
