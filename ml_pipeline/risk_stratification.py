"""
4-Stage Clinical Risk Stratification and DermaSensor 1-10 Spectral Score Engine.
Replaces biopsy-dependent TNM staging with non-invasive visual risk triage.
"""

def map_to_4_tier_risk(predicted_code, probability_dict=None, patient_metadata=None):
    """
    Maps 7 HAM10000 disease categories to actionable clinical risk tiers:
    - Stage 1: Benign (Melanocytic Nevi 'nv', Vascular 'vasc', Dermatofibroma 'df')
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
    else: # nv, vasc, df
        stage_num = 1
        stage_name = 'Stage 1: Benign'
        risk_level = 'Benign'
        urgency = 'Reassuring'
        action = 'Reassuring. Routine Annual Skin Surveillance'
        base_spectral_score = 1.8 if code == 'nv' else (1.3 if code == 'vasc' else 2.0)
        
    # 2. Patient Clinical Context Calibration
    spectral_score = base_spectral_score
    if patient_metadata:
        age = patient_metadata.get('age', 50)
        if age > 60:
            spectral_score += 0.4
        elif age < 30:
            spectral_score -= 0.3
            
        if patient_metadata.get('history_of_melanoma', False):
            spectral_score += 0.8
            
        loc = str(patient_metadata.get('localization', '')).lower()
        if loc in ['face', 'scalp', 'back']:
            spectral_score += 0.2
            
    # Clamp score to [1.0, 10.0]
    spectral_score = round(max(1.0, min(10.0, spectral_score)), 1)
    
    # 3. DermaSensor Binary Clinical Triage Recommendation
    dermasensor_result = 'INVESTIGATE FURTHER' if spectral_score >= 5.5 else 'MONITOR'
    
    return {
        'predicted_class': code,
        'risk_stage': stage_name,
        'risk_stage_num': stage_num,
        'risk_level': risk_level,
        'urgency_level': urgency,
        'action_needed': action,
        'spectral_score': spectral_score,
        'dermasensor_assessment': dermasensor_result
    }

if __name__ == '__main__':
    test_res = map_to_4_tier_risk('mel', patient_metadata={'age': 65, 'history_of_melanoma': True, 'localization': 'back'})
    print("Risk Stratification Test Result:", test_res)
    assert test_res['risk_stage_num'] == 4
    assert test_res['dermasensor_assessment'] == 'INVESTIGATE FURTHER'
    print("Risk Stratification logic verified!")
