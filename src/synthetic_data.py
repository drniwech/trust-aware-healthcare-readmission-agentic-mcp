import pandas as pd
from faker import Faker
import numpy as np

fake = Faker()

def generate_synthetic_ehr(patient_id: str = "12345", n_patients: int = 1) -> dict:
    """Generate synthetic FHIR-like patient record for readmission prediction."""
    data = []
    for _ in range(n_patients):
        age = fake.random_int(18, 95)
        comorbidities = fake.random_int(0, 8)
        lab_glucose = round(np.random.normal(120, 30), 1)
        lab_creatinine = round(np.random.normal(1.2, 0.5), 2)
        admission_type = np.random.choice(["emergency", "elective", "urgent"])
        length_of_stay = fake.random_int(1, 14)
        readmission_risk = 1 if (age > 70 or comorbidities > 4 or lab_glucose > 160) else 0

        record = {
            "patient_id": patient_id,
            "age": age,
            "comorbidities_count": comorbidities,
            "lab_glucose": lab_glucose,
            "lab_creatinine": lab_creatinine,
            "admission_type": admission_type,
            "length_of_stay": length_of_stay,
            "notes": f"Patient presents with {fake.sentence()}. History of {fake.random_element(['hypertension', 'diabetes', 'COPD'])}.",
            "true_readmission_30d": readmission_risk
        }
        data.append(record)
    return pd.DataFrame(data).to_dict(orient="records")[0] if n_patients == 1 else data
