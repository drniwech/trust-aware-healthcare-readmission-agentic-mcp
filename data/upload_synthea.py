import os
import requests
import json
import glob

# ================== CONFIGURATION ==================
SERVER_URL = "http://localhost:8080/fhir"

# Path to your Synthea fhir folder
FHIR_DIR = "fhir"

# ===================================================

def upload_bundle(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    response = requests.post(
        SERVER_URL,
        json=bundle,
        headers={"Content-Type": "application/fhir+json"}
    )

    print(f"Uploaded {os.path.basename(file_path)} → Status: {response.status_code}")
    if response.status_code >= 400:
        try:
            error = response.json()
            print("  Error:", error.get("issue", [{}])[0].get("diagnostics", response.text))
        except:
            print("  Raw error:", response.text[:500])
    return response.status_code

# === Step 1: Upload shared resources first (critical!) ===
print("=== Uploading shared resources (Hospital + Practitioners) ===")

shared_patterns = ["hospitalInformation*.json", "practitionerInformation*.json"]

for pattern in shared_patterns:
    files = glob.glob(os.path.join(FHIR_DIR, pattern))
    for f in sorted(files):          # sort for consistent order
        upload_bundle(f)

# === Step 2: Upload patient bundles ===
print("\n=== Uploading patient bundles ===")

patient_files = glob.glob(os.path.join(FHIR_DIR, "*.json"))
for f in sorted(patient_files):
    filename = os.path.basename(f)
    # Skip the shared files we already uploaded
    if any(skip in filename for skip in ["hospitalInformation", "practitionerInformation"]):
        continue
    upload_bundle(f)

print("\nUpload process completed.")
