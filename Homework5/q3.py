import hashlib
import pickle
from pathlib import Path

from model import normalize_record


model_path = Path(__file__).with_name("pipeline.bin")
model_bytes = model_path.read_bytes()
expected_sha256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"
if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
    raise ValueError("pipeline.bin does not match the official HW5 model checksum")

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

pipeline = pickle.loads(model_bytes)
probability = float(pipeline.predict_proba([normalize_record(lead)])[0, 1])
print(f"{probability:.3f}")
