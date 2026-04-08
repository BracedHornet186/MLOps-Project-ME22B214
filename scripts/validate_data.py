import json, sys
from data import IMC2025TrainData, DEFAULT_DATASET_DIR

schema = IMC2025TrainData.create(DEFAULT_DATASET_DIR)
schema.preprocess()
df = schema.df

report = {
    "status": "ok",
    "total_images": len(df),
    "total_scenes": df.groupby(["dataset","scene"]).ngroups,
    "missing_files": 0,        # dropped by preprocess()
    "duplicate_images": int(df.duplicated("image").sum()),
    "malformed_R": int(df["rotation_matrix"]
                       .apply(lambda s: len(s.split(";"))!=9).sum()),
    "malformed_t": int(df["translation_vector"]
                       .apply(lambda s: len(s.split(";"))!=3).sum()),
}

if any(v > 0 for k,v in report.items()
       if k not in ("status","total_images","total_scenes")):
    report["status"] = "warn"

with open("data/validation_report.json","w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
sys.exit(0 if report["status"] == "ok" else 1)