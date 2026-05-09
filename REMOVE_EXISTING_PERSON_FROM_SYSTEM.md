## Steps to Remove a Person

### 1. **Remove from Configuration**
Edit config.py and delete the person from the `PEOPLE` list:

```python
PEOPLE: list[tuple[str, str, str]] = [
    ("ahsan_22K-4176", "Ahsan Ali", "22K-4176"),
    ("anas_m_22K-4548", "Mohammad Anas", "22K-4548"),
    ("anas_k_22K-4483", "Anas Khan", "22K-4483"),
    # Remove: ("sarah_22K-5000", "Sarah Khan", "22K-5000"),
]
```

### 2. **Delete Raw Images**
Remove the person's raw dataset folder:
```powershell
Remove-Item -Recurse -Force "data\raw\sarah_22K-5000"
```

### 3. **Delete Processed Images**
Remove processed/augmented images:
```powershell
Remove-Item -Recurse -Force "data\processed\sarah_22K-5000"
```

### 4. **Delete Encoding Files**
Remove embeddings for that person:
```powershell
Remove-Item -Recurse -Force "data\encodings\sarah_22K-5000"
```

### 5. **Retrain the System**
Re-encode and retrain with the remaining people (venv must be activated):

```powershell
python -m src.preprocess
python -m src.encode
python -m src.train_classifier
```

### 6. **Verify**
Test real-time recognition:
```powershell
python -m src.recognize
```

## Summary

| Step | Action |
|------|--------|
| 1 | Remove person from `PEOPLE` in config.py |
| 2 | Delete `data/raw/<person_label>/` folder |
| 3 | Delete `data/processed/<person_label>/` folder |
| 4 | Delete `data/encodings/<person_label>/` folder |
| 5 | Run `preprocess` → `encode` → `train_classifier` |
| 6 | Test with `recognize` |

**Note:** The person will no longer be recognized after retraining. The existing classifier model (classifier.pkl) will be updated to exclude that person.