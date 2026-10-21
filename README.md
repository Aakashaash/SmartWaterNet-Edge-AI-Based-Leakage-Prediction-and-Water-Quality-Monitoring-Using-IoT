# SmartWaterNet

End-to-end smart water network pipeline:

| Component | Purpose |
|-----------|---------|
| **GJA** (Gekko Japonicus Algorithm) | Optimal sensor node selection |
| **ADQ-CL** (Autoencoder + Deep Q + EWC) | Leakage detection with continual learning |
| **SmartWaterNet** (ST-Graph) | Water quality classification |
| **WaterQualityPredictor** | Turbidity & DO multi-output regression |
| **LeakRateEstimator** | Leak rate regression |
| **EdgeDecisionLayer** | Valve / isolation triggers |

## Layout

```
.
├── utils.py           # Data loading, training/eval helpers, Focal Loss
├── gja.py             # Gekko Japonicus Algorithm
├── adqcl.py           # Autoencoder + Deep Q + EWC
├── networks.py        # SmartWaterNet + regression nets
├── main.py            # Pipelines + entrypoint
├── requirements.txt

```

## Run

```bash
pip install -r requirements.txt
python main.py
```

> Edit the `CSV` path at the bottom of `main.py` to point to your dataset.
