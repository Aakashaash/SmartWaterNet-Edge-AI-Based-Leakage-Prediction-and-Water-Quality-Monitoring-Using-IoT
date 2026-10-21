

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
)
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from utils import (
    SEED, DEVICE, load_and_split, print_class_distribution, scale_features,
    to_tensor, make_loader, train_classifier, train_regressor, eval_classifier,
)
from gja import run_gja_sensor_selection
from adqcl import ADQCL, train_adqcl_continual
from networks import SmartWaterNetQuality, WaterQualityPredictor, LeakRateEstimator


# ==================================================================
# 6. Edge decision layer
# ==================================================================
class EdgeDecisionLayer:
    def __init__(self, leak_prob_thr=0.7, leak_rate_thr=5.0):
        self.leak_prob_thr = leak_prob_thr
        self.leak_rate_thr = leak_rate_thr

    def act(self, leak_prob, leak_rate, quality_class):
        a = {"leak_warning": False, "quality_warning": False,
             "valve_close": False, "supply_isolate": False}
        if leak_prob >= self.leak_prob_thr:
            a["leak_warning"] = True
        if leak_rate >= self.leak_rate_thr:
            a["leak_warning"] = True
            a["valve_close"] = True
        if quality_class in ("Warning", "Poor"):
            a["quality_warning"] = True
        if quality_class == "Poor":
            a["supply_isolate"] = True
        return a


# ==================================================================
# 7. TASK PIPELINES
# ==================================================================
def task_leakage_detection(train_df, val_df, test_df, n_continual_rounds=3):
    print("\n" + "=" * 72)
    print("TASK 1: Leakage Detection  (GJA + ADQ-CL with EWC continual learning)")
    print("=" * 72)
    feats = ["pressure_bar", "flow_rate_lpm", "pressure_change_bar",
             "flow_change_lpm", "historical_pressure_bar", "historical_flow_rate_lpm"]

    nodes_matrix = StandardScaler().fit_transform(train_df[feats].values)
    mask, _, _ = run_gja_sensor_selection(nodes_matrix, feats, min_keep=4,
                                          pop_size=30, max_iter=80)
    sel_idx = np.where(mask == 1)[0]
    if len(sel_idx) < 3:
        mi = mutual_info_classif(train_df[feats].values, train_df["leak_state"].values,
                                 random_state=SEED)
        sel_idx = np.argsort(mi)[-4:]
    sel_feats = [feats[i] for i in sel_idx]
    print(f"Final selected sensors: {sel_feats}")

    Xtr = train_df[sel_feats].values
    Xva = val_df[sel_feats].values
    Xte = test_df[sel_feats].values
    ytr = train_df["leak_state"].values.astype(int)
    yva = val_df["leak_state"].values.astype(int)
    yte = test_df["leak_state"].values.astype(int)

    Xtr, Xva, Xte, leak_scaler = scale_features(Xtr, Xva, Xte)
    Xtr, ytr = RandomOverSampler(random_state=SEED).fit_resample(Xtr, ytr)
    print("After RandomOverSampler:")
    print_class_distribution("train", ytr)

    idx = np.random.permutation(len(Xtr))
    batches = [(Xtr[c], ytr[c]) for c in np.array_split(idx, n_continual_rounds)]
    val_loader = make_loader(*to_tensor(Xva, yva), shuffle=False)
    test_loader = make_loader(*to_tensor(Xte, yte), shuffle=False)

    model = ADQCL(in_dim=len(sel_feats), latent=32).to(DEVICE)
    model = train_adqcl_continual(model, batches, val_loader,
                                  epochs_per_task=100, lam_ewc=1e2,
                                  task_name="ADQ-CL (EWC)")
    metrics = eval_classifier(model, test_loader, ["Normal", "Leakage"],
                              "Leakage Detection (ADQ-CL)", adqcl=True)
    return model, sel_feats, leak_scaler, metrics


def task_quality_classification(train_df, val_df, test_df):
    print("\n" + "=" * 72)
    print("TASK 2: Water Quality Classification (SmartWaterNet)")
    print("=" * 72)
    feats = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l",
             "electrical_conductivity_us_cm", "temperature_c"]

    le = LabelEncoder().fit(pd.concat([train_df, val_df, test_df])["quality_class"])
    Xtr = train_df[feats].values
    Xva = val_df[feats].values
    Xte = test_df[feats].values
    ytr = le.transform(train_df["quality_class"])
    yva = le.transform(val_df["quality_class"])
    yte = le.transform(test_df["quality_class"])

    Xtr, Xva, Xte, quality_scaler = scale_features(Xtr, Xva, Xte)
    try:
        Xtr, ytr = ADASYN(random_state=SEED, n_neighbors=3).fit_resample(Xtr, ytr)
    except Exception as e:
        print("ADASYN failed, using SMOTE:", e)
        Xtr, ytr = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(Xtr, ytr)
    print("After oversampling:")
    print_class_distribution("train", ytr)

    counts = np.bincount(ytr)
    class_w = torch.tensor(counts.max() / counts, dtype=torch.float32, device=DEVICE)
    print("Class weights:", class_w.cpu().numpy())

    tr = make_loader(*to_tensor(Xtr, ytr))
    va = make_loader(*to_tensor(Xva, yva), shuffle=False)
    te = make_loader(*to_tensor(Xte, yte), shuffle=False)

    model = SmartWaterNetQuality(len(feats), len(le.classes_), hidden=128).to(DEVICE)
    model = train_classifier(model, tr, va, epochs=100, lr=5e-4,
                             class_weights=class_w, task_name="SmartWaterNet")
    metrics = eval_classifier(model, te, list(le.classes_),
                              "Water Quality (SmartWaterNet)")
    return model, le, quality_scaler, metrics


def task_water_quality_prediction(train_df, val_df, test_df):
    print("\n" + "=" * 72)
    print("TASK 3+4: Water Quality Prediction (Turbidity & DO)")
    print("=" * 72)
    feats = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l",
             "electrical_conductivity_us_cm", "temperature_c"]
    targets = ["future_turbidity_ntu_10min", "future_do_mg_l_10min"]

    Xtr = train_df[feats].values
    Xva = val_df[feats].values
    Xte = test_df[feats].values
    ytr = train_df[targets].values
    yva = val_df[targets].values
    yte = test_df[targets].values

    Xtr, Xva, Xte, _ = scale_features(Xtr, Xva, Xte)
    ys = StandardScaler()
    ytr_s = ys.fit_transform(ytr)
    yva_s = ys.transform(yva)
    yte_s = ys.transform(yte)

    tr = make_loader(*to_tensor(Xtr, ytr_s))
    va = make_loader(*to_tensor(Xva, yva_s), shuffle=False)
    te = make_loader(*to_tensor(Xte, yte_s), shuffle=False)

    model = WaterQualityPredictor(len(feats), out_dim=2, hidden=128).to(DEVICE)
    model = train_regressor(model, tr, va, epochs=100, task_name="WQ Predictor")

    model.eval()
    pr, tr_ = [], []
    with torch.no_grad():
        for xb, yb in te:
            pr.append(model(xb.to(DEVICE)).cpu().numpy())
            tr_.append(yb.numpy())
    pr = ys.inverse_transform(np.concatenate(pr))
    tr_ = ys.inverse_transform(np.concatenate(tr_))

    print("\n===== Water Quality Prediction - TEST =====")
    m = {}
    for i, t in enumerate(targets):
        mae = mean_absolute_error(tr_[:, i], pr[:, i])
        rmse = np.sqrt(mean_squared_error(tr_[:, i], pr[:, i]))
        mape = mean_absolute_percentage_error(tr_[:, i], pr[:, i]) * 100
        print(f"  {t}: MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")
        m[f"MAE_{t}"] = mae
        m[f"RMSE_{t}"] = rmse
        m[f"MAPE_{t}"] = mape
    return model, m


def task_leak_rate_estimation(train_df, val_df, test_df):
    print("\n" + "=" * 72)
    print("TASK 5: Leakage Rate Estimation")
    print("=" * 72)
    feats = ["pressure_bar", "flow_rate_lpm", "pressure_change_bar",
             "flow_change_lpm", "historical_pressure_bar", "historical_flow_rate_lpm"]
    Xtr = train_df[feats].values
    Xva = val_df[feats].values
    Xte = test_df[feats].values
    ytr = train_df["leak_rate_lpm"].values
    yva = val_df["leak_rate_lpm"].values
    yte = test_df["leak_rate_lpm"].values

    Xtr, Xva, Xte, rate_X_scaler = scale_features(Xtr, Xva, Xte)
    ys = StandardScaler()
    ytr_s = ys.fit_transform(ytr.reshape(-1, 1)).ravel()
    yva_s = ys.transform(yva.reshape(-1, 1)).ravel()
    yte_s = ys.transform(yte.reshape(-1, 1)).ravel()

    tr = make_loader(*to_tensor(Xtr, ytr_s))
    va = make_loader(*to_tensor(Xva, yva_s), shuffle=False)
    te = make_loader(*to_tensor(Xte, yte_s), shuffle=False)

    model = LeakRateEstimator(len(feats), hidden=128).to(DEVICE)
    model = train_regressor(model, tr, va, epochs=100, task_name="Leak Rate")

    model.eval()
    pr, tr_ = [], []
    with torch.no_grad():
        for xb, yb in te:
            pr.append(model(xb.to(DEVICE)).cpu().numpy())
            tr_.append(yb.numpy())
    pr = ys.inverse_transform(np.concatenate(pr).reshape(-1, 1)).ravel()
    tr_ = ys.inverse_transform(np.concatenate(tr_).reshape(-1, 1)).ravel()

    mae = mean_absolute_error(tr_, pr)
    rmse = np.sqrt(mean_squared_error(tr_, pr))

    nonzero = np.abs(tr_) > 1e-6
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((tr_[nonzero] - pr[nonzero]) / tr_[nonzero])) * 100
    else:
        mape = float("nan")

    print(f"\n===== Leak Rate - TEST =====")
    print(f"  MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE(non-zero)={mape:.2f}%")
    print(f"  (Note: {100 * (~nonzero).mean():.1f}% of test targets are ~0, "
          f"which inflates raw MAPE.)")
    return model, rate_X_scaler, {"MAE": mae, "RMSE": rmse, "MAPE_nonzero": mape}


# ==================================================================
# 8. Inference helper for edge decision layer
# ==================================================================
def demo_edge_inference(leak_model, leak_scaler, leak_feats,
                        quality_model, quality_le, quality_scaler, quality_feats,
                        rate_model, rate_X_scaler, edge_layer, sample_row):
    rate_feats = ["pressure_bar", "flow_rate_lpm", "pressure_change_bar",
                  "flow_change_lpm", "historical_pressure_bar", "historical_flow_rate_lpm"]

    # ---- leakage (only GJA-selected features)
    x_leak = torch.tensor(leak_scaler.transform(sample_row[leak_feats].values.reshape(1, -1)),
                          dtype=torch.float32).to(DEVICE)
    leak_model.eval()
    with torch.no_grad():
        _, _, q = leak_model(x_leak)
        leak_prob = F.softmax(q, dim=1)[0, 1].item()

    # ---- quality
    x_q = torch.tensor(quality_scaler.transform(sample_row[quality_feats].values.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    quality_model.eval()
    with torch.no_grad():
        qc = quality_model(x_q).argmax(1).item()
    quality_label = quality_le.inverse_transform([qc])[0]

    # ---- leak rate
    x_r = torch.tensor(rate_X_scaler.transform(sample_row[rate_feats].values.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    rate_model.eval()
    with torch.no_grad():
        rate = rate_model(x_r).cpu().numpy().ravel()[0]

    action = edge_layer.act(leak_prob, rate, quality_label)
    print(f"\nEdge inference -> leak_prob={leak_prob:.3f}, "
          f"leak_rate={rate:.2f} L/min, quality={quality_label}")
    print(f"  Action: {action}")
    return action


# ==================================================================
# 9. MAIN
# ==================================================================
if __name__ == "__main__":
    CSV = r"csv"
    train_df, val_df, test_df = load_and_split(CSV)

    print("\nClass distribution per split:")
    print("Train leak_state:", train_df["leak_state"].value_counts().to_dict())
    print("Val   leak_state:", val_df["leak_state"].value_counts().to_dict())
    print("Test  leak_state:", test_df["leak_state"].value_counts().to_dict())
    print("Train quality_class:", train_df["quality_class"].value_counts().to_dict())
    print("Val   quality_class:", val_df["quality_class"].value_counts().to_dict())
    print("Test  quality_class:", test_df["quality_class"].value_counts().to_dict())

    leak_model, selected_nodes, leak_scaler, m1 = task_leakage_detection(train_df, val_df, test_df)
    quality_model, quality_le, quality_scaler, m2 = task_quality_classification(train_df, val_df, test_df)
    wqp_model, m3 = task_water_quality_prediction(train_df, val_df, test_df)
    rate_model, rate_X_scaler, m4 = task_leak_rate_estimation(train_df, val_df, test_df)

    # ---- Demo edge decision ----
    quality_feats = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l",
                     "electrical_conductivity_us_cm", "temperature_c"]

    edge = EdgeDecisionLayer(leak_prob_thr=0.7, leak_rate_thr=5.0)
    demo_edge_inference(
        leak_model, leak_scaler, selected_nodes,
        quality_model, quality_le, quality_scaler, quality_feats,
        rate_model, rate_X_scaler, edge, test_df.iloc[[0]]
    )

    print("\n\n========== FINAL SUMMARY ==========")
    print("GJA-selected nodes  :", selected_nodes)
    print("Leakage Detection   :", m1)
    print("Quality Classif.    :", m2)
    print("WQ Prediction       :", m3)
    print("Leak Rate Est.      :", m4)