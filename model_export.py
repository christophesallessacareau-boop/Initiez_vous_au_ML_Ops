import mlflow
import mlflow.pyfunc
import cloudpickle          # cloudpickle
import os
from sklearn.metrics import confusion_matrix


def cost_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = fn * 10 + fp * 1
    return -cost


mlflow.set_tracking_uri(r"file:///C:/Users/chris/Initiez_vous_au_ML_Ops/mlruns")
model = mlflow.pyfunc.load_model(
    r"C:/Users/chris/Initiez_vous_au_ML_Ops/mlruns/1/models/m-75f49439c2764e6e91a4371402c42cdc/artifacts"
)

inner = model._model_impl
best_threshold = float(getattr(inner, "best_threshold_", 0.5))

os.makedirs("model_export", exist_ok=True)

with open("model_export/model.pkl", "wb") as f:
    cloudpickle.dump(inner, f)      # cloudpickle.dump

with open("model_export/threshold.txt", "w") as f:
    f.write(str(best_threshold))

taille = os.path.getsize("model_export/model.pkl") / 1e6
print(f"Taille modèle : {taille:.1f} Mo")
print(f"Threshold     : {best_threshold:.4f}")
print("Export terminé ")