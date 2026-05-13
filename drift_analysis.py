"""
drift_analysis.py
-----------------
Analyse de data drift avec Evidently AI.

Usage :
    python drift_analysis.py
"""

import catboost
import pandas as pd
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from evidently import Report
from evidently.presets import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import ColumnMapping


# récupération des données du train et du dataframe X_train_api pour Evidently::
X_train_api = pd.read_parquet("data/X_train_api.parquet")

catboost_model = CatBoostClassifier(random_state=42)
catboost_model.fit(X_train_api, X_train_api["TARGET"])

# ---------------------------------------------------------------------------
# Column Mapping
# ---------------------------------------------------------------------------
column_mapping = ColumnMapping()

column_mapping.numerical_features = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]

column_mapping.categorical_features = [
    "CODE_GENDER_M",
    "NAME_EDUCATION_TYPE_Higher_education",
]

column_mapping.target = "TARGET"
column_mapping.prediction = "prediction"

# ---------------------------------------------------------------------------
# Drift Analysis
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    column_mapping.numerical_features
    + column_mapping.categorical_features
)

data_drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])

data_drift_report.run(
    reference_data=X_train_api[FEATURE_NAMES],
    current_data=X_train_api[FEATURE_NAMES],
    column_mapping=column_mapping,
)

data_drift_report