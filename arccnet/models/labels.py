import numpy as np

LABEL_TO_INDEX = {
    "QS": 0,
    "IA": 1,
    "Alpha": 2,
    "Beta": 3,
    "Beta-Gamma": 4,
    "Beta-Delta": 5,
    "Beta-Gamma-Delta": 6,
    "Gamma": 7,
    "Gamma-Delta": 8,
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

GREEK_MAPPING = {
    "Alpha": "α",
    "Beta": "β",
    "Gamma": "γ",
    "Delta": "δ",
}


def convert_to_greek_label(names_array):
    def map_to_greek(name):
        parts = name.split("-")
        greek_parts = [GREEK_MAPPING.get(part, part) for part in parts]
        return "-".join(greek_parts)

    return np.array([map_to_greek(name) for name in names_array])


# Label mappings for training
QS_IA_AR_MAPPING = {
    "QS": "QS",
    "IA": "IA",
    "Alpha": "AR",
    "Beta": "AR",
    "Beta-Delta": "AR",
    "Beta-Gamma": "AR",
    "Beta-Gamma-Delta": "AR",
    "Gamma": "AR",
    "Gamma-Delta": "AR",
}

IA_AR_MAPPING = {
    "QS": None,
    "IA": "IA",
    "Alpha": "AR",
    "Beta": "AR",
    "Beta-Delta": "AR",
    "Beta-Gamma": "AR",
    "Beta-Gamma-Delta": "AR",
    "Gamma": "AR",
    "Gamma-Delta": "AR",
}

QS_IA_MAPPING = {
    "QS": "QS",
    "IA": "IA",
    "Alpha": None,
    "Beta": None,
    "Beta-Delta": None,
    "Beta-Gamma": None,
    "Beta-Gamma-Delta": None,
    "Gamma": None,
    "Gamma-Delta": None,
}

QS_IA_A_B_BG_MAPPING = {
    "QS": "QS",
    "IA": "IA",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

A_B_BG_MAPPING = {
    "QS": None,
    "IA": None,
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

LABEL_MAPPING_DICT = {
    "qs-ia-ar": QS_IA_AR_MAPPING,
    "ia-ar": IA_AR_MAPPING,
    "qs-ia": QS_IA_MAPPING,
    "qs-ia-a-b-bg": QS_IA_A_B_BG_MAPPING,
    "a-b-bg": A_B_BG_MAPPING,
}
