import numpy as np

label_to_index = {
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

index_to_label = {v: k for k, v in label_to_index.items()}

greek_mapping = {
    "Alpha": "α",
    "Beta": "β",
    "Gamma": "γ",
    "Delta": "δ",
}


def convert_to_greek_label(names_array):
    def map_to_greek(name):
        parts = name.split("-")
        greek_parts = [greek_mapping.get(part, part) for part in parts]
        return "-".join(greek_parts)

    return np.array([map_to_greek(name) for name in names_array])


# label mappings for training
qs_ia_ar_mapping = {
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

ia_ar_mapping = {
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

qs_ia_mapping = {
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

qs_ia_a_b_bg_mapping = {
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

a_b_bg_mapping = {
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


label_mapping_dict = {
    "qs-ia-ar": qs_ia_ar_mapping,
    "ia-ar": ia_ar_mapping,
    "qs-ia": qs_ia_mapping,
    "qs-ia-a-b-bg": qs_ia_a_b_bg_mapping,
    "a-b-bg": a_b_bg_mapping,
}
