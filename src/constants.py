ID_COL = "SeriesInstanceUID"

LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]

ANEURYSM_NAME = "Aneurysm Present"
ANEURYSM_IDX = LABEL_COLS.index(ANEURYSM_NAME)


# Label pairs that must be swapped whenever a volume is mirrored left-right

LR_SWAP_PAIRS = [
    ("Left Infraclinoid Internal Carotid Artery", "Right Infraclinoid Internal Carotid Artery"),
    ("Left Supraclinoid Internal Carotid Artery", "Right Supraclinoid Internal Carotid Artery"),
    ("Left Middle Cerebral Artery", "Right Middle Cerebral Artery"),
    ("Left Anterior Cerebral Artery", "Right Anterior Cerebral Artery"),
    ("Left Posterior Communicating Artery", "Right Posterior Communicating Artery"),
]

LR_SWAP_INDEX_PAIRS = [(LABEL_COLS.index(a), LABEL_COLS.index(b)) for a, b in LR_SWAP_PAIRS]
