import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

panel_file = config["evaluation"]["panels"]

#CONSTANTS USED IN DIFFERENT MODULES
# ANNOTATIONS JSON KEYS
DATASET_KEY = "ds"
SLIDE_KEY = "slide"
TILE_KEY = "tile"
ID_KEY = "id"
TYPE_KEY = "type"
X_KEY = "x"
Y_KEY = "y"
PHENO_KEY = "positivity"
BG_KEY = "background"
PANEL_KEY = "panel"
SLIDES_KEY = "slides"
TILES_KEY = "tiles"
ANNOTATIONS_KEY = "annotations"

# MATCHED PREDICTION DATA FRAME COLUMNS
PANEL_COL = "panel"
DATASET_COL = "dataset"
SLIDE_COL = "slide"
TILE_COL = "tile"
ID_COL = "id"
FG_KEY = "foreground"
ANN_TYPE_COL = "ann_type"
PRED_TYPE_COL = "pred_type"
ANN_SUBTYPE_COL = "ann_subtype"
PRED_SUBTYPE_COL = "pred_subtype"
DISTANCE_COL = "distance"
ANN_PHENO_COL = "ann_pheno"
PRED_PHENO_COL = "pred_pheno"
INVALID_COL = "invalid"
DETECTED_COL = "detected"
HIT_MAIN_COL = "hit_main"
HIT_SUBTYPE_COL = "hit_subtype"

# PERFORMANCE CSV COLUMNS
ERROR_NUM_COL = "errors"
CASES_NUM_COL = "cases"
ERROR_RATE_COL = "error_rate"
CELL_TYPE_COL = "type"
LABEL_COL = "label"

# ERRORS JSON KEYS
ANN_PHENO_KEY = "ann_phen"
PRED_PHENO_KEY = "pred_phen"
TYPE_FROM_ANN_KEY = "type_from_ann"