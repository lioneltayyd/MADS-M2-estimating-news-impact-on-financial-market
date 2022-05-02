# -------------------------------------------------------
# General config 
# -------------------------------------------------------

# Directory path for saving and loading files or objects. 
DIR_DATASET = "dataset" 
DIR_DATASET_COLLECTION = f"{DIR_DATASET}/ner_collection" 
DIR_DATASET_LABELLINGS = f"{DIR_DATASET}/labelling" 
DIR_MLSPACY = "model/spacy_sentiment" 
DIR_MLTOPIC = "model/topic_modelling" 
DIR_MLESTIM = "model/mktmv_estimator" 

# Define th starting and ending date when collecting the ticker data. 
TICKER_DATE_COLLECT = "1998-12-01", "2022-03-21" 

# Define the list of tickers we are interested to investigate on. 
TICKER_TO_COLLECT = ["SPY"] 

# Wikifier URL. 
WIKIFIER_URL = "http://www.wikifier.org/annotate-article"

# -------------------------------------------------------
# Define parameters 
# -------------------------------------------------------

PARAM_SEED = 42

# Wikifier parameters. 
PARAM_THRESHOLD = 0.8 
PARAM_LANG = "en" 

# Define the # of topics. 
PARAM_N_TOPIC = 10 
PARAM_TOP_N_TERM = 20 
