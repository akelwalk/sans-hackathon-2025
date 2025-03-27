import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

selected_features = [
    'proto', #Transaction protocol
    'state', #Indicates to the state and its dependent protocol, e.g. ACC, CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD, URH, URN, and (-) (if not used state)
    'dur', #Record total duration
    'sbytes', #Source to destination transaction bytes 
    'dbytes', #Destination to source transaction bytes
    'Spkts', #Source to destination packet count 
    'Dpkts', #Destination to source packet count
    'ct_state_ttl', #No. for each state (6) according to specific range of values for source/destination time to live (10) (11).
    'ct_dst_ltm', #No. of connections of the same destination address (3) in 100 connections according to the last time (26).
    'tcprtt' #TCP connection setup round-trip time, the sum of ’synack’ and ’ackdat’.
]

path = "../clean_data/"

columns = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb',
    'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

dtypes = {
    'sport': 'object',
    'dsport': 'object',
    'attack_cat': 'str'
}

train_df = pd.read_csv(path + "balanced_train.csv", header=None, names=columns, dtype=dtypes, low_memory=False)
test_df = pd.read_csv(path + "balanced_test.csv", header=None, names=columns, dtype=dtypes, low_memory=False)

# Select features & target labels
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train_attack = train_df['attack_cat']
y_test_attack = test_df['attack_cat']

y_train_label = train_df['label']
y_test_label = test_df['label']

# Combine train & test for encoding to avoid unseen label issues
combined_proto = pd.concat([X_train['proto'], X_test['proto']])
combined_state = pd.concat([X_train['state'], X_test['state']])

# Label Encoding
le_proto = LabelEncoder()
le_state = LabelEncoder()

le_proto.fit(combined_proto)
le_state.fit(combined_state)

X_train['proto'] = le_proto.transform(X_train['proto'])
X_test['proto'] = le_proto.transform(X_test['proto'])

X_train['state'] = le_state.transform(X_train['state'])
X_test['state'] = le_state.transform(X_test['state'])

# Convert all columns to numeric (sanity check)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Check for NaNs
print("Missing values in train:", X_train.isnull().sum().sum())
print("Missing values in test:", X_test.isnull().sum().sum())

# Train models
rf_attack = RandomForestClassifier(random_state=16)
rf_label = RandomForestClassifier(random_state=16)

rf_attack.fit(X_train, y_train_attack)
rf_label.fit(X_train, y_train_label)

# Predictions
y_pred_attack = rf_attack.predict(X_test)
y_pred_label = rf_label.predict(X_test)

# Accuracy
attack_accuracy = accuracy_score(y_test_attack, y_pred_attack)
label_accuracy = accuracy_score(y_test_label, y_pred_label)

print(f"Attack Category Prediction Accuracy: {attack_accuracy:.4f}")
print(f"Normal/Attack Label Prediction Accuracy: {label_accuracy:.4f}")
