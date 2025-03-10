import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd

# Your columns list from above
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
    'sport': 'object',    # treat as string to avoid mixed types
    'dsport': 'object',
    'attack_cat': 'str'
}



# Load one CSV file without headers, using the column list
path = "../data/"
df1 = pd.read_csv(path+'UNSW-NB15_1.csv', header=None, names=columns, dtype=dtypes, low_memory=False)
df2 = pd.read_csv(path+'UNSW-NB15_2.csv', header=None, names=columns, dtype=dtypes, low_memory=False)
df3 = pd.read_csv(path+'UNSW-NB15_3.csv', header=None, names=columns, dtype=dtypes, low_memory=False)
df4 = pd.read_csv(path+'UNSW-NB15_4.csv', header=None, names=columns, dtype=dtypes, low_memory=False)

# Combine seperate files into 1 dataframe
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
combined_df['attack_cat'] = combined_df['attack_cat'].str.strip() #cleaning up whitespace that's in the attack_cat column

# print(combined_df['attack_cat'].value_counts()) #these display only the counts of the attack data
# print(combined_df['label'].value_counts()) #these display the counts of normal vs attack - 0 is normal, 1 is attack

# making a copy of the data frame for cleaning
df = combined_df.copy()

# fill missing attack categories with keyword Normal
df['attack_cat'] = df['attack_cat'].replace('', 'Normal')
df['attack_cat'] = df['attack_cat'].fillna('Normal')

# seperate normal and attack samples
normal_df = df[df['label'] == 0]
attack_df = df[df['label'] == 1]

# undersample normal data to match attack data
normal_sampled = normal_df.sample(n=len(attack_df), random_state=16)

# sepearte each attack category for undersampling generic data
generic_df = df[df['attack_cat'] == 'Generic']
exploits_df = df[df['attack_cat'] == 'Exploits']
fuzzers_df = df[df['attack_cat'] == 'Fuzzers']
dos_df = df[df['attack_cat'] == 'DoS']
recon_df = df[df['attack_cat'] == 'Reconnaissance']
analysis_df = df[df['attack_cat'] == 'Analysis']
backdoor_df = df[df['attack_cat'] == 'Backdoor']
shellcode_df = df[df['attack_cat'] == 'Shellcode']
backdoors_df = df[df['attack_cat'] == 'Backdoors']
worms_df = df[df['attack_cat'] == 'Worms']

# undersample generic data to match exploit data
generic_sampled = generic_df.sample(n=len(exploits_df), random_state=16) 

attacks_balanced = pd.concat([
    generic_sampled,
    exploits_df,
    fuzzers_df,
    dos_df,
    recon_df,
    analysis_df,
    backdoor_df,
    shellcode_df,
    backdoors_df,
    worms_df
])

balanced_df = pd.concat([normal_sampled, attacks_balanced]) #final dataset
balanced_df = balanced_df.sample(frac=1, random_state=16).reset_index(drop=True) #shuffled

train_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=16,
    stratify=balanced_df['attack_cat']  #stratify the split so that the proportions of this column is maintained in both the test and training split
)

path = "../clean_data/"
train_df.to_csv(path+"balanced_train.csv", index=False)
test_df.to_csv(path+"balanced_test.csv", index=False)
