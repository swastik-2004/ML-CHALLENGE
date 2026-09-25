"""
Benchmark enhanced blocking rules on the 100k Dev set
to push candidate recall from 82.5% to 90%+.
"""
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import WORK_DIR, DEV_IDS_FILE
from src.data import id_to_int
from src.normalize import norm_path

print("Starting Enhanced Recall Exploration on 100k Dev Set...")
t_start = time.time()

# 1. Load dev IDs and Ground Truth
dev_s1_ids = pd.read_csv(DEV_IDS_FILE)["s1_id"].values
dev_s1_int_set = set(id_to_int(dev_s1_ids))

truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
dev_truth = truth[truth["s1_int"].isin(dev_s1_int_set)].copy()
n_dev_truth = len(dev_truth)
print(f"Loaded {len(dev_s1_ids):,} Dev entities with {n_dev_truth:,} true matching pairs.")

# 2. Load Normalized Data
cols = ["entity_id", "country", "name_norm", "name_core", "name_key", "name_compact", "addr_norm", "addr_key", "house_no", "state"]
s1 = pd.read_parquet(norm_path("train", 1), columns=cols)
tg2 = pd.read_parquet(norm_path("train", 2), columns=cols)
tg3 = pd.read_parquet(norm_path("train", 3), columns=cols)
tg = pd.concat([tg2, tg3], ignore_index=True)
del tg2, tg3

n1 = len(s1)
s1_ids = id_to_int(s1.entity_id.values)
tg_ids = id_to_int(tg.entity_id.values)

dev_mask_s1 = np.isin(s1_ids, list(dev_s1_int_set))
dev_indices_set = set(np.flatnonzero(dev_mask_s1))

# 3. Extract tokens
STOP_WORDS = {
    "the", "and", "inc", "ltd", "corp", "llc", "pvt", "for", "new", "co",
    "company", "group", "services", "private", "limited", "enterprises",
    "solutions", "technologies", "holdings", "industries", "association",
    "sarl", "sas", "eurl", "sa", "sasu", "sci"
}

def extract_tokens(names):
    w1_arr = np.empty(len(names), dtype=object)
    w2_arr = np.empty(len(names), dtype=object)
    p3_arr = np.empty(len(names), dtype=object)
    for i, name in enumerate(names):
        s = str(name).lower().strip() if name is not None else ""
        toks = [t for t in s.split() if len(t) >= 3 and t not in STOP_WORDS]
        w1 = toks[0] if len(toks) > 0 else (s.split()[0] if s.split() else "")
        w2 = toks[1] if len(toks) > 1 else ""
        p3 = w1[:3] if len(w1) >= 3 else w1
        w1_arr[i] = w1
        w2_arr[i] = w2
        p3_arr[i] = p3
    return w1_arr, w2_arr, p3_arr

def extract_street_w1(addrs):
    street_arr = np.empty(len(addrs), dtype=object)
    noise_addr = {"road", "street", "st", "rd", "ave", "lane", "dr", "near", "opp", "fl", "floor", "block", "bldg", "plot", "shop", "hn"}
    for i, a in enumerate(addrs):
        s = str(a).lower().strip() if a is not None else ""
        toks = [t for t in s.split() if not t.isdigit() and len(t) >= 3 and t not in noise_addr]
        street_arr[i] = toks[0] if toks else ""
    return street_arr

print("Extracting derived tokens...")
s1["w1"], s1["w2"], s1["p3"] = extract_tokens(s1["name_core"].values)
tg["w1"], tg["w2"], tg["p3"] = extract_tokens(tg["name_core"].values)

s1["street_w1"] = extract_street_w1(s1["addr_norm"].values)
tg["street_w1"] = extract_street_w1(tg["addr_norm"].values)

# Composite keys
s1["house_p3"] = np.where((s1["house_no"].values != "") & (s1["p3"].values != ""), s1["house_no"].values + "_" + s1["p3"].values, "")
tg["house_p3"] = np.where((tg["house_no"].values != "") & (tg["p3"].values != ""), tg["house_no"].values + "_" + tg["p3"].values, "")

s1["state_w1"] = np.where((s1["state"].values != "") & (s1["w1"].values != ""), s1["state"].values + "_" + s1["w1"].values, "")
tg["state_w1"] = np.where((tg["state"].values != "") & (tg["w1"].values != ""), tg["state"].values + "_" + tg["w1"].values, "")

# NEW KEYS:
# 1. House Number + Street Token: catches pairs where name is completely abbreviated or scrambled!
s1["house_street"] = np.where((s1["house_no"].values != "") & (s1["street_w1"].values != ""), s1["house_no"].values + "_" + s1["street_w1"].values, "")
tg["house_street"] = np.where((tg["house_no"].values != "") & (tg["street_w1"].values != ""), tg["house_no"].values + "_" + tg["street_w1"].values, "")

# 2. House Number + State (for locations with matching house and state, with max_tg <= 25)
s1["house_state"] = np.where((s1["house_no"].values != "") & (s1["state"].values != ""), s1["house_no"].values + "_" + s1["state"].values, "")
tg["house_state"] = np.where((tg["house_no"].values != "") & (tg["state"].values != ""), tg["house_no"].values + "_" + tg["state"].values, "")

# 3. House Number + First Word of Name: catches sub-brands and address shifts
s1["house_w1"] = np.where((s1["house_no"].values != "") & (s1["w1"].values != ""), s1["house_no"].values + "_" + s1["w1"].values, "")
tg["house_w1"] = np.where((tg["house_no"].values != "") & (tg["w1"].values != ""), tg["house_no"].values + "_" + tg["w1"].values, "")

# 4. First Two Words of Name: e.g. "cozy_grill", "red_global"
s1["w1_w2"] = np.where((s1["w1"].values != "") & (s1["w2"].values != ""), s1["w1"].values + "_" + s1["w2"].values, "")
tg["w1_w2"] = np.where((tg["w1"].values != "") & (tg["w2"].values != ""), tg["w1"].values + "_" + tg["w2"].values, "")

# 5. Name Stem (stripping trailing generic business noise words)
GENERIC_NOISE = {
    "enterprises", "solutions", "technologies", "services", "group", 
    "industries", "holdings", "consulting", "properties", "management", 
    "associates", "partners", "ventures", "global", "international", 
    "systems", "agency", "logistics", "products", "marketing", "center", 
    "store", "shop", "care", "restaurant", "cafe", "grill", "hotel"
}

def extract_name_stem(series):
    stems = np.empty(len(series), dtype=object)
    for i, name in enumerate(series.values):
        toks = [w for w in str(name).split() if len(w) >= 3 and w not in GENERIC_NOISE]
        stems[i] = " ".join(toks[:2]) if toks else (str(name).split()[0] if str(name).split() else "")
    return stems

print("Extracting name stems...")
s1["name_stem"] = extract_name_stem(s1["name_core"])
tg["name_stem"] = extract_name_stem(tg["name_core"])

# 6. House Number + 2-letter prefix (for severe phonetic shifts, e.g. "urban" vs "arpn")
s1["p2"] = s1["name_norm"].str.slice(0, 2)
tg["p2"] = tg["name_norm"].str.slice(0, 2)
s1["house_p2"] = np.where((s1["house_no"].values != "") & (s1["p2"].values != ""), s1["house_no"].values + "_" + s1["p2"].values, "")
tg["house_p2"] = np.where((tg["house_no"].values != "") & (tg["p2"].values != ""), tg["house_no"].values + "_" + tg["p2"].values, "")

# 7. House Number + Second Word of Name
s1["house_w2"] = np.where((s1["house_no"].values != "") & (s1["w2"].values != ""), s1["house_no"].values + "_" + s1["w2"].values, "")
tg["house_w2"] = np.where((tg["house_no"].values != "") & (tg["w2"].values != ""), tg["house_no"].values + "_" + tg["w2"].values, "")

both_country = pd.concat([s1["country"], tg["country"]], ignore_index=True)
cc, _ = pd.factorize(both_country)
del both_country

def run_pass(s1_col, tg_col, max_s1=20, max_tg=50, min_len=0):
    both_col = pd.concat([pd.Series(s1_col), pd.Series(tg_col)], ignore_index=True)
    kc, _ = pd.factorize(both_col)
    key = kc.astype(np.int64) * (int(cc.max()) + 1) + cc
    
    vals = both_col.values
    mask = (vals == "") | (vals == "None")
    if min_len > 0:
        mask = mask | (both_col.str.len() < min_len).values
    key[mask] = -1
    del both_col, kc
    
    ka, kb = key[:n1], key[n1:]
    nk = int(key.max()) + 1
    ca = np.bincount(ka[ka >= 0], minlength=nk)
    cb = np.bincount(kb[kb >= 0], minlength=nk)
    
    ok = (ca >= 1) & (ca <= max_s1) & (cb >= 1) & (cb <= max_tg)
    ia = np.flatnonzero((ka >= 0) & ok[np.maximum(ka, 0)])
    ib = np.flatnonzero((kb >= 0) & ok[np.maximum(kb, 0)])
    
    pairs = pd.DataFrame({"s1": ia, "k": ka[ia]}).merge(pd.DataFrame({"tg": ib, "k": kb[ib]}), on="k")
    return pairs[["s1", "tg"]].astype(np.int32)

rules = [
    ("R1_name_key", s1["name_key"].values, tg["name_key"].values, 20, 50, 0),
    ("R2_name_compact", s1["name_compact"].values, tg["name_compact"].values, 20, 50, 0),
    ("R3_addr_key", s1["addr_key"].values, tg["addr_key"].values, 20, 50, 0),
    ("R4_name_stem", s1["name_stem"].values, tg["name_stem"].values, 20, 50, 4),
    ("R5_house_w1", s1["house_w1"].values, tg["house_w1"].values, 20, 50, 4),
    ("R6_house_p3", s1["house_p3"].values, tg["house_p3"].values, 20, 50, 4),
    ("R7_house_p2", s1["house_p2"].values, tg["house_p2"].values, 15, 30, 3),
    ("R8_house_street", s1["house_street"].values, tg["house_street"].values, 15, 30, 4),
    ("R9_w1_w2", s1["w1_w2"].values, tg["w1_w2"].values, 15, 35, 5),
    ("R10_state_w1", s1["state_w1"].values, tg["state_w1"].values, 15, 40, 4),
    ("R11_house_w2", s1["house_w2"].values, tg["house_w2"].values, 15, 30, 4),
    ("R12_w1_name", s1["w1"].values, tg["w1"].values, 15, 30, 3),
    ("R13_house_state", s1["house_state"].values, tg["house_state"].values, 10, 20, 4),
]

cumulative_pairs_set = set()
print(f"\n{'Rule Name':<20} | {'Total Pairs':<12} | {'Dev Pairs':<10} | {'Rule Recall':<12} | {'Cumulative Recall'}")
print("-" * 75)

for rule_name, s1_c, tg_c, max_s1, max_tg, min_len in rules:
    t1 = time.time()
    p = run_pass(s1_c, tg_c, max_s1=max_s1, max_tg=max_tg, min_len=min_len)
    
    dev_p = p[p["s1"].isin(dev_indices_set)].copy()
    dev_p["s1_int"] = s1_ids[dev_p["s1"].values]
    dev_p["tg_int"] = tg_ids[dev_p["tg"].values]
    
    rule_truth_m = dev_truth.merge(dev_p[["s1_int", "tg_int"]].drop_duplicates(), on=["s1_int", "tg_int"])
    rule_recall = len(rule_truth_m) / n_dev_truth
    
    new_pairs = set(zip(dev_p["s1_int"].values, dev_p["tg_int"].values))
    cumulative_pairs_set.update(new_pairs)
    
    cum_df = pd.DataFrame(list(cumulative_pairs_set), columns=["s1_int", "tg_int"])
    cum_m = dev_truth.merge(cum_df, on=["s1_int", "tg_int"])
    cum_recall = len(cum_m) / n_dev_truth
    
    print(f"{rule_name:<20} | {len(p):<12,d} | {len(dev_p):<10,d} | {rule_recall*100:>10.2f}% | {cum_recall*100:>15.2f}%")

print(f"\nFinal Cumulative Recall: {cum_recall*100:.2f}%")
