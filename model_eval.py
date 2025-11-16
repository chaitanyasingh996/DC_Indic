from collections import Counter
import matplotlib.pyplot as plt
import json


input_path = "ground_truth.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ground_truth = {}

# make ground_truth dictionary
for image_key, entry in data.items():
    annotations = entry.get("annotations", {})
    lang_dict = {}

    for poly_key, poly in annotations.items():
        lang = poly.get("script_language", "").strip()
        text = poly.get("text", "").strip()

        if lang not in lang_dict:
            lang_dict[lang] = []

        lang_dict[lang].append(text)

    ground_truth[image_key] = lang_dict

# ----------------------------------------------------------------------------- #

input_path = "model_output.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model_output = {}

# make model_output dictionary
for image_key, entry in data.items():
    annotations = entry.get("annotations", {})
    lang_dict = {}

    for poly_key, poly in annotations.items():
        lang = poly.get("script_language", "").strip()
        text = poly.get("text", "").strip()

        if lang not in lang_dict:
            lang_dict[lang] = []

        lang_dict[lang].append(text)

    model_output[image_key] = lang_dict


# ---------------------------------------------------------
# Calculating Word Recognition Rate
# ---------------------------------------------------------

# case-sensitive wrr calculation
def compute_wrr(gt_list, pred_list):
    gt_counter = Counter(gt_list)
    pred_counter = Counter(pred_list)

    correct = 0

    for word, gt_count in gt_counter.items():
        pred_count = pred_counter.get(word, 0)
        correct += min(gt_count, pred_count)

    total = len(gt_list)
    return correct / total if total > 0 else 0.0, correct, total

# case-insensitive wrr calculation
def compute_wrr_case_insensitive(gt_list, pred_list):
    # convert all words to lowercase for case-insensitive comparison
    gt_list_lower = [w.lower() for w in gt_list]
    pred_list_lower = [w.lower() for w in pred_list]

    gt_counter = Counter(gt_list_lower)
    pred_counter = Counter(pred_list_lower)

    correct = 0

    for word, gt_count in gt_counter.items():
        pred_count = pred_counter.get(word, 0)
        correct += min(gt_count, pred_count)

    total = len(gt_list)
    return correct / total if total > 0 else 0.0, correct, total

# Running totals (for computing overall average WRR later)
e_wrr = 0          # English WRR (case-sensitive) cumulative sum
h_wrr = 0          # Hindi WRR (case-sensitive) cumulative sum
e_wrr_ci = 0       # English WRR (case-insensitive) cumulative sum
h_wrr_ci = 0       # Hindi WRR (case-insensitive) cumulative sum

# Per-image WRR lists (for plotting per-image performance curves)
english_wrr_list = []       # English WRR (case-sensitive) for each image
hindi_wrr_list = []         # Hindi WRR (case-sensitive) for each image

english_wrr_ci_list = []    # English WRR (case-insensitive) for each image
hindi_wrr_ci_list = []      # Hindi WRR (case-insensitive) for each image

img_ids = []                # Stores the dataset index (image number) for plotting on x-axis


for i in range(1,62):
    if i == 34:
        continue
    image_path = "dc_" + str(i);
    if i >= 51:
        image_path = image_path + ".png"
    else:
        image_path = image_path + ".jpg"

    d1 = ground_truth[image_path] 
    d2 = model_output[image_path] 

    # Combine Marathi → Hindi
    if 'marathi' in d2:
        d2.setdefault('hindi', []).extend(d2['marathi'])


    # Calculate WRR for Hindi and English
    hindi_wrr, _, _ = compute_wrr(
        d1.get('hindi', []),
        d2.get('hindi', [])
    )

    english_wrr, _, _ = compute_wrr(
        d1.get('english', []),
        d2.get('english', [])
    )

    hindi_wrr_ci, _, _ = compute_wrr_case_insensitive(
        d1.get('hindi', []),
        d2.get('hindi', [])
    )

    english_wrr_ci, _, _ = compute_wrr_case_insensitive(
        d1.get('english', []),
        d2.get('english', [])
    )

    e_wrr = e_wrr + english_wrr
    h_wrr = h_wrr + hindi_wrr
    e_wrr_ci = e_wrr_ci + english_wrr_ci
    h_wrr_ci = h_wrr_ci + hindi_wrr_ci


    english_wrr_list.append(english_wrr)
    hindi_wrr_list.append(hindi_wrr)

    english_wrr_ci_list.append(english_wrr_ci)
    hindi_wrr_ci_list.append(hindi_wrr_ci)

    img_ids.append(i)


print("Case Sensitive in English")
print(f"Hindi : {h_wrr/60 * 100:.2f}%")
print(f"English : {e_wrr/60 * 100:.2f}%")
print(f"Case insensitive in English")
print(f"Hindi : {h_wrr_ci/60 * 100:.2f}%")
print(f"English : {e_wrr_ci/60 * 100:.2f}%")




# ---------------------------------------------------------
# Precision / Recall / F1 (Case-Sensitive)
# ---------------------------------------------------------
def compute_prf(gt_list, pred_list):
    gt_counter = Counter(gt_list)
    pred_counter = Counter(pred_list)

    TP = sum(min(gt_counter[w], pred_counter.get(w, 0)) for w in gt_counter)
    FP = sum(max(pred_counter[w] - gt_counter.get(w, 0), 0) for w in pred_counter)
    FN = sum(max(gt_counter[w] - pred_counter.get(w, 0), 0) for w in gt_counter)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


# ---------------------------------------------------------
# Precision / Recall / F1 (Case-Insensitive)
# ---------------------------------------------------------
def compute_prf_ci(gt_list, pred_list):
    gt_lower = [w.lower() for w in gt_list]
    pred_lower = [w.lower() for w in pred_list]

    return compute_prf(gt_lower, pred_lower)


img_ids = []

# Hindi — Case Sensitive
h_p, h_r, h_f = [], [], []
# Hindi — Case Insensitive
h_p_ci, h_r_ci, h_f_ci = [], [], []

# English — Case Sensitive
e_p, e_r, e_f = [], [], []
# English — Case Insensitive
e_p_ci, e_r_ci, e_f_ci = [], [], []


for i in range(1, 62):
    if i == 34:
        continue

    image_path = f"dc_{i}.png" if i >= 51 else f"dc_{i}.jpg"
    img_ids.append(i)

    d1 = ground_truth[image_path]
    d2 = model_output[image_path]

    # Combine Marathi → Hindi
    if "marathi" in d2:
        d2.setdefault("hindi", []).extend(d2["marathi"])

    # -------- Hindi Case-Sensitive --------
    hp, hr, hf = compute_prf(
        d1.get("hindi", []),
        d2.get("hindi", [])
    )
    h_p.append(hp)
    h_r.append(hr)
    h_f.append(hf)

    # -------- Hindi Case-Insensitive --------
    hpci, hrci, hfci = compute_prf_ci(
        d1.get("hindi", []),
        d2.get("hindi", [])
    )
    h_p_ci.append(hpci)
    h_r_ci.append(hrci)
    h_f_ci.append(hfci)

    # -------- English Case-Sensitive --------
    ep, er, ef = compute_prf(
        d1.get("english", []),
        d2.get("english", [])
    )
    e_p.append(ep)
    e_r.append(er)
    e_f.append(ef)

    # -------- English Case-Insensitive --------
    epci, erci, efci = compute_prf_ci(
        d1.get("english", []),
        d2.get("english", [])
    )
    e_p_ci.append(epci)
    e_r_ci.append(erci)
    e_f_ci.append(efci)



# ---------------------------------------------------------
# Plot helper function
# ---------------------------------------------------------

def plot_combined(x, 
                  p1, r1, f1, title1,
                  p2, r2, f2, title2):

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ---- Subplot 1 ----
    axs[0].plot(x, p1, marker='o', label='Precision')
    axs[0].plot(x, r1, marker='s', label='Recall')
    axs[0].plot(x, f1, marker='^', label='F1-Score')
    axs[0].set_title(title1)
    axs[0].set_ylabel("Metric Value")
    axs[0].grid(True)
    axs[0].legend()

    # ---- Subplot 2 ----
    axs[1].plot(x, p2, marker='o', label='Precision')
    axs[1].plot(x, r2, marker='s', label='Recall')
    axs[1].plot(x, f2, marker='^', label='F1-Score')
    axs[1].set_title(title2)
    axs[1].set_xlabel("Dataset Image Index")
    axs[1].set_ylabel("Metric Value")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_wrr(img_ids, wrr1, title1, wrr2, title2):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ---- Subplot 1 ----
    axs[0].plot(img_ids, wrr1, marker='o', label='WRR')
    axs[0].set_title(title1)
    axs[0].set_ylabel("WRR Value")
    axs[0].grid(True)

    # ---- Subplot 2 ----
    axs[1].plot(img_ids, wrr2, marker='o', label='WRR')
    axs[1].set_title(title2)
    axs[1].set_xlabel("Dataset Image Index")
    axs[1].set_ylabel("WRR Value")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()



# -------------------------------------------------------------
# Hindi Window: Case Sensitive + Case Insensitive
# -------------------------------------------------------------
plot_combined(
    img_ids,
    h_p, h_r, h_f, "Hindi — Case Sensitive",
    h_p_ci, h_r_ci, h_f_ci, "Hindi — Case Insensitive"
)

plot_wrr(
    img_ids,
    hindi_wrr_list,
    "Hindi WRR — Case Sensitive",
    hindi_wrr_ci_list,
    "Hindi WRR — Case Insensitive"
)


# -------------------------------------------------------------
# English Window: Case Sensitive + Case Insensitive
# -------------------------------------------------------------
plot_combined(
    img_ids,
    e_p, e_r, e_f, "English — Case Sensitive",
    e_p_ci, e_r_ci, e_f_ci, "English — Case Insensitive"
)

plot_wrr(
    img_ids,
    english_wrr_list,
    "English WRR — Case Sensitive",
    english_wrr_ci_list,
    "English WRR — Case Insensitive"
)