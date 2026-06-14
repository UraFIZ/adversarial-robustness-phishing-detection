"""madry_one.py EPS_TRAIN NB_EPOCHS — тренує одну Madry-модель, оцінює, додає точку."""
import sys, json, os, time
import numpy as np
sys.path.insert(0, "src")
import common as C
import threat_model as tm
from art.defences.trainer import AdversarialTrainerMadryPGD

DATA = "data/dataset_small.csv"
OUT = "results/defense_madry.json"

etr = float(sys.argv[1])
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 15

C.set_seed()
Xtr, Xte, ytr, yte, names = C.load_data(DATA)
masks = tm.build_masks(names)
t0 = time.time()

print(f"Madry training eps_train={etr}, epochs={nb}", flush=True)
C.set_seed()
clf = C.make_art_classifier(C.MLP(Xtr.shape[1]))
tr = AdversarialTrainerMadryPGD(clf, nb_epochs=nb, batch_size=256,
                                eps=etr, eps_step=etr / 5, max_iter=7)
tr.fit(Xtr, ytr)

lab_c, sc_c = C.predict_labels_scores(clf, Xte, is_torch=True)
mc = C.full_metrics(yte, lab_c, sc_c)
Xev_full = C.run_attack(clf, Xte, "pgd", eps=0.1, mask=masks["full"])
Xev_lex = C.run_attack(clf, Xte, "pgd", eps=0.1, mask=masks["lexical"])
rf_full = C.full_metrics(yte, C.predict_labels_scores(clf, Xev_full)[0])
rf_lex = C.full_metrics(yte, C.predict_labels_scores(clf, Xev_lex)[0])

point = {"eps_train": etr, "nb_epochs": nb, "clean": mc,
         "robust_full": rf_full, "robust_lexical": rf_lex,
         "sec": round(time.time() - t0, 1)}

if os.path.exists(OUT):
    with open(OUT) as f:
        res = json.load(f)
else:
    res = {"method": "madry_pgd_adversarial_training",
           "eval_attack": "PGD eps=0.1, max_iter=20", "points": []}
# replace if same eps already present
res["points"] = [p for p in res["points"] if p["eps_train"] != etr]
res["points"].append(point)
res["points"].sort(key=lambda p: p["eps_train"])
with open(OUT, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("clean acc=%.3f recall=%.3f | robust_recall full=%.3f lex=%.3f | %.0fs" %
      (mc["accuracy"], mc["recall"], rf_full["recall"], rf_lex["recall"], point["sec"]), flush=True)
