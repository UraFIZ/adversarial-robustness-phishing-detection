"""
run_experiments.py — повний експериментальний прогін (Блок A).

Експерименти:
  1. Базові детектори (чисті дані): RF + MLP, повні метрики, ROC, матриці помилок.
  2. Атака vs обмеження реалістичності: FGSM/PGD під масками {full, realistic, lexical}.
  3. Sweep по бюджету збурення: PGD(lexical), eps у діапазоні — recall(eps).
  4. Валідність URL: дискретизація adversarial-прикладів (округлення до цілих) і ре-оцінка.
  5. Захист: adversarial training, sweep по eps_train — компроміс «чиста точність ↔ стійкість».

Усе детерміноване (SEED=42). Результати → results/*.json|*.npz.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
import common as C
import threat_model as tm
from sklearn.metrics import roc_curve, recall_score, accuracy_score

DATA = "data/dataset_small.csv"
RES = "results"
t0 = time.time()


def scaler_params(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(np.float64)
    mn, mx = X.min(0), X.max(0)
    rng = np.where((mx - mn) == 0, 1, (mx - mn))
    # ознака цілочисельна, якщо всі вихідні значення цілі
    is_int = np.all(np.equal(np.mod(X, 1), 0), axis=0)
    return mn, mx, rng, is_int


def denorm(Xn, mn, rng):
    return Xn * rng + mn


def renorm(Xr, mn, rng):
    return ((Xr - mn) / rng).astype(np.float32)


def discretize(Xn_adv, mn, mx, rng, is_int):
    """Округлення цілочисельних ознак до валідних значень + кліп у [min,max]."""
    Xr = denorm(Xn_adv.astype(np.float64), mn, rng)
    Xr[:, is_int] = np.round(Xr[:, is_int])
    Xr = np.clip(Xr, mn, mx)
    return renorm(Xr, mn, rng)


def main():
    C.set_seed()
    Xtr, Xte, ytr, yte, names = C.load_data(DATA)
    mn, mx, rng, is_int = scaler_params(DATA)
    masks = tm.build_masks(names)
    out = {"meta": {"seed": C.SEED, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                    "n_features": len(names), "int_features": int(is_int.sum()),
                    "taxonomy": tm.SUMMARY}}

    # ---------- Exp 1: базові детектори ----------
    print("[1] Базові детектори...", flush=True)
    rf = C.train_rf(Xtr, ytr)
    rf_lab, rf_sc = C.predict_labels_scores(rf, Xte, is_torch=False)
    mlp = C.train_mlp(Xtr, ytr, epochs=15)
    mlp_lab, mlp_sc = C.predict_labels_scores(mlp, Xte, is_torch=True)
    out["baseline"] = {
        "random_forest": C.full_metrics(yte, rf_lab, rf_sc),
        "mlp": C.full_metrics(yte, mlp_lab, mlp_sc),
    }
    # ROC + матриці помилок
    rf_fpr, rf_tpr, _ = roc_curve(yte, rf_sc)
    mlp_fpr, mlp_tpr, _ = roc_curve(yte, mlp_sc)
    np.savez(f"{RES}/roc.npz", rf_fpr=rf_fpr, rf_tpr=rf_tpr,
             mlp_fpr=mlp_fpr, mlp_tpr=mlp_tpr)
    b = out["baseline"]
    np.savez(f"{RES}/confusion.npz",
             rf=np.array([[b['random_forest']['tn'], b['random_forest']['fp']],
                          [b['random_forest']['fn'], b['random_forest']['tp']]]),
             mlp=np.array([[b['mlp']['tn'], b['mlp']['fp']],
                           [b['mlp']['fn'], b['mlp']['tp']]]))
    print("    RF acc=%.3f | MLP acc=%.3f recall=%.3f" %
          (b['random_forest']['accuracy'], b['mlp']['accuracy'], b['mlp']['recall']), flush=True)

    # ---------- Exp 2: атака vs обмеження ----------
    print("[2] Атаки під масками {full, realistic, lexical}...", flush=True)
    out["attacks"] = {}
    for atk, eps in [("fgsm", 0.05), ("pgd", 0.1)]:
        out["attacks"][atk] = {"eps": eps}
        for mk in ["full", "realistic", "lexical"]:
            Xadv = C.run_attack(mlp, Xte, atk, eps=eps, mask=masks[mk])
            lab, sc = C.predict_labels_scores(mlp, Xadv, is_torch=True)
            out["attacks"][atk][mk] = C.full_metrics(yte, lab, sc)
            print("    %s/%s: acc=%.3f recall=%.3f" %
                  (atk, mk, out["attacks"][atk][mk]["accuracy"],
                   out["attacks"][atk][mk]["recall"]), flush=True)

    # ---------- Exp 3: sweep по бюджету збурення ----------
    print("[3] Sweep eps (PGD, lexical)...", flush=True)
    eps_grid = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    sweep = {"eps": eps_grid, "recall": [], "accuracy": []}
    for e in eps_grid:
        Xadv = C.run_attack(mlp, Xte, "pgd", eps=e, eps_step=max(e/10, 0.005),
                            max_iter=20, mask=masks["lexical"])
        lab, _ = C.predict_labels_scores(mlp, Xadv, is_torch=True)
        sweep["recall"].append(float(recall_score(yte, lab, pos_label=1)))
        sweep["accuracy"].append(float(accuracy_score(yte, lab)))
        print("    eps=%.2f -> recall=%.3f" % (e, sweep["recall"][-1]), flush=True)
    out["eps_sweep"] = sweep

    # ---------- Exp 4: валідність URL (дискретизація) ----------
    print("[4] Дискретизація (валідні URL)...", flush=True)
    Xadv_cont = C.run_attack(mlp, Xte, "pgd", eps=0.1, mask=masks["lexical"])
    lab_cont, _ = C.predict_labels_scores(mlp, Xadv_cont, is_torch=True)
    Xadv_disc = discretize(Xadv_cont, mn, mx, rng, is_int)
    lab_disc, _ = C.predict_labels_scores(mlp, Xadv_disc, is_torch=True)
    out["validity"] = {
        "continuous": C.full_metrics(yte, lab_cont),
        "discretized": C.full_metrics(yte, lab_disc),
    }
    print("    continuous recall=%.3f | discretized recall=%.3f" %
          (out["validity"]["continuous"]["recall"],
           out["validity"]["discretized"]["recall"]), flush=True)

    # ---------- Exp 5: захист (adversarial training) + sweep ----------
    print("[5] Adversarial training sweep...", flush=True)
    defense = {"eps_train": [], "clean_acc": [], "clean_recall": [],
               "robust_recall_full": [], "robust_recall_lexical": []}
    for etr in [0.05, 0.10, 0.20]:
        C.set_seed()
        rob = C.train_mlp(Xtr, ytr, epochs=10)  # базове тренування
        # генеруємо adv-приклади на train із заданим eps_train (lexical-обмежені)
        Xadv_tr = C.run_attack(rob, Xtr, "pgd", eps=etr, eps_step=max(etr/10, 0.005),
                               max_iter=10, mask=masks["lexical"])
        Xmix = np.vstack([Xtr, Xadv_tr]).astype(np.float32)
        ymix = np.concatenate([ytr, ytr])
        rob.fit(Xmix, ymix, nb_epochs=8, batch_size=256, verbose=False)
        # оцінка
        lab_c, _ = C.predict_labels_scores(rob, Xte, is_torch=True)
        mc = C.full_metrics(yte, lab_c)
        Xev_full = C.run_attack(rob, Xte, "pgd", eps=0.1, mask=masks["full"])
        Xev_lex = C.run_attack(rob, Xte, "pgd", eps=0.1, mask=masks["lexical"])
        rr_full = float(recall_score(yte, C.predict_labels_scores(rob, Xev_full)[0], pos_label=1))
        rr_lex = float(recall_score(yte, C.predict_labels_scores(rob, Xev_lex)[0], pos_label=1))
        defense["eps_train"].append(etr)
        defense["clean_acc"].append(mc["accuracy"])
        defense["clean_recall"].append(mc["recall"])
        defense["robust_recall_full"].append(rr_full)
        defense["robust_recall_lexical"].append(rr_lex)
        print("    eps_train=%.2f -> clean_acc=%.3f robust_recall(full)=%.3f (lex)=%.3f" %
              (etr, mc["accuracy"], rr_full, rr_lex), flush=True)
    out["defense_sweep"] = defense

    out["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    with open(f"{RES}/metrics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nГотово за %.1f c. Результати у %s/metrics.json" %
          (out["meta"]["runtime_sec"], RES), flush=True)


if __name__ == "__main__":
    main()
