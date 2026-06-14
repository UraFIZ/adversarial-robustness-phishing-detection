"""make_figures.py — усі рисунки з results/metrics.json + npz. Реальні дані прогонів."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY, ORANGE = "#3b6ea5", "#c0392b", "#27ae60", "#7f8c8d", "#e67e22"
FIG = "figures"
m = json.load(open("results/metrics.json"))


def fig_baseline():
    b = m["baseline"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    rf = [b["random_forest"][k] * 100 for k in metrics]
    mlp = [b["mlp"][k] * 100 for k in metrics]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4))
    b1 = ax.bar(x - w/2, rf, w, label="Random Forest", color=BLUE)
    b2 = ax.bar(x + w/2, mlp, w, label="MLP", color=ORANGE)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.6,
                    f"{r.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Значення, %"); ax.set_ylim(0, 105)
    ax.set_title("Базова продуктивність детекторів (чисті дані)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.savefig(f"{FIG}/fig1_baseline_metrics.png"); plt.close()


def fig_roc():
    d = np.load("results/roc.npz")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(d["rf_fpr"], d["rf_tpr"], color=BLUE, lw=2,
            label=f"Random Forest (AUC={m['baseline']['random_forest']['roc_auc']:.3f})")
    ax.plot(d["mlp_fpr"], d["mlp_tpr"], color=ORANGE, lw=2,
            label=f"MLP (AUC={m['baseline']['mlp']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-криві детекторів"); ax.legend(loc="lower right")
    ax.grid(alpha=0.3); plt.savefig(f"{FIG}/fig2_roc.png"); plt.close()


def fig_confusion():
    d = np.load("results/confusion.npz")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in [(axes[0], "rf", "Random Forest"), (axes[1], "mlp", "MLP")]:
        cm = d[key]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Легіт.", "Фішинг"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Легіт.", "Фішинг"])
        ax.set_xlabel("Передбачено"); ax.set_ylabel("Справжній клас")
        ax.set_title(title)
    fig.suptitle("Матриці помилок (чисті дані)", y=1.02)
    plt.savefig(f"{FIG}/fig3_confusion.png"); plt.close()


def fig_attack_masks():
    a = m["attacks"]
    masks = ["full", "realistic", "lexical"]
    mask_lbl = ["Необмежена\n(111 ознак)", "Реалістична\n(105 ознак)", "Лексична\n(98 ознак)"]
    clean = m["baseline"]["mlp"]["recall"] * 100
    fgsm = [a["fgsm"][k]["recall"] * 100 for k in masks]
    pgd = [a["pgd"][k]["recall"] * 100 for k in masks]
    x = np.arange(len(masks)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhline(clean, ls="--", color=GREEN, lw=1.5, label=f"Чисті дані ({clean:.1f}%)")
    b1 = ax.bar(x - w/2, fgsm, w, label="FGSM (ε=0.05)", color=ORANGE)
    b2 = ax.bar(x + w/2, pgd, w, label="PGD (ε=0.1)", color=RED)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.8,
                    f"{r.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(mask_lbl)
    ax.set_ylabel("Recall (виявлення фішингу), %"); ax.set_ylim(0, 105)
    ax.set_title("Деградація детектора під атакою vs обмеження реалістичності")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.savefig(f"{FIG}/fig4_attack_masks.png"); plt.close()


def fig_eps_sweep():
    s = m["eps_sweep"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s["eps"], [r*100 for r in s["recall"]], "o-", color=RED, lw=2, ms=7,
            label="Recall під PGD (лексична маска)")
    ax.axhline(m["baseline"]["mlp"]["recall"]*100, ls="--", color=GREEN, lw=1.5,
               label="Recall на чистих даних")
    for xe, ye in zip(s["eps"], s["recall"]):
        ax.text(xe, ye*100+2.5, f"{ye*100:.0f}", ha="center", fontsize=9)
    ax.set_xlabel("Бюджет збурення ε (L∞)"); ax.set_ylabel("Recall, %")
    ax.set_ylim(0, 105)
    ax.set_title("Залежність стійкості від величини збурення")
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(f"{FIG}/fig5_eps_sweep.png"); plt.close()


def fig_defense_tradeoff():
    # точки: (clean_acc, robust_recall під PGD full)
    pts = []
    pts.append(("Без захисту", m["baseline"]["mlp"]["accuracy"]*100,
                m["attacks"]["pgd"]["full"]["recall"]*100, GRAY, "o"))
    ds = m["defense_sweep"]
    # найкраща точка аугментації за robust_recall_full
    bi = int(np.argmax(ds["robust_recall_full"]))
    pts.append((f"Аугментація\n(ε_tr={ds['eps_train'][bi]})",
                ds["clean_acc"][bi]*100, ds["robust_recall_full"][bi]*100, ORANGE, "s"))
    for p in m["defense_madry"]["points"]:
        pts.append((f"Madry AT\n(ε_tr={p['eps_train']})",
                    p["clean"]["accuracy"]*100, p["robust_full"]["recall"]*100, GREEN, "^"))
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for name, cx, cy, col, mk in pts:
        ax.scatter(cx, cy, s=160, color=col, marker=mk, zorder=3, edgecolor="black", lw=0.5)
        ax.annotate(name, (cx, cy), textcoords="offset points", xytext=(10, -4),
                    fontsize=9.5)
    ax.set_xlabel("Точність на чистих даних, %")
    ax.set_ylabel("Robust recall під PGD (ε=0.1), %")
    ax.set_title("Компроміс «чиста точність ↔ стійкість» для методів захисту")
    ax.set_xlim(75, 95); ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    plt.savefig(f"{FIG}/fig6_defense_tradeoff.png"); plt.close()


def fig_taxonomy():
    tax = m["meta"]["taxonomy"]
    cats = ["Контрольовані\n(лексика URL)", "Напівконтрольовані\n(інфраструктура)",
            "Неконтрольовані\n(треті сторони/час)"]
    vals = [tax["controllable"], tax["semi_controllable"], tax["uncontrollable"]]
    cols = [GREEN, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.barh(cats, vals, color=cols)
    for r in bars:
        ax.text(r.get_width()+1, r.get_y()+r.get_height()/2,
                f"{int(r.get_width())}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Кількість ознак"); ax.set_xlim(0, 110)
    ax.set_title("Таксономія 111 ознак за контролем атакувальника (модель загроз)")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.savefig(f"{FIG}/fig7_threat_taxonomy.png"); plt.close()


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.4)); ax.axis("off")
    stages = [("Датасет\nVrbancic\n58 645×111", BLUE),
              ("Детектори\nRF · MLP", BLUE),
              ("Атака ухилення\nFGSM · PGD\n(маски)", RED),
              ("Захист\nadversarial\ntraining", GREEN),
              ("Оцінка\nповні метрики", GRAY)]
    x = 0.02; w = 0.16; gap = (1 - len(stages)*w) / (len(stages)-1) - 0.005
    centers = []
    for txt, col in stages:
        box = FancyBboxPatch((x, 0.25), w, 0.5, boxstyle="round,pad=0.012",
                             ec=col, fc="white", lw=2)
        ax.add_patch(box)
        ax.text(x+w/2, 0.5, txt, ha="center", va="center", fontsize=9.5, color=col)
        centers.append(x+w/2); x += w + gap
    for i in range(len(centers)-1):
        ax.add_patch(FancyArrowPatch((centers[i]+w/2-0.005, 0.5),
                                     (centers[i+1]-w/2+0.005, 0.5),
                                     arrowstyle="-|>", mutation_scale=16, color="black", lw=1.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Експериментальний конвеєр дослідження", fontsize=12)
    plt.savefig(f"{FIG}/fig8_pipeline.png"); plt.close()


if __name__ == "__main__":
    fig_baseline(); fig_roc(); fig_confusion(); fig_attack_masks()
    fig_eps_sweep(); fig_defense_tradeoff(); fig_taxonomy(); fig_pipeline()
    import os
    print("Рисунки створено:")
    for f in sorted(os.listdir(FIG)):
        print("  figures/" + f)
