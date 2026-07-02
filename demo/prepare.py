"""
prepare.py — ВАЖКА частина демо. Запусти ОДИН раз удома до захисту.
Навчає базовий і робастний детектори, проводить атаки, зберігає моделі,
результати (results.json) і графіки. На самому захисті це не запускається —
там працює швидкий demo.py, що лише вантажить готове.
"""
import json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cpu"
t0 = time.time()

# ---------- дані ----------
df = pd.read_csv("dataset_small.csv")
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
sc = MinMaxScaler().fit(Xtr)                      # нормалізація ознак у [0,1]
Xtr = sc.transform(Xtr).astype(np.float32)
Xte = sc.transform(Xte).astype(np.float32)
NF = X.shape[1]
print(f"ознак: {NF} | train {Xtr.shape[0]} | test {Xte.shape[0]}")

# ---------- модель: MLP 111 -> 128 -> 64 -> 2 ----------
class MLP(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(nf, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2))
    def forward(s, x): return s.net(x)

def make_clf(model):
    return PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(NF,), nb_classes=2, clip_values=(0.0, 1.0), device_type=DEV)

def metrics(clf, Xx, yy):
    pred = np.argmax(clf.predict(Xx), axis=1)
    rec = recall_score(yy, pred)                 # recall по класу "фішинг" (=1)
    acc = accuracy_score(yy, pred)
    tn, fp, fn, tp = confusion_matrix(yy, pred).ravel()
    fpr = fp / (fp + tn)                          # хибні тривоги на легітимних
    return dict(acc=acc, recall=rec, fpr=fpr)

# ---------- 1) базовий детектор ----------
print("\n[1] навчаю базовий детектор...")
base = MLP(NF)
clf = make_clf(base)
clf.fit(Xtr, ytr, nb_epochs=15, batch_size=256, verbose=False)
clean = metrics(clf, Xte, yte)
print("    чистий:", {k: round(v, 4) for k, v in clean.items()})

# ---------- 2) атака PGD (білий ящик, L-inf) ----------
EPS = 0.20
print(f"\n[2] атака PGD (eps={EPS})...")
pgd = ProjectedGradientDescent(clf, eps=EPS, eps_step=EPS/10, max_iter=20, norm=np.inf, verbose=False)
mask_phish = yte == 1                              # атакуємо фішингові приклади (ціль — щоб їх пропустили)
Xadv = pgd.generate(Xte[mask_phish])
under = metrics_under = recall_score(
    yte[mask_phish], np.argmax(clf.predict(Xadv), axis=1))
print(f"    recall фішингу: {clean['recall']:.4f} -> {under:.4f}")

# ---------- 3) adversarial training (Madry) ----------
print("\n[3] adversarial training (робастна модель)...")
rob = MLP(NF)
rclf = make_clf(rob)
rpgd = ProjectedGradientDescent(rclf, eps=EPS, eps_step=EPS/10, max_iter=7, norm=np.inf, verbose=False)
EPOCHS = 12; BS = 256
idx = np.arange(Xtr.shape[0])
for ep in range(EPOCHS):
    np.random.shuffle(idx)
    for b in range(0, len(idx), BS):
        bi = idx[b:b+BS]
        xb, yb = Xtr[bi], ytr[bi]
        xadv = rpgd.generate(xb)                   # половина чистих + adversarial
        xmix = np.concatenate([xb, xadv]); ymix = np.concatenate([yb, yb])
        rclf.fit(xmix, ymix, nb_epochs=1, batch_size=BS, verbose=False)
    if ep % 4 == 0: print(f"    епоха {ep+1}/{EPOCHS}")
rob_clean = metrics(rclf, Xte, yte)
Xadv_r = rpgd.generate(Xte[mask_phish])
rob_strong = ProjectedGradientDescent(rclf, eps=EPS, eps_step=EPS/10, max_iter=20, norm=np.inf, verbose=False).generate(Xte[mask_phish])
rob_under = recall_score(yte[mask_phish], np.argmax(rclf.predict(rob_strong), axis=1))
print("    робастна, чистий:", {k: round(v, 4) for k, v in rob_clean.items()})
print(f"    робастна, recall під атакою: {rob_under:.4f}")

# ---------- 4) eps-sweep (доказ справжньої стійкості) ----------
print("\n[4] eps-sweep...")
sweep_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
base_curve, rob_curve = [], []
Xp = Xte[mask_phish]; yp = yte[mask_phish]
for e in sweep_eps:
    if e == 0:
        base_curve.append(recall_score(yp, np.argmax(clf.predict(Xp), 1)))
        rob_curve.append(recall_score(yp, np.argmax(rclf.predict(Xp), 1)))
    else:
        a1 = ProjectedGradientDescent(clf, eps=e, eps_step=e/10, max_iter=20, norm=np.inf, verbose=False).generate(Xp)
        a2 = ProjectedGradientDescent(rclf, eps=e, eps_step=e/10, max_iter=20, norm=np.inf, verbose=False).generate(Xp)
        base_curve.append(recall_score(yp, np.argmax(clf.predict(a1), 1)))
        rob_curve.append(recall_score(yp, np.argmax(rclf.predict(a2), 1)))
print("    base:", [round(v,3) for v in base_curve])
print("    rob :", [round(v,3) for v in rob_curve])

# ---------- зберегти все ----------
torch.save(base.state_dict(), "model_base.pt")
torch.save(rob.state_dict(), "model_robust.pt")
np.save("scaler_min.npy", sc.data_min_); np.save("scaler_max.npy", sc.data_max_)
np.save("Xte.npy", Xte); np.save("yte.npy", yte)
results = dict(
    n_features=int(NF), eps=EPS, seed=SEED,
    clean=clean, under_attack_recall=float(under),
    robust_clean=rob_clean, robust_under_recall=float(rob_under),
    sweep_eps=sweep_eps, base_curve=[float(v) for v in base_curve],
    rob_curve=[float(v) for v in rob_curve])
json.dump(results, open("results.json", "w"), indent=2)

# графік 1: колапс і відновлення
fig, ax = plt.subplots(figsize=(7,4.2))
labels = ["Чистий\n(без атаки)", "Базовий\nпід PGD", "Робастний\nпід PGD"]
vals = [clean["recall"], under, rob_under]
cols = ["#13A085", "#D7263D", "#1C3566"]
bars = ax.bar(labels, [v*100 for v in vals], color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v*100+2, f"{v*100:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Recall фішингу, %"); ax.set_ylim(0, 105)
ax.set_title("Колапс детектора під атакою та відновлення захистом", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_collapse.png", dpi=130); plt.close()

# графік 2: eps-sweep
fig, ax = plt.subplots(figsize=(7,4.2))
ax.plot(sweep_eps, [v*100 for v in base_curve], "o-", color="#D7263D", lw=2.2, label="Базовий")
ax.plot(sweep_eps, [v*100 for v in rob_curve], "s-", color="#13A085", lw=2.2, label="Робастний (adv. training)")
ax.set_xlabel("ε — бюджет атаки (L∞)"); ax.set_ylabel("Recall фішингу, %")
ax.set_title("ε-sweep: плавна деградація = справжня стійкість", fontweight="bold")
ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.25)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_sweep.png", dpi=130); plt.close()

print(f"\nГОТОВО за {time.time()-t0:.0f} c. Збережено: model_base.pt, model_robust.pt, results.json, fig_collapse.png, fig_sweep.png")
