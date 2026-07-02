"""
demo.py — ШВИДКА частина для самого захисту (<30 c, без навчання).
Вантажить готові моделі й показує живу демонстрацію:
  чистий детектор -> атака PGD валить його -> робастна модель тримається.
Запуск:  python demo.py
"""
import json, time, sys, numpy as np, torch, torch.nn as nn
from sklearn.metrics import recall_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

# ── невеличкі утиліти для «театру» в терміналі ──
G, R, B, Y, C, X = "\033[92m", "\033[91m", "\033[94m", "\033[93m", "\033[96m", "\033[0m"
def head(t): print(f"\n{B}{'─'*64}\n  {t}\n{'─'*64}{X}")
def slow(t, d=0.012):
    for ch in t: sys.stdout.write(ch); sys.stdout.flush(); time.sleep(d)
    print()
def bar(v, col): 
    n = int(round(v*40)); return f"{col}{'█'*n}{X}{'░'*(40-n)} {v*100:5.1f}%"

NF = 111
class MLP(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(nf,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,2))
    def forward(s, x): return s.net(x)

def load(path):
    m = MLP(NF); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    return PyTorchClassifier(model=m, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(m.parameters()), input_shape=(NF,),
        nb_classes=2, clip_values=(0.0,1.0), device_type="cpu")

print(f"{C}Завантаження готових моделей і тестових даних...{X}")
base, rob = load("model_base.pt"), load("model_robust.pt")
Xte, yte = np.load("Xte.npy"), np.load("yte.npy")
res = json.load(open("results.json"))
ph = yte == 1                      # фішингові приклади
Xp, yp = Xte[ph], yte[ph]

head("КРОК 1 · Детектор на чистих даних")
slow("Подаємо реальні фішингові URL на навчений MLP-детектор...")
r_clean = recall_score(yp, np.argmax(base.predict(Xp), 1))
print(f"  Recall фішингу: {bar(r_clean, G)}")
slow(f"{G}  Детектор ловить майже всі фішингові URL. Виглядає надійно.{X}")

head("КРОК 2 · Жива змагальна атака (PGD, білий ящик)")
slow("Той самий детектор. Атакувальник додає дрібні, майже непомітні")
slow("збурення до ознак URL і рахує атаку PGD просто зараз...")
eps = res["eps"]
t = time.time()
adv = ProjectedGradientDescent(base, eps=eps, eps_step=eps/10, max_iter=20, norm=np.inf, verbose=False).generate(Xp)
r_under = recall_score(yp, np.argmax(base.predict(adv), 1))
d = np.abs(adv - Xp); print(f"  (атака порахована за {time.time()-t:.1f} c; макс. зміна ознаки ε = {eps})")
print(f"  Recall фішингу: {bar(r_under, R)}")
slow(f"{R}  Детектор «осліп»: майже весь фішинг проходить повз нього.{X}")
print(f"  {Y}► {r_clean*100:.1f}%  →  {r_under*100:.1f}%   (падіння на {(r_clean-r_under)*100:.0f} в.п.){X}")

head("КРОК 3 · Захищена модель проти тієї самої атаки")
slow("Тепер — модель після adversarial training. Та сама атака PGD...")
adv_r = ProjectedGradientDescent(rob, eps=eps, eps_step=eps/10, max_iter=20, norm=np.inf, verbose=False).generate(Xp)
r_rob = recall_score(yp, np.argmax(rob.predict(adv_r), 1))
print(f"  Recall фішингу під атакою: {bar(r_rob, G)}")
fpr = res["robust_clean"]["fpr"]
print(f"  Ціна стійкості — хибні тривоги (FPR): {bar(fpr, Y)}")
slow(f"{G}  Робастна модель тримає атаку — але платить вищим FPR.{X}")

head("ПІДСУМОК")
print(f"  {'Чистий детектор':<34}{G}{r_clean*100:5.1f}%{X}")
print(f"  {'Базовий під PGD':<34}{R}{r_under*100:5.1f}%{X}   ← незахищений крихкий")
print(f"  {'Робастний під PGD':<34}{G}{r_rob*100:5.1f}%{X}   ← захист відновлює стійкість")
print(f"\n  {C}Графіки: fig_collapse.png (колапс) і fig_sweep.png (ε-sweep — доказ справжньої стійкості).{X}\n")
