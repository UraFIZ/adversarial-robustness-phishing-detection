"""train_defended.py — тренує Madry-модель (eps_train=0.1, 15 епох) і зберігає ваги
для подальшої діагностики gradient masking. Sanity-check: clean acc має ≈ 0.7957."""
import sys, time
import torch
sys.path.insert(0, "src")
import common as C
import threat_model as tm
from art.defences.trainer import AdversarialTrainerMadryPGD

DATA = "data/dataset_small.csv"
SAVE = "results/defended_eps0.1.pt"
ETR = 0.10

C.set_seed()
Xtr, Xte, ytr, yte, names = C.load_data(DATA)
t0 = time.time()

print(f"Madry training eps_train={ETR}, 15 epochs ...", flush=True)
C.set_seed()
model = C.MLP(Xtr.shape[1])
clf = C.make_art_classifier(model)
tr = AdversarialTrainerMadryPGD(clf, nb_epochs=15, batch_size=256,
                                eps=ETR, eps_step=ETR / 5, max_iter=7)
tr.fit(Xtr, ytr)

# sanity-check на чистих даних
lab_c, _ = C.predict_labels_scores(clf, Xte, is_torch=True)
acc = (lab_c == yte).mean()
torch.save(model.state_dict(), SAVE)
print(f"Збережено {SAVE} | clean acc={acc:.4f} (очікувано ≈0.7957) | {time.time()-t0:.0f}s", flush=True)
