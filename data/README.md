# Adversarial Robustness of ML-Based Phishing Detection

Дослідницький код до магістерської дипломної роботи (MSc Computer Science /
Cybersecurity, Neoversity): **стійкість ML-систем виявлення фішингу до змагальних
(adversarial) атак** на рівні URL.

Робота емпірично оцінює, наскільки детектори фішингу на основі машинного навчання
залишаються надійними під атаками ухилення, і робить акцент на **реалістичності
атак** — які ознаки атакувальник реально здатен контролювати.

---

## Ключові результати

| Етап | Метрика | Значення |
|---|---|---|
| Random Forest (чисті дані) | Accuracy | **95.5%** |
| MLP (чисті дані) | Accuracy / Recall | **91.7% / 94.5%** |
| MLP під PGD (ε=0.1, необмежена) | Recall | **5.6%** |
| MLP під PGD (ε=0.1, лише контрольовані ознаки) | Recall | **5.7%** |
| Захист: наївна аугментація | Robust recall / Clean acc | ~12% / ~90% |
| Захист: Madry adversarial training (ε=0.05) | Robust recall / Clean acc | **92.9% / 82.5%** |
| Захист: Madry adversarial training (ε=0.1) | Robust recall / Clean acc | **97.6% / 79.6%** |

**Основні висновки:**

1. Обмеження атаки лише реалістично контрольованими ознаками URL майже не
   послаблює її — вразливість є **практично реалізовною**, а не артефактом
   необмеженого збурення.
2. Достатньо невеликого збурення: вже за ε=0.02 recall падає з 94.5% до ~27%.
3. Округлення збурень до валідних цілих значень (валідність URL) не відновлює
   виявлення — атака переживає дискретизацію.
4. Канонічне adversarial training (Madry) відновлює стійкість до 93–98% ціною
   чистої точності — класичний компроміс «точність ↔ стійкість». Проста
   аугментація для цього непридатна.

---

## Модель загроз (таксономія ознак)

Усі 111 ознак датасету класифіковано за рівнем контролю атакувальника
(див. `src/threat_model.py`):

* **Контрольовані — 98** (лексика URL: символи, довжини, структура);
* **Напівконтрольовані — 7** (інфраструктура: SPF, NS, MX, TTL, TLS, редиректи, термін реєстрації);
* **Неконтрольовані — 6** (вік домену, індексація Google, час відповіді, ASN, к-сть IP).

---

## Відтворюваність

* **Seed = 42** скрізь (numpy, torch, спліти).
* Точні версії бібліотек — у `requirements.txt` (Python 3.12).
* Датасет завантажується скриптом, не зберігається в репозиторії.

### Запуск

```bash
# 1. Залежності
pip install -r requirements.txt

# 2. Датасет (~16 МБ) у data/
python scripts/download_data.py

# 3. Основні експерименти (базові детектори, атаки, sweep, валідність, аугментація)
python src/run_experiments.py            # -> results/metrics.json

# 4. Канонічне adversarial training (по одній моделі за прогін)
python src/madry_one.py 0.05 15          # -> results/defense_madry.json
python src/madry_one.py 0.10 15

# 5. Усі рисунки
python src/make_figures.py               # -> figures/*.png
```

---

## Структура репозиторію

```
.
├── src/
│   ├── threat_model.py       # таксономія ознак (модель загроз) + маски збурення
│   ├── common.py             # дані, моделі (RF, MLP), метрики, атаки з масками
│   ├── run_experiments.py    # експерименти 1–5 -> results/metrics.json
│   ├── madry_one.py          # канонічне Madry adversarial training (1 модель/прогін)
│   └── make_figures.py       # генерація всіх рисунків
├── scripts/
│   └── download_data.py      # завантаження датасету Vrbancic et al.
├── figures/                  # 8 рисунків (PNG) з реальних прогонів
├── results/                  # metrics.json, defense_madry.json
├── data/                     # датасет (gitignored)
├── requirements.txt
└── README.md
```

---

## Дані

Датасет фішингових URL: **Vrbancic et al.**, ~58 645 зразків, 111 ознак.
Джерело: [GregaVrbancic/Phishing-Dataset](https://github.com/GregaVrbancic/Phishing-Dataset)
(`dataset_small.csv`).

## Стек

Python · scikit-learn · PyTorch · Adversarial Robustness Toolbox (ART) · matplotlib

## Академічна доброчесність

Код підготовлено в межах магістерської роботи. Використання AI-інструментів
задокументовано відповідно до вимог академічної доброчесності Neoversity.
