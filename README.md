```
███╗   ██╗██╗██████╗ ██╗   ██╗ █████╗ ███╗   ██╗ █████╗
████╗  ██║██║██╔══██╗██║   ██║██╔══██╗████╗  ██║██╔══██╗
██╔██╗ ██║██║██████╔╝██║   ██║███████║██╔██╗ ██║███████║
██║╚██╗██║██║██╔══██╗╚██╗ ██╔╝██╔══██║██║╚██╗██║██╔══██║
██║ ╚████║██║██║  ██║ ╚████╔╝ ██║  ██║██║ ╚████║██║  ██║
╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
```
> *local ai-threat detector* — watch the wire. kill the noise.

---

## what is this?

**NIRVANA** is a self-hosted, ML-powered network intrusion detection system. It watches your traffic, classifies threats in real time, and keeps a log of everything trying to get through. No cloud. No telemetry. Just you, your models, and your network.

It runs locally via a FastAPI backend, uses JWT for auth, and ships with a CLI client and admin dashboard so you don't need to touch a browser unless you want to.

---

## what it does

| capability | description |
|---|---|
| 🔐 **auth** | JWT-based login with bcrypt hashing. stateless, fast, secure. |
| 👤 **user roles** | admin and user tiers with separate access controls |
| 🧠 **threat detection** | ML ensemble — XGBoost, RandomForest, ExtraTreesClassifier |
| 📊 **smart preprocessing** | FCBF feature selection + SMOTE resampling for class imbalance |
| 💬 **LLM integration** | Ollama hook for natural language threat analysis |
| 🖥️ **admin dashboard** | CLI-based admin panel for monitoring and control |
| 📡 **REST API** | clean endpoints for prediction, upload, and threat retrieval |

---

## stack

```
backend     →  FastAPI + Uvicorn
auth        →  JWT + Bcrypt
ml models   →  XGBoost, RandomForest, ExtraTreesClassifier
data layer  →  Pandas, NumPy, Scikit-learn, FCBF
llm         →  Ollama (local inference)
```

---

## get started

**1. clone it**
```bash
git clone https://github.com/nirvana58/Nirvana.git
cd Nirvana
```

**2. install deps**
```bash
pip install -r requirements.txt
```

**3. configure**
```bash
# edit config/config.yaml — set your data paths, thresholds, SMOTE strategy
# change SECRET_KEY in app.py before any real deployment
```

**4. train your models**
```bash
python train_model.py
```

**5. start the server**
```bash
python app.py
# listening at http://localhost:8000
```

**6. connect a client**
```bash
python ntd-client.py   # register, login, submit traffic data
python ntd-admin.py    # admin interface
```

---

## api surface

```
POST  /register   →  create an account
POST  /login      →  authenticate and get a token
POST  /predict    →  submit network data for threat classification (auth required)
GET   /threats    →  retrieve logged threat events
POST  /upload     →  push a dataset for batch analysis
```

---

## project layout

```
nirvana/
├── app.py              ← fastapi core, routes, auth
├── model.py            ← model loading and inference
├── train_model.py      ← training pipeline
├── ntd-client.py       ← interactive cli client
├── ntd-admin.py        ← admin dashboard
├── config/
│   └── config.yaml     ← all tunable settings
├── data/
│   ├── users.json      ← user store
│   └── tdata/          ← training datasets
├── models/             ← serialized model files
└── FCBF_module/        ← feature selection implementation
```

---

## tuning

everything lives in `config/config.yaml`:

- data source paths
- preprocessing pipeline options
- FCBF selection thresholds
- SMOTE sampling ratios per class

---

## ⚠️ before you deploy

```bash
# this is not optional
# change the SECRET_KEY in app.py
# running with the default key in production is a vulnerability
```

---

## requirements

see `requirements.txt` for the full list. core packages:

`fastapi` `pydantic` `pandas` `scikit-learn` `xgboost` `joblib` `ollama`

---

## license

MIT. use it, fork it, break it, fix it.

---

*built for operators who want answers, not dashboards.*
