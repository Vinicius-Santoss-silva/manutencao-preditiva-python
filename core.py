"""VibraSense: processamento de sinais didático e experimento de ML."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, resample_poly
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

FS = 12000
WINDOW = 2048
CLASSES = ("Saudável", "Pista interna", "Elemento rolante", "Pista externa")
FILE_IDS = {
    "Saudável": {0:97, 1:98, 2:99, 3:100},
    "Pista interna": {0:105, 1:106, 2:107, 3:108},
    "Elemento rolante": {0:118, 1:119, 2:120, 3:121},
    "Pista externa": {0:130, 1:131, 2:132, 3:133},
}
RPM_APPROX = {0:1797, 1:1772, 2:1750, 3:1730}
FEATURE_NAMES = ["RMS", "Fator de crista", "Assimetria", "Curtose", "Centroide espectral", "Entropia espectral", "Energia >1kHz"]

def data_path(folder, label, load):
    return Path(folder) / f"{FILE_IDS[label][load]}.mat"

def load_record(path, label, load):
    contents = loadmat(path)
    keys = [k for k in contents if k.endswith("_DE_time")]
    if not keys:
        raise ValueError(f"Sem DE_time no arquivo {path}")
    wave = np.asarray(contents[keys[0]], dtype=float).ravel()
    if label == "Saudável":
        wave = resample_poly(wave, 1, 4)
    if len(wave) < WINDOW or not np.all(np.isfinite(wave)):
        raise ValueError(f"Sinal inválido: {path}")
    rpm_keys = [k for k in contents if k.endswith("RPM")]
    rpm = float(np.asarray(contents[rpm_keys[0]]).ravel()[0]) if rpm_keys else RPM_APPROX[load]
    return wave, rpm

def demo_signal(label, seed=8, seconds=1.0, rpm=1797.0):
    """Sinal sintético: nunca confundir com medição industrial."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS*seconds))/FS
    fr = rpm/60
    wave = 0.07*np.sin(2*np.pi*fr*t) + 0.035*np.sin(2*np.pi*2*fr*t)
    wave += rng.normal(0, 0.055, len(t))
    freqs = {"Pista interna":5.4*fr, "Elemento rolante":2.3*fr, "Pista externa":3.6*fr}
    if label in freqs:
        envelope = np.maximum(np.sin(2*np.pi*freqs[label]*t), 0)**16
        strength = {"Pista interna":.6, "Elemento rolante":.4, "Pista externa":.52}[label]
        wave += strength*envelope*np.sin(2*np.pi*1800*t)
    return wave, rpm

def features(wave):
    x = np.asarray(wave, dtype=float) - np.mean(wave)
    rms = np.sqrt(np.mean(x*x))
    f, p = welch(x, FS, nperseg=min(512, len(x)))
    total = np.sum(p)+1e-12
    entropy = -np.sum((p/total)*np.log(p/total+1e-12))/np.log(len(p))
    return np.array([rms, np.max(np.abs(x))/(rms+1e-12), skew(x), kurtosis(x, fisher=False),
                     np.sum(f*p)/total, entropy, np.sum(p[f>=1000])/total], dtype=float)

def spectrum(wave):
    x = np.asarray(wave, dtype=float) - np.mean(wave)
    f, p = welch(x, FS, nperseg=min(1024, len(x)))
    return f, 10*np.log10(p+1e-12)

def windows(signal, max_windows=90):
    count = min(len(signal)//WINDOW, max_windows)
    return (signal[i*WINDOW:(i+1)*WINDOW] for i in range(count))

def train_cross_load(folder, train_load=0, test_load=1):
    """Separa arquivos de carga diferente. Validação experimental, não industrial."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for i, label in enumerate(CLASSES):
        train_signal, _ = load_record(data_path(folder, label, train_load), label, train_load)
        test_signal, _ = load_record(data_path(folder, label, test_load), label, test_load)
        tr = [features(w) for w in windows(train_signal)]
        te = [features(w) for w in windows(test_signal)]
        Xtr.extend(tr); ytr.extend([i]*len(tr))
        Xte.extend(te); yte.extend([i]*len(te))
    model = RandomForestClassifier(n_estimators=170, max_depth=12, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(np.asarray(Xtr), np.asarray(ytr))
    pred = model.predict(np.asarray(Xte))
    return model, {
        "acuracia":accuracy_score(yte,pred),
        "acuracia_balanceada":balanced_accuracy_score(yte,pred),
        "matriz":confusion_matrix(yte,pred,labels=list(range(len(CLASSES)))),
        "amostras_teste":len(yte),
    }
