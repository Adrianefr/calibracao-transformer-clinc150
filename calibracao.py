"""Exemplo prático do seminário: calibração e detecção de fora de escopo no CLINC150.

Treina do zero um Transformer pequeno (2 camadas, ~0,96 M de parâmetros) como classificador de
150 intenções e mede: (1) ECE antes e depois de temperature scaling; (2) detecção de frases fora
de escopo com MSP, energia e distância de Mahalanobis (AUROC e FPR@95).

Dados: data_full.json de https://github.com/clinc/oos-eval (pasta data/).
Uso:   python3 calibracao.py data_full.json [semente]
Salva os logits em saida_semente<semente>.npz, que o ensemble.py lê (predição seletiva e ensemble).
Requer: numpy, torch, scikit-learn. Roda em CPU (cerca de 5 a 8 minutos por semente).
"""
import json, re, sys, time
import numpy as np, torch, torch.nn as nn

path = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0; ls = 0.0
torch.manual_seed(seed); np.random.seed(seed); 
d = json.load(open(path))
tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())
intents = sorted({y for _, y in d['train']})
yid = {y: i for i, y in enumerate(intents)}
vocab = {'<pad>': 0, '<unk>': 1}
for s, _ in d['train']:
    for w in tok(s): vocab.setdefault(w, len(vocab))
L = 30
def enc(rows):
    X = np.zeros((len(rows), L), dtype=np.int64)
    for i, (s, _) in enumerate(rows):
        ids = [vocab.get(w, 1) for w in tok(s)][:L]
        X[i, :len(ids)] = ids
    return torch.tensor(X)
lab = lambda rows: torch.tensor([yid[y] for _, y in rows])
Xtr, ytr = enc(d['train']), lab(d['train'])
Xva, yva = enc(d['val']), lab(d['val'])
Xte, yte = enc(d['test']), lab(d['test'])
Xoo = enc(d['oos_test'])

class TinyTransformer(nn.Module):
    def __init__(s, V, C, D=128, depth=2, heads=4):
        super().__init__()
        s.emb = nn.Embedding(V, D, padding_idx=0)
        s.pos = nn.Parameter(torch.randn(1, L, D) * .02)
        layer = nn.TransformerEncoderLayer(D, heads, 2 * D, dropout=0.1, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(D)
        s.head = nn.Linear(D, C)
    def features(s, x):
        pad = x == 0
        z = s.enc(s.emb(x) + s.pos, src_key_padding_mask=pad)
        m = (~pad).unsqueeze(-1).float()
        return s.norm((z * m).sum(1) / m.sum(1).clamp(min=1))   # média dos tokens
    def forward(s, x):
        return s.head(s.features(x))

model = TinyTransformer(len(vocab), len(intents))
EPOCHS, BS = 20, 64
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * ((len(Xtr) + BS - 1) // BS))
lossf = nn.CrossEntropyLoss(label_smoothing=ls)
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), BS):
        b = perm[i:i + BS]
        opt.zero_grad(); lossf(model(Xtr[b]), ytr[b]).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
model.eval()
@torch.no_grad()
def run(X):
    F, Z = [], []
    for i in range(0, len(X), 512):
        f = model.features(X[i:i + 512]); F.append(f); Z.append(model.head(f))
    return torch.cat(F).numpy(), torch.cat(Z).numpy()
Ftr, Ztr = run(Xtr); Fva, Zva = run(Xva); Fte, Zte = run(Xte); Foo, Zoo = run(Xoo)

# ---------------------------------------------------------------- calibração
from sklearn.metrics import roc_auc_score
yva, yte = yva.numpy(), yte.numpy()

def softmax(z):
  z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def nll(p, y):
  return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

def ece(p, y, bins=10):   # p: softmax, y: rótulos
  c, ok = p.max(1), p.argmax(1) == y
  e = 0.0
  for a in np.arange(bins) / bins:
    m = (c > a) & (c <= a + 1 / bins)
    if m.any():
      e += m.mean() * abs(ok[m].mean() - c[m].mean())
  return e

def fit_T(z, y):          # z: logits de validação
  Ts = np.arange(0.05, 5, 0.01)
  return min(Ts, key=lambda T: nll(softmax(z / T), y))

T = fit_T(Zva, yva)
p0, p1 = softmax(Zte), softmax(Zte / T)
print(f'semente {seed}: treino {time.time() - t0:.0f}s, parâmetros {sum(p.numel() for p in model.parameters())}')
print(f'acurácia {100 * (p0.argmax(1) == yte).mean():.2f}%   T aprendido {T:.2f}')
print(f'sem calibração: confiança média {100 * p0.max(1).mean():.2f}%  ECE {100 * ece(p0, yte):.2f}%')
print(f'com T:          confiança média {100 * p1.max(1).mean():.2f}%  ECE {100 * ece(p1, yte):.2f}%')

# ---------------------------------------------------------------- fora de escopo (pontuação maior = mais estranho)
def maha(Ftr, ytr, F):
  Ftr, F = Ftr.astype(np.float64), F.astype(np.float64)
  mu = np.stack([Ftr[ytr == c].mean(0) for c in range(int(ytr.max()) + 1)])
  Xc = Ftr - mu[ytr]
  S = Xc.T @ Xc / len(Ftr)
  S = S + 1e-3 * np.trace(S) / len(S) * np.eye(len(S))   # pequena regularização
  P = np.linalg.inv(S)
  return np.stack([np.einsum('nd,de,ne->n', F - m, P, F - m) for m in mu], 1).min(1)

lse = lambda z: z.max(1) + np.log(np.exp(z - z.max(1, keepdims=True)).sum(1))
scores = {'MSP': (-softmax(Zte).max(1), -softmax(Zoo).max(1)),
          'Energia': (-lse(Zte), -lse(Zoo)),
          'Mahalanobis': (maha(Ftr, ytr.numpy(), Fte), maha(Ftr, ytr.numpy(), Foo))}
for k, (s_id, s_oo) in scores.items():
  auroc = roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s_oo))], np.r_[s_id, s_oo])
  fpr95 = (s_id >= np.percentile(s_oo, 5)).mean()        # frases normais barradas ao detectar 95% das estranhas
  print(f'{k:12s} AUROC {100 * auroc:.1f}%   FPR@95 {100 * fpr95:.1f}%')

# ---------------------------------------------------------------- predição seletiva e arquivo para o ensemble.py
ok, conf = p0.argmax(1) == yte, p0.max(1)
for cob in (1.0, 0.9, 0.8):
  k = int(round(cob * len(yte)))
  print(f'respondendo {100 * cob:.0f}% das frases (maior confiança): acurácia {100 * ok[np.argsort(-conf)[:k]].mean():.2f}%')
print(f'AUROC de erro (a confiança separa acerto de erro?): {100 * roc_auc_score(ok, conf):.1f}%')
np.savez_compressed(f'saida_semente{seed}.npz', Zva=Zva, yva=yva, Zte=Zte, yte=yte, Zoo=Zoo)
