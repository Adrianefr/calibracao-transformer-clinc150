"""Predição seletiva e ensemble a partir dos logits salvos por calibracao.py (uma ou mais sementes).
Uso: python3 ensemble.py saida_semente0.npz saida_semente1.npz saida_semente2.npz
"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score

def softmax(z):
  z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def nll(p, y):
  return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

def ece(p, y, bins=10):
  c, ok = p.max(1), p.argmax(1) == y
  e = 0.0
  for a in np.arange(bins) / bins:
    m = (c > a) & (c <= a + 1 / bins)
    if m.any():
      e += m.mean() * abs(ok[m].mean() - c[m].mean())
  return e

def fit_T(z, y):
  Ts = np.arange(0.05, 5, 0.01)
  return min(Ts, key=lambda T: nll(softmax(z / T), y))

def seletiva(p, y, cobertura):
  """acurácia quando o modelo só responde nas frases de maior confiança (fração = cobertura)"""
  c, ok = p.max(1), p.argmax(1) == y
  k = int(round(cobertura * len(y)))
  return ok[np.argsort(-c)[:k]].mean()

def auroc_erro(p, y):
  """quão bem a confiança separa acertos de erros (50% = acaso)"""
  return roc_auc_score(p.argmax(1) == y, p.max(1))

def resumo(nome, p, y, p_oo):
  msp_id, msp_oo = p.max(1), p_oo.max(1)
  au = roc_auc_score(np.r_[np.zeros(len(msp_id)), np.ones(len(msp_oo))], np.r_[-msp_id, -msp_oo])
  print(f'{nome:22s} acc {100 * (p.argmax(1) == y).mean():5.2f}  ECE {100 * ece(p, y):5.2f}  AUROC erro {100 * auroc_erro(p, y):5.1f}  '
        f'acc@90% {100 * seletiva(p, y, .9):5.2f}  acc@80% {100 * seletiva(p, y, .8):5.2f}  AUROC OOD (MSP) {100 * au:5.1f}')
  return dict(acc=(p.argmax(1) == y).mean(), ece=ece(p, y), auroc_err=auroc_erro(p, y), sel90=seletiva(p, y, .9), sel80=seletiva(p, y, .8), auroc_ood=au)

if __name__ == '__main__':
  runs = [np.load(f) for f in sys.argv[1:]]
  y = runs[0]['yte']
  P, PT, POO = [], [], []
  for i, r in enumerate(runs):
    T = fit_T(r['Zva'], r['yva'])
    P.append(softmax(r['Zte'])); PT.append(softmax(r['Zte'] / T)); POO.append(softmax(r['Zoo']))
    resumo(f'semente {i}', P[-1], y, POO[-1]); resumo(f'semente {i} + T={T:.2f}', PT[-1], y, softmax(r['Zoo'] / T))
  if len(runs) > 1:
    pe, poo = np.mean(P, 0), np.mean(POO, 0)
    resumo(f'ensemble de {len(runs)}', pe, y, poo)
    pred = np.stack([p.argmax(1) for p in P]); conc = (pred == pred[0]).all(0)
    print(f'as {len(runs)} sementes discordam em {100 * (~conc).mean():.1f}% das frases; '
          f'acurácia do ensemble: {100 * (pe.argmax(1) == y)[conc].mean():.1f}% quando concordam, {100 * (pe.argmax(1) == y)[~conc].mean():.1f}% quando discordam')
