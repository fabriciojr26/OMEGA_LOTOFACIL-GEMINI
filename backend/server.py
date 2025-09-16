from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, validator
from typing import List, Dict, Any, Optional
import os
import json
import time
import hashlib
import random
import numpy as np
import pandas as pd
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
DATA_CSV_URL = os.getenv("OMEGA_DATA_CSV_URL", "") 
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(',')

app = FastAPI(
    title="Lotofácil Ômega v2.0 — Engine",
    version="2.0",
    description="Motor de geração de portfólios para Lotofácil com R adaptativo e heurísticas."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Cache de Dados ----------------
X_CACHE = None
L_POS_CACHE = None
S_CACHE = None
LAST_MODIFIED_TIME = 0
CACHE_TTL_SECONDS = 3600 # Cache de 1 hora

def load_data_from_url(url: str) -> np.ndarray:
    try:
        logger.info(f"Carregando dados de: {url}")
        df = pd.read_csv(url)
        bin_cols = sorted([c for c in df.columns if str(c).startswith("Dezena_")], key=lambda x: int(x.split("_")[1]))
        X = df[bin_cols].astype(np.uint8).values
        X = X[X.sum(axis=1) == 15]
        logger.info(f"Dados carregados com sucesso. Shape: {X.shape}")
        return X
    except Exception as e:
        logger.error(f"Falha ao carregar dados da URL: {e}")
        return None

def generate_synthetic_data() -> np.ndarray:
    logger.warning("Gerando dados sintéticos como fallback.")
    rng = np.random.default_rng(42)
    P = np.array([0.62 if (i % 5 in [1, 2, 3]) else 0.58 for i in range(25)])
    rows = []
    for _ in range(2000):
        row = np.zeros(25, dtype=np.uint8)
        indices = rng.choice(25, size=15, replace=False, p=P/P.sum())
        row[indices] = 1
        if row.sum() == 15:
            rows.append(row)
    return np.array(rows)

def get_data() -> np.ndarray:
    global X_CACHE, LAST_MODIFIED_TIME
    now = time.time()
    if X_CACHE is not None and (now - LAST_MODIFIED_TIME < CACHE_TTL_SECONDS):
        return X_CACHE
    
    if DATA_CSV_URL:
        data = load_data_from_url(DATA_CSV_URL)
        if data is not None:
            X_CACHE = data
            LAST_MODIFIED_TIME = now
            return X_CACHE

    X_CACHE = generate_synthetic_data()
    LAST_MODIFIED_TIME = now
    return X_CACHE

# ---------------- Funções Matemáticas e Heurísticas ----------------
def z_score(v):
    v = np.asarray(v, dtype=float)
    std = v.std()
    return (v - v.mean()) / (std if std > 1e-8 else 1)

def softmax(x, tau=1.0):
    e_x = np.exp((x - np.max(x)) / tau)
    return e_x / e_x.sum()

def mepa_score(nums: List[int]) -> float:
    rows, cols = np.zeros(5), np.zeros(5)
    evens = 0
    for n in nums:
        r, c = divmod(n - 1, 5)
        rows[r] += 1
        cols[c] += 1
        if n % 2 == 0:
            evens += 1
    
    row_pen = np.mean(np.abs(rows - 3))
    col_pen = np.mean(np.abs(cols - 3))
    par_pen = 0 if 7 <= evens <= 8 else np.abs(evens - 7.5)
    penalty = row_pen + col_pen + 0.5 * par_pen
    return 1.0 / (1.0 + penalty)

def poisson_binomial_pmf(p):
    n = len(p)
    dp = np.zeros(n + 1)
    dp[0] = 1.0
    for pi in p:
        dp[1:] = dp[1:] * (1 - pi) + dp[:-1] * pi
        dp[0] *= (1 - pi)
    return dp

def edge_density(nums: List[int], L_pos: np.ndarray) -> float:
    s_nums = sorted([n - 1 for n in nums])
    hits = sum(1 for i in range(15) for j in range(i + 1, 15) if L_pos[s_nums[i], s_nums[j]])
    return hits / 105.0 # 15 choose 2

def hamming_distance(a: List[int], b: List[int]) -> int:
    return len(set(a) ^ set(b))

# ---------------- Lógica de Sinais (SIPSA-lite) ----------------
def compute_signals_and_L_pos(X: np.ndarray):
    global S_CACHE, L_POS_CACHE
    
    T_fft = min(X.shape[0], 256)
    s_aoh = z_score([np.mean(np.abs(np.fft.rfft(X[:T_fft, j].astype(float))[1:])) for j in range(25)])

    Lbk = min(X.shape[0], 400)
    recent = X[:Lbk]
    last_pos = [np.where(recent[:, j] == 1)[0][0] if np.any(recent[:, j]) else Lbk for j in range(25)]
    s_irs = z_score(-np.log1p(last_pos))

    P = recent.mean(axis=0)
    C = (recent.T @ recent) / Lbk
    L = C - np.outer(P, P)
    L_pos = L > 0.06
    np.fill_diagonal(L_pos, False)
    s_caic = z_score(L_pos.sum(axis=0))

    rnp = np.zeros(25)
    for j in range(25):
        idx = np.where(recent[:, j] == 1)[0]
        if len(idx) > 1:
            diffs = np.diff(idx)
            rnp[j] = np.sum(np.isin(diffs, [1, 2, 3, 4, 5]))
    s_rnp = z_score(rnp)

    weights = {"AOH": 0.35, "RNPAR": 0.20, "IRS": 0.25, "CAIC": 0.20}
    S = weights["AOH"]*s_aoh + weights["RNPAR"]*s_rnp + weights["IRS"]*s_irs + weights["CAIC"]*s_caic
    
    S_CACHE, L_POS_CACHE = S, L_pos
    return S, L_pos

def get_S_and_L_pos():
    if S_CACHE is not None and L_POS_CACHE is not None and (time.time() - LAST_MODIFIED_TIME < CACHE_TTL_SECONDS):
        return S_CACHE, L_POS_CACHE
    X = get_data()
    return compute_signals_and_L_pos(X)

def S_to_p(S):
    ranks = (S - S.min()) / (S.max() - S.min() + 1e-9)
    p = np.clip(0.5 + 0.3 * (ranks - 0.5), 0.35, 0.75)
    return p * (15.0 / p.sum())

# ---------------- Schemas da API ----------------
class FromLastReq(BaseModel):
    ultimo: List[conint(ge=1, le=25)]
    N: conint(ge=100, le=5000) = 500
    R_mode: conint(ge=5, le=15) = 9
    R_range: List[conint(ge=5, le=15)] = [8, 11]
    K: conint(ge=1, le=1000) = 10
    k_tail: conint(ge=11, le=15) = 13

    @validator('ultimo')
    def check_ultimo_length(cls, v):
        if len(v) != 15 or len(set(v)) != 15:
            raise ValueError('O último sorteio deve conter 15 dezenas únicas.')
        return v

class CheckReq(BaseModel):
    resultado: List[conint(ge=1, le=25)]
    jogos: List[List[conint(ge=1, le=25)]]

# ---------------- Endpoints ----------------
@app.get("/omega/adaptive_r_suggestion")
def get_adaptive_r():
    X = get_data()
    if X.shape[0] < 20: return {"suggested_r_mode": 9}
    
    overlaps = [hamming_distance(X[i], X[i+1]) / 2 for i in range(min(X.shape[0]-1, 200))]
    mean_overlap = 15 - np.mean(overlaps)
    return {"suggested_r_mode": int(round(mean_overlap))}

@app.post("/omega/from_last_draw")
def from_last_draw(req: FromLastReq):
    try:
        S, L_pos = get_S_and_L_pos()
        p = S_to_p(S)
        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)

        base, non_base_universe = set(req.ultimo), [i for i in range(1, 26) if i not in req.ultimo]
        non_base_s = np.array([S[i-1] for i in non_base_universe])
        
        R_vals = list(range(req.R_range[0], req.R_range[1] + 1))
        R_weights = softmax([1.0 / (1.0 + abs(r - req.R_mode)) for r in R_vals])

        cands = []
        for _ in range(req.N):
            R = random.choices(R_vals, weights=R_weights, k=1)[0]
            base_sample = set(random.sample(sorted(list(base)), k=R))
            
            non_base_probs = softmax(non_base_s, tau=0.8)
            completion = np.random.choice(non_base_universe, size=15-R, replace=False, p=non_base_probs)
            
            nums = sorted(list(base_sample) + list(completion))
            
            if not (200 <= sum(nums) <= 320): continue
            evens = sum(1 for n in nums if n % 2 == 0)
            if not (7 <= evens <= 8): continue
            
            dens = edge_density(nums, L_pos)
            if not (0.20 <= dens <= 0.45): continue
            
            mean_s = np.mean([S[n-1] for n in nums])
            mepa_v = mepa_score(nums)
            dens_comp = 1.0 - abs(dens - 0.32)
            score = 0.55 * mean_s + 0.25 * mepa_v + 0.20 * dens_comp
            cands.append({"nums": nums, "R": R, "score": score})

        # Seleção de portfólio com guard-rails
        cands.sort(key=lambda x: x['score'], reverse=True)
        portfolio, pair_counts = [], {}
        for cand in cands:
            if len(portfolio) >= req.K: break
            
            # Hamming check
            if any(hamming_distance(cand['nums'], p['nums']) < 4 for p in portfolio): continue
            
            # Pair exposure check
            s_nums = sorted(cand['nums'])
            pairs = [(s_nums[i], s_nums[j]) for i in range(15) for j in range(i+1, 15)]
            if any(pair_counts.get(p, 0) >= 12 for p in pairs): continue
            
            portfolio.append(cand)
            for p in pairs: pair_counts[p] = pair_counts.get(p, 0) + 1
        
        # Cálculo de probabilidades
        q_list = []
        for item in portfolio:
            item_p = [p[n-1] for n in item['nums']]
            pmf = poisson_binomial_pmf(item_p)
            q_ge_k = pmf[req.k_tail:].sum()
            item['q_ge_k'] = float(q_ge_k)
            q_list.append(q_ge_k)

        p_success = 1.0 - np.prod([1 - q for q in q_list]) if q_list else 0.0

        payload = {
            "ts": time.time(), "ultimo": req.ultimo, "R_mode": req.R_mode,
            "R_range": req.R_range, "K": len(portfolio), "k_tail": req.k_tail,
            "p_success_ge_k": float(p_success), "results": portfolio, "seed_used": seed,
        }
        return payload
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro em from_last_draw: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

@app.post('/omega/check')
def omega_check(req: CheckReq):
    res = set(req.resultado)
    if len(res) != 15: raise HTTPException(status_code=400, detail="O resultado deve ter 15 dezenas únicas.")
    
    hits = [{"idx": i, "acertos": len(res.intersection(set(jogo)))} for i, jogo in enumerate(req.jogos)]
    hist = {str(i): 0 for i in range(16)}
    for h in hits: hist[str(h['acertos'])] += 1
        
    return {"hits_per_game": hits, "hist": hist, "max_hit": max(h['acertos'] for h in hits) if hits else 0}

@app.get("/")
def root():
    return {"message": "Lotofácil Ômega v2.0 Engine - Online"}

