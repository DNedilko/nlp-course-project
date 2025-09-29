import unicodedata, regex as re, math, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

TRAIN_CSV = "data/train_translated.csv"
TEST_CSV = "data/validation_translated.csv"
LANG_COL, TEXT_COL, CONTEXT_COL, LABEL_COL = "lang", "question", "context", "answerable"
LANG_MAP = {"ar": "Arabic", "ko": "Korean", "te": "Telugu"}
LANGS = ["Arabic", "Korean", "Telugu"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
MAX_Q, MAX_CTX, STRIDE = 64, 384, 96
ADD_SEP, SEP = True, "<sep>"

WORD_RE = re.compile(r"\p{L}+\p{M}*|\p{N}+|[^\p{Z}\p{C}\p{L}\p{N}]")

def tok(text):
    if not isinstance(text, str):
        return []
    t = unicodedata.normalize("NFKC", text).replace("\u0640", "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return WORD_RE.findall(t.strip().casefold())

def build_vocab(texts, min_freq):
    freq = {}
    for t in texts:
        for x in tok(t):
            freq[x] = freq.get(x, 0) + 1
    itos = [PAD, UNK, BOS, EOS] + [w for w, c in sorted(freq.items(), key=lambda z: (-z[1], z[0])) if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode(text, stoi):
    return [stoi.get(x, stoi[UNK]) for x in tok(text)]

def build_vocab_from_qc(df, min_freq):
    texts = pd.concat([df[TEXT_COL], df[CONTEXT_COL]], axis=0).dropna().astype(str).tolist()
    texts.append(SEP)
    return build_vocab(texts, min_freq)

def make_windows(q, ctx, y, stoi):
    q_ids = encode(q, stoi)[:MAX_Q]
    c_ids = encode(ctx, stoi)
    if not c_ids:
        return []
    sep_id = stoi.get(SEP, stoi.get(EOS)) if ADD_SEP else None
    out = []
    start = 0
    while start < len(c_ids):
        end = min(len(c_ids), start + MAX_CTX)
        win = c_ids[start:end]
        ids = [stoi[BOS]] + q_ids + ([sep_id] if sep_id is not None else []) + win
        seg = [0] * (1 + len(q_ids) + (1 if sep_id is not None else 0)) + [1] * len(win)
        out.append((ids, seg, int(y)))
        if end == len(c_ids):
            break
        start = end - STRIDE
    return out

class PairWindows(Dataset):
    def __init__(self, df, stoi):
        self.items = []
        for _, r in df.iterrows():
            self.items += make_windows(r[TEXT_COL], r[CONTEXT_COL], r[LABEL_COL], stoi)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ids, seg, y = self.items[i]
        return torch.tensor(ids).long(), torch.tensor(seg).long(), torch.tensor(y).long()

def pad_collate(batch, pad_id):
    xs, segs, ys = zip(*batch)
    T = max(len(x) for x in xs)
    X = torch.full((len(xs), T), pad_id).long()
    S = torch.zeros((len(segs), T)).long()
    for i, (x, s) in enumerate(zip(xs, segs)):
        X[i, :len(x)] = x
        S[i, :len(s)] = s
    return X, S, torch.stack(ys)


class LSTMCrossEncoder(nn.Module):
    def __init__(self, vocab, emb, hid, layers, drop, pad_id, ln_in, ln_out, w,attend_context_only=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.seg = nn.Embedding(2, emb)  # 0=Q, 1=C
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, dropout=drop, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(drop)
        # question-aware additive attention
        self.W_h = nn.Linear(hid, hid, bias=False)
        self.W_q = nn.Linear(hid, hid, bias=False)
        self.v   = nn.Linear(hid, 1,  bias=False)
        self.fc  = nn.Linear(hid, 2)
        self.attend_context_only = attend_context_only

    def forward(self, X, S):
        # X: [B,T] tokens, S: [B,T] 0=Q,1=C
        mask = (X != self.emb.padding_idx)  # [B,T]
        e = self.emb(X) + self.seg(S)       # [B,T,E]
        H, _ = self.lstm(e)                 # [B,T,H]

        # summary of the question to condition attention
        q_mask = (S == 0) & mask            # [B,T]
        # avoid empty-question edge cases
        q_lens = q_mask.sum(dim=1).clamp(min=1)
        q_vec = (H * q_mask.unsqueeze(-1)).sum(dim=1) / q_lens.unsqueeze(-1)  # [B,H]

        # optionally restrict attention to context tokens only
        att_mask = ((S == 1) if self.attend_context_only else (mask)) & mask  # [B,T]

        # additive attention scores
        # score_t = v^T tanh(W_h h_t + W_q q_vec)
        Wh = self.W_h(H)                                 # [B,T,H]
        Wq = self.W_q(q_vec).unsqueeze(1)                # [B,1,H]
        scores = self.v(torch.tanh(Wh + Wq)).squeeze(-1) # [B,T]

        # mask out non-attended positions
        scores = scores.masked_fill(~att_mask, float('-inf'))
        alpha = torch.softmax(scores, dim=1)             # [B,T]

        # attended context vector
        ctx = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B,H]
        out = self.fc(self.drop(ctx))                     # [B,2]
        return out
# class LSTMCrossEncoder(nn.Module):
#     def __init__(self, vocab, emb, hid, layers, drop, pad_id, ln_in, ln_out, wn):
#         super().__init__()
#         self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
#         self.seg = nn.Embedding(2, emb)
#         self.ln_in = nn.LayerNorm(emb) if ln_in else nn.Identity()
#         self.lstm = nn.LSTM(emb, hid, num_layers=layers, dropout=drop, batch_first=True)
#         self.drop = nn.Dropout(drop)
#         self.ln_out = nn.LayerNorm(hid) if ln_out else nn.Identity()
#         head = nn.Linear(hid, 2)
#         self.fc = nn.utils.parametrizations.weight_norm(head) if wn else head
#     def forward(self, X, S):
#         e = self.ln_in(self.emb(X) + self.seg(S))
#         _, (h, _) = self.lstm(e)
#         h = self.ln_out(h[-1])
#         return self.fc(self.drop(h))  # [B,2]

@torch.no_grad()
def eval_df(model, stoi, df, device, pooling):
    yt, yp = [], []
    for _, r in df.iterrows():
        items = make_windows(r[TEXT_COL], r[CONTEXT_COL], 0, stoi)
        if not items:
            yp.append(0)
            yt.append(int(r[LABEL_COL]))
            continue
        pad_id = stoi[PAD]
        xs, ss = [], []
        for ids, seg, _ in items:
            xs.append(torch.tensor(ids).long())
            ss.append(torch.tensor(seg).long())
        T = max(len(x) for x in xs)
        X = torch.full((len(xs), T), pad_id).long()
        S = torch.zeros((len(ss), T)).long()
        for i, (x, s) in enumerate(zip(xs, ss)):
            X[i, :len(x)] = x
            S[i, :len(s)] = s
        logits = model(X.to(device), S.to(device))  # [W,2]
        if pooling == "mean":
            agg = logits.mean(dim=0)
        else:
            agg = logits.max(dim=0).values
        pred = int(torch.argmax(agg).item())
        yp.append(pred)
        yt.append(int(r[LABEL_COL]))
    acc = accuracy_score(yt, yp)
    bacc = balanced_accuracy_score(yt, yp)
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    return {"acc": acc, "bacc": bacc, "f1": f, "prec": p, "rec": r}

def epoch_loss(model, loader, crit, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for X, S, Y in loader:
            X, S, Y = X.to(device), S.to(device), Y.to(device)
            total += crit(model(X, S), Y).item()
            n += 1
    return total / max(1, n)

def init_weights(m, kind):
    if isinstance(m, nn.Linear):
        if kind == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif kind == "uniform":
            nn.init.uniform_(m.weight, -0.1, 0.1)
        elif kind == "normal":
            nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        if kind == "normal":
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif kind == "uniform":
            nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].zero_()
    elif isinstance(m, (nn.LSTM, nn.GRU)):
        for name, p in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

def make_loader(df, stoi, batch_size, shuffle, sampler_mode):
    ds = PairWindows(df, stoi)
    pad_id = stoi[PAD]
    sampler = None
    if sampler_mode == "weighted" and len(ds) > 0:
        ys = torch.tensor([y for _, _, y in ds])
        pos = float(ys.sum())
        neg = float(len(ys) - pos)
        w1 = neg / max(1.0, pos)
        w0 = 1.0
        weights = torch.where(ys > 0.5, torch.tensor(w1), torch.tensor(w0)).double()
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=lambda b: pad_collate(b, pad_id)), len(ds)

def class_weights_from_df(df, device):
    y = df[LABEL_COL].astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    total = counts.sum()
    # inverse frequency normalized to mean 1.0
    w = total / (2.0 * np.maximum(1, counts))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float, device=device)  # [w0,w1]

def train_one(tr_df, va_df, cfg, device):
    stoi, _ = build_vocab_from_qc(tr_df, cfg["min_freq"])
    if SEP not in stoi:
        stoi[SEP] = len(stoi)
    pad_id = stoi[PAD]
    V = len(stoi)

    tr_ld, _ = make_loader(tr_df, stoi, cfg["batch_size"], True, cfg["sampler"])
    va_ld, _ = make_loader(va_df, stoi, cfg["batch_size"], False, "none")

    model = LSTMCrossEncoder(
        V, cfg["embed_dim"], cfg["hid_dim"], cfg["layers"], cfg["dropout"],
        pad_id, cfg["ln_in"], cfg["ln_out"], cfg["weight_norm"]
    ).to(device)
    model.apply(lambda m: init_weights(m, cfg["init"]))

    ce_weights = class_weights_from_df(tr_df, device) if cfg["use_class_weight"] else None
    crit = nn.CrossEntropyLoss(weight=ce_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1, verbose=False) if cfg["use_sched"] else None

    best_metric = -math.inf
    best_state = None
    best_val_loss = math.inf
    bad_epochs = 0
    min_delta = float(cfg["early_min_delta"])

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        run = 0.0
        steps = 0
        for X, S, Y in tr_ld:
            X, S, Y = X.to(device), S.to(device), Y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(X, S), Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            run += loss.item()
            steps += 1

        tr_loss = run / max(1, steps)
        va_loss = epoch_loss(model, va_ld, crit, device)
        if sched is not None:
            sched.step(va_loss)

        m = eval_df(model, stoi, va_df[[TEXT_COL, CONTEXT_COL, LABEL_COL]], device, cfg["pooling"])
        score = m["bacc"] if cfg["early_key"] == "bacc" else m["f1"]
        print(f"  ep {ep:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_bacc {m['bacc']:.3f} | va_f1 {m['f1']:.3f}")

        improved_metric = score > best_metric + min_delta
        improved_loss = va_loss < best_val_loss - min_delta

        if improved_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        elif not improved_loss:
            bad_epochs += 1
        else:
            bad_epochs = 0

        best_val_loss = min(best_val_loss, va_loss)

        if bad_epochs >= int(cfg["early_patience"]):
            print(f"  early stop at epoch {ep} (best {cfg['early_key']}={best_metric:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, stoi


def run_language(train_df, test_df, lang, cfg_list, device):
    df_tr = train_df[train_df[LANG_COL] == lang].dropna(subset=[TEXT_COL, CONTEXT_COL, LABEL_COL]).copy()
    df_te = test_df[test_df[LANG_COL] == lang].dropna(subset=[TEXT_COL, CONTEXT_COL, LABEL_COL]).copy()
    df_tr[LABEL_COL] = df_tr[LABEL_COL].astype(int)
    df_te[LABEL_COL] = df_te[LABEL_COL].astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg_list[0]["val_fraction"], random_state=SEED)
    tr_idx, va_idx = next(sss.split(np.zeros(len(df_tr)), df_tr[LABEL_COL].to_numpy()))
    inner_tr = df_tr.iloc[tr_idx].copy()
    inner_va = df_tr.iloc[va_idx].copy()
    rows = []
    best_name = None
    best_score = -math.inf
    best_bundle = None
    for cfg in cfg_list:
        print(f"\n[{lang}] {cfg['name']} | train={len(inner_tr)} val={len(inner_va)} test={len(df_te)}")
        model, stoi = train_one(inner_tr, inner_va, cfg, device)
        val_m = eval_df(model, stoi, inner_va[[TEXT_COL, CONTEXT_COL, LABEL_COL]], device, cfg["pooling"])
        test_m = eval_df(model, stoi, df_te[[TEXT_COL, CONTEXT_COL, LABEL_COL]], device, cfg["pooling"])
        rows.append({"lang": lang, "name": cfg["name"], "val_bacc": val_m["bacc"], "val_f1": val_m["f1"], "test_bacc": test_m["bacc"], "test_f1": test_m["f1"], "val_acc": val_m["acc"], "test_acc": test_m["acc"]})
        score = val_m["bacc"]
        if score > best_score:
            best_score = score
            best_name = cfg["name"]
            best_bundle = (model, stoi)
    res = pd.DataFrame(rows)
    print(f"\n[{lang}] best by val_bacc: {best_name} -> {best_score:.3f}")
    if best_bundle is not None:
        model, stoi = best_bundle
        torch.save({"state": model.state_dict(), "stoi": stoi, "config": best_name}, f"best_{lang.lower()}.pt")
    res.to_csv(f"results_{lang.lower()}.csv", index=False)
    return res

def main():
    print("device:", DEVICE)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    for df in (train_df, test_df):
        df[LANG_COL] = df[LANG_COL].astype(str).map(lambda s: LANG_MAP.get(s, s))
    base = {"epochs": 8, "batch_size": 64, "embed_dim": 256, "hid_dim": 512, "layers": 2,
            "dropout": 0.4, "lr": 2e-3, "weight_decay": 1e-3, "clip": 0.5, "min_freq": 3,
            "val_fraction": 0.15, "pooling": "mean", "sampler": "none", "use_class_weight": True,
            "use_sched": True, "ln_in": True, "ln_out": True, "weight_norm": True, "init": "xavier",
            "early_key": "bacc", "early_patience": 2, "early_min_delta": 1e-4,
}

    def cfg(name, **kw):
        c = base.copy()
        c.update(kw)
        c["name"] = name
        return c
    grid_by_lang = {
        "Arabic": [
            cfg("ar_a", embed_dim=256, hid_dim=512, dropout=0.4, lr=2e-3, sampler="weighted", use_class_weight=False, pooling="max", ln_in=True, ln_out=True, weight_norm=True, init="xavier"),
            cfg("ar_b", embed_dim=128, hid_dim=256, dropout=0.3, lr=1e-3, sampler="none", use_class_weight=True, pooling="mean", ln_in=False, ln_out=True, weight_norm=False, init="uniform")
        ],
        "Korean": [
            cfg("ko_a", embed_dim=256, hid_dim=512, dropout=0.5, lr=1e-3, sampler="weighted", use_class_weight=False, pooling="mean", ln_in=True, ln_out=False, weight_norm=True, init="xavier"),
            cfg("ko_b", embed_dim=256, hid_dim=256, dropout=0.3, lr=2e-3, sampler="none", use_class_weight=True, pooling="max", ln_in=False, ln_out=False, weight_norm=False, init="uniform")
        ],
        "Telugu": [
            cfg("te_a", embed_dim=128, hid_dim=512, dropout=0.4, lr=2e-3, sampler="weighted", use_class_weight=False, pooling="max", ln_in=True, ln_out=True, weight_norm=True, init="xavier"),
            cfg("te_b", embed_dim=256, hid_dim=256, dropout=0.3, lr=1e-3, sampler="none", use_class_weight=True, pooling="mean", ln_in=False, ln_out=True, weight_norm=False, init="uniform")
        ]
    }
    all_out = []
    for lang in LANGS:
        res = run_language(train_df, test_df, lang, grid_by_lang[lang], DEVICE)
        all_out.append(res)
    full = pd.concat(all_out, ignore_index=True)
    print("\nTop per language by val_bacc:")
    print(full.sort_values(["lang", "val_bacc"], ascending=[True, False]).groupby("lang").head(3).to_string(index=False))
    full.to_csv("results_all_languages.csv", index=False)

if __name__ == "__main__":
    main()
