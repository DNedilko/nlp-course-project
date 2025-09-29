
import math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- config --------------------
TRAIN_CSV = "data/train_translated.csv"
VALID_CSV = "data/validation_translated.csv"
LANG_COL = "lang"
TEXT_COL = "question"
CONTEXT_COL = "context"

LANG_MAP = {"ar": "Arabic", "ko": "Korean", "te": "Telugu"}
LANGS_TO_RUN = ["Arabic", "Korean", "Telugu"]
SEQ_LENS = [3, 4, 5, 6, 7, 8]

MIN_FREQ = 3
LOWERCASE = True
ADD_EOS = True

BATCH_SIZE = 64
EMBED_DIM = 256
HID_DIM = 512
LAYERS = 2
DROPOUT = 0.3
LR = 2e-3
EPOCHS = 10
CLIP = 1.0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# -------------------- tokenize (Unicode-aware) --------------------
import regex as re
import unicodedata

WORD_RE = re.compile(r"\p{L}+\p{M}*|\p{N}+|[^\p{Z}\p{C}\p{L}\p{N}]", re.UNICODE)

def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u0640", "")  # Arabic tatweel
    # remove combining marks (diacritics); comment out if you want to keep them
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    t = text.strip()
    t = normalize_text(t)
    if LOWERCASE:
        t = t.lower()
    return WORD_RE.findall(t)

# -------------------- vocab utils --------------------
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

def build_vocab(texts, min_freq=MIN_FREQ):
    freq = {}
    for t in texts:
        toks = tokenize(t)
        if ADD_EOS and toks: toks += [EOS]
        for tok in toks: freq[tok] = freq.get(tok, 0) + 1
    itos = [PAD, UNK, BOS, EOS]
    stoi = {tok: i for i, tok in enumerate(itos)}
    for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and tok not in stoi:
            stoi[tok] = len(itos); itos.append(tok)
    return stoi, itos

def encode(text, stoi, add_eos=True):
    toks = tokenize(text)
    if ADD_EOS and add_eos and toks: toks += [EOS]
    return [stoi.get(tok, stoi[UNK]) for tok in toks]

# -------------------- dataset --------------------
class ConcatDataset(Dataset):
    def __init__(self, texts, stoi, seq_len):
        self.PAD_ID = stoi[PAD]; self.BOS_ID = stoi[BOS]
        buf = []
        for t in texts:
            ids = encode(t, stoi)
            if ids:
                buf.extend([self.BOS_ID] + ids)
        self.ids = torch.tensor(buf, dtype=torch.long)
        self.seq_len = seq_len
        self.n = max(0, (len(self.ids) - 1) // self.seq_len)

    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + 1 + self.seq_len]
        return x, y

# -------------------- model --------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab, emb, hid, layers, drop, pad_id, tie=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.lstm  = nn.LSTM(emb, hid, num_layers=layers, dropout=drop, batch_first=True)
        self.drop  = nn.Dropout(drop)
        if tie and emb == hid:
            self.proj = nn.Identity()
            self.fc   = nn.Linear(emb, vocab, bias=False)
            self.fc.weight = self.embed.weight
        else:
            self.proj = nn.Identity() if emb == hid else nn.Linear(hid, emb, bias=False)
            self.fc   = nn.Linear(emb, vocab)
        self.reset()

    def reset(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        for name, p in self.lstm.named_parameters():
            if "weight" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x, state=None):
        z, _ = self.lstm(self.embed(x), state)
        z = self.drop(z)
        z = self.proj(z)
        return self.fc(z)

# -------------------- train / eval --------------------
def eval_ppl(model, loader, pad_id, vocab_size):
    model.eval(); total_loss = 0.0; total_tok = 0
    crit = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            loss = crit(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            total_tok  += (y != pad_id).sum().item()
    avg_nll = total_loss / max(1, total_tok)
    return math.exp(avg_nll)

def train_once(train_texts, valid_texts, seq_len):
    # vocab from TRAIN ONLY
    stoi, itos = build_vocab(train_texts)
    PAD_ID = stoi[PAD]; V = len(itos)

    # datasets / loaders
    tr_ds = ConcatDataset(train_texts, stoi, seq_len)
    va_ds = ConcatDataset(valid_texts, stoi, seq_len)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        return float("nan")
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # model, opt, loss
    model = LSTMLM(V, EMBED_DIM, HID_DIM, LAYERS, DROPOUT, PAD_ID, tie=True).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")

    # training
    for _ in range(EPOCHS):
        model.train()
        for x, y in tr_ld:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits.reshape(-1, V), y.reshape(-1))
            (loss / max(1, (y != PAD_ID).sum().item())).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

    # validation perplexity
    return eval_ppl(model, va_ld, PAD_ID, V)

# -------------------- run grid --------------------
if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)

    # normalize language labels in both
    for df in (train_df, valid_df):
        df[LANG_COL] = df[LANG_COL].astype(str).map(lambda s: LANG_MAP.get(s, s))

    rows = []
    for L in LANGS_TO_RUN:
        train_texts = train_df.loc[train_df[LANG_COL] == L, TEXT_COL].dropna().astype(str).tolist()
        valid_texts = valid_df.loc[valid_df[LANG_COL] == L, TEXT_COL].dropna().astype(str).tolist()
        for S in SEQ_LENS:
            ppl = train_once(train_texts, valid_texts, seq_len=S)
            rows.append({"Language": L, "LSTM Perplexity": round(ppl, 2), "Sequence Length": S})
            print(f"{L:7s} | seq={S} | PPL={ppl:.2f}")

    # Also run for English context (same for all)
    train_texts = train_df[CONTEXT_COL].dropna().astype(str).tolist()
    valid_texts = valid_df[CONTEXT_COL].dropna().astype(str).tolist()
    for S in range(40, 90, 5):
        ppl = train_once(train_texts, valid_texts, seq_len=S)
        rows.append({"Language": "English context", "LSTM Perplexity": round(ppl, 2), "Sequence Length": S})
        print(f"En| seq={S} | PPL={ppl:.2f}")

    results = pd.DataFrame(rows)
    print("\n=== Results ===")
    print(results)

    # LaTeX table (simple, one row per (lang, seq_len))
    latex = results[["Language", "LSTM Perplexity", "Sequence Length"]] \
        .to_latex(index=False, escape=False, column_format="|l|r|c|", longtable=False)

    print("\nLaTeX table:\n")
    print("\\begin{table}\n\\centering\n" + latex +
          "\\caption{LSTM validation perplexity by language and sequence length}"
          "\\label{tab:lstm-ppl}\n\\end{table}")
