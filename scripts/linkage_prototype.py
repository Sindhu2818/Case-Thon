import os, sys, csv, re, itertools, json, math
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "hackathon", "Infinum_2026_Candidate_Dataset_Pack")

# ---------------- stdlib csv ----------------

def load(name):
    with open(os.path.join(DATA, name + ".csv"), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

# ---------------- normalization ----------------
DIACRITICS = {}
TOKEN_FIX = {
    "mm": "mm", "kum": "kumari", "kumar": "kumar", "sharma": "sharma", "devi": "devi",
    "amma": "amma", "bai": "bai", "lour": "lourdu", "gari": "", "garu": "", "rao": "rao",
}

def strip_diacritics(s):
    out = []
    for ch in s:
        o = ord(ch)
        if 0x0300 <= o <= 0x036F:
            continue
        if 0xC0 <= o <= 0x17F:
            for base, reps in [("A", "AAAA"), ]:
                pass
        # simple latin-1 fold
        if ch in "ÀÁÂÃÄÅàáâãäå":
            ch = "a"
        elif ch in "ÈÉÊËèéêë":
            ch = "e"
        elif ch in "ÌÍÎÏìíîï":
            ch = "i"
        elif ch in "ÒÓÔÕÖòóôõö":
            ch = "o"
        elif ch in "ÙÚÛÜùúûü":
            ch = "u"
        elif ch in "Çç":
            ch = "c"
        elif ch in "Ññ":
            ch = "n"
        out.append(ch)
    return "".join(out)

def norm_name(s):
    if not s:
        return ""
    s = strip_diacritics(str(s)).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    toks = s.split()
    stop = {"mr", "mrs", "ms", "dr", "smt", "shri", "kumari"}  # keep kumari? it's meaningful; keep
    stop = {"mr", "mrs", "ms", "dr", "smt", "shri"}
    toks = [t for t in toks if t and t not in stop]
    return " ".join(toks)

def name_tokens(s):
    return norm_name(s).split()

def norm_phone(s):
    if s is None:
        return ""
    s = re.sub(r"\D", "", str(s))
    if len(s) > 10 and s.startswith("91") and len(s) >= 12:
        s = s[-10:]
    return s

def last4(s):
    return s[-4:] if len(s) >= 4 else ""

def norm_village(s):
    if not s:
        return ""
    s = strip_diacritics(str(s)).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # common transformations
    s = s.replace("pally", "palli")
    return s

def norm_geo(s):
    if not s:
        return ""
    s = strip_diacritics(str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()

def edit1(a, b):
    """True if edit distance <=1 for same-length typo (char swap or single sub)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        # single substitution
        diff = sum(1 for x, y in zip(a, b) if x != y)
        if diff == 1:
            return True
        # single adjacent transposition
        i = 0
        while i < la and a[i] == b[i]:
            i += 1
        if i < la - 1 and a[i + 1:] == b[i + 1:][::-1][:0] + b[i + 1:]:
            pass
        # transposition check
        if la >= 2:
            for i in range(la - 1):
                if a[i] != b[i]:
                    if a[i] == b[i+1] and a[i+1] == b[i] and a[i+2:] == b[i+2:]:
                        return True
            return diff == 1
    # length diff 1: insertion/deletion
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            if skipped:
                return False
            skipped = True
            j += 1
    return True

def lev(a, b, cap=2):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            c = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + c)
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[lb]

# ---------------- load ----------------
ref = load("patient_360_reference")
tele = load("teleconsultations")
ncd = load("ncd_screening")
rx = load("prescriptions")
disp = load("medicine_dispensing")
lab = load("lab_tests")
fh = load("followup_visits")
vh = load("visit_history")
out = load("outreach_actions")
stock = load("medicine_stock_status")
fac = load("facility_reference")
geo = load("geography_reference")

print("loaded. ref:", len(ref), "tele:", len(tele))

# ---------------- name blocking + candidate generation ----------------
# Build inverted index: token4 -> patient ids
def tok_key(tok):
    return tok[:4]

ref_index = defaultdict(set)
ref_by_id = {r["patient_id"]: r for r in ref}
for r in ref:
    toks = name_tokens(r["canonical_name"])
    for t in toks:
        ref_index[tok_key(t)].add(r["patient_id"])

VILL_ALIASES = defaultdict(set)
VILL_ALIAS_OF = {}

def village_alias(v):
    key = norm_village(v)
    if key in VILL_ALIAS_OF:
        return VILL_ALIAS_OF[key]
    # find canonical alias via blocking by first 4 chars
    best = key
    for a in VILL_ALIAS_GROUP.get(key[:4], set()):
        pass
    VILL_ALIAS_OF[key] = key
    return key

# canonical village vocabulary
VILL_VOCAB = {}
for g in geo:
    VILL_VOCAB[norm_village(g["village"])] = g["village"]

ref_vill_vocab = {norm_village(r["village"]) for r in ref}

def edit1(a, b):
    return None  # replaced below

def sim_name(a_toks, b_toks):
    """Return (score, evidence). a = source tokens, b = reference tokens."""
    if not a_toks or not b_toks:
        return 0.0, "missing"
    bset = set(b_toks)
    matched = 0
    ev = []
    for t in a_toks:
        if t in bset:
            matched += 1
            continue
        hit = False
        for bt in bset:
            d = lev(t, bt, cap=1)
            if d <= 1:
                matched += 0.75
                ev.append(t + "~" + bt)
                hit = lev(t, bt, cap=1) <= 1
                break
        # else unmatched
    score = matched / max(len(a_toks), len(b_toks))
    return min(score, 1.0), ("fuzzy:" + ",".join(ev[:3]) if ev else "exact")

def sim_village(a, b):
    a, b = norm_village(a), norm_village(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 5 and len(b) >= 5 and lev(a, b, cap=2) <= 1:
        return 0.9
    return 0.0

# ---------------- candidate generation & scoring ----------------
_sc_cache = {}

def candidates_for(name, village, mobile):
    # phone-suffix candidates are the strongest blocking key; use them first
    cand = set()
    if mobile:
        l4 = last4(norm_phone(mobile))
        if l4:
            cand |= ref_by_l4.get(l4, set())
    if not cand:
        toks = name_tokens(name)
        keys = sorted({tok_key(t) for t in toks if t})
        for k in keys[:3]:
            cand |= ref_index.get(k, set())
            if len(cand) > 400:
                break
    return cand

ref_by_l4 = defaultdict(set)
for r in ref:
    l4 = last4(norm_phone(r.get("masked_mobile", "")))
    if l4:
        ref_by_l4[l4].add(r["patient_id"])

def score_pair(src_name, src_vill, src_phone, src_gender, src_age, r):
    ev = []
    # phone evidence
    p = 0.0
    if src_phone and r.get("masked_mobile"):
        lp = last4(norm_phone(src_phone))
        lr = last4(norm_phone(r["masked_mobile"]))
        if lp and lp == lr:
            p = 1.0
            ev.append("phone4")
    nsc, nev = sim_name(name_tokens(src_name), name_tokens(r["canonical_name"]))
    vsc = sim_village(src_vill, r["village"])
    gsc = 1.0 if (src_gender and r.get("gender") and src_gender == r["gender"]) else 0.0
    asc = 0.0
    if src_age and r.get("age_as_of_2026"):
        d = abs(int(float(src_age)) - int(float(r["age_as_of_2026"])))
        asc = 1.0 if d == 0 else (0.5 if d <= 2 else (0.25 if d <= 5 else 0.0))
    score = 0.45 * p + 0.35 * nsc + 0.12 * vsc + 0.05 * gsc + 0.03 * asc
    return score, {"phone4": p, "name": round(nsc, 2), "village": vsc,
                   "gender": gsc, "age": asc, "evidence": ev or ([nev] if nev not in ("missing", "exact") else [])}

# ---------------- linkage run ----------------
LINK_TABLES = {
    "teleconsultations": (tele, "tele_source_patient_id"),
    "ncd_screening": (ncd, "ncd_source_patient_id"),
    "prescriptions": (rx, "rx_source_patient_id"),
    "medicine_dispensing": (disp, "pharm_source_patient_id"),
    "lab_tests": (lab, "lab_source_patient_id"),
    "followup_visits": (fh, "visit_source_patient_id"),
    "visit_history": (vh, "visit_source_patient_id"),
    "outreach_actions": (out, "outreach_source_patient_id"),
}

linkage_rows = []   # submission rows
audit = defaultdict(list)  # source_id -> top candidates
assigned = defaultdict(int)  # (table, src_id) -> count of canonical ids assigned

for tname, (rows, sid_col) in LINK_TABLES.items():
    for i, row in enumerate(rows):
        src_id = row[sid_col]
        key = (row.get("patient_name", ""), row.get("village", ""), row.get("mobile", ""),
               row.get("gender", ""), row.get("age", ""))
        if key in _sc_cache:
            scored = _sc_cache[key]
        else:
            cands = candidates_for(*key[:3])
            scored = []
            for pid in cands:
                r = ref_by_id[pid]
                s, ev = score_pair(key[0], key[1], key[2], key[3], key[4], r)
                scored.append((s, pid, ev))
            scored.sort(key=lambda x: -x[0])
            scored = scored[:8]
            _sc_cache[key] = scored
        top = scored[0] if scored else (0.0, "", {})
        second = scored[1] if len(scored) > 1 else (0.0, "", {})
        margin = top[0] - second[0]
        # --- decision rule (documented) ---
        auto = False
        if top[0] >= 0.90 and margin >= 0.05:
            auto = True
        elif (top[0] >= 0.80 and margin >= 0.10):
            auto = True
        elif (top[0] >= 0.55 and margin >= 0.35 and top[1]):
            auto = top[1].startswith("P")  # requires phone4 match component? no - simplified
        score, pid, ev = top
        conf = "high" if score >= 0.90 else ("medium" if score >= 0.75 else ("low" if score >= 0.55 else "none"))
        if auto and pid:
            linkage_rows.append({
                "source_table": tname,
                "source_patient_id": src_id,
                "canonical_patient_id": pid,
                "match_score": round(score, 3),
                "match_confidence": conf,
                "match_method": "auto_phone4_name_geo" if (p in (1.0,) if False else False) else "auto_rule",
            })
            assigned[(tname, src_id)] += 1
        else:
            audit[tname].append((src_id, round(score, 3), round(margin, 3),
                                 pid, ev.get("evidence", [])))

print("auto-linked rows:", len(linkage_rows), "of",
      sum(len(rows) for rows, _ in LINK_TABLES.values()))
c = Counter(r["match_confidence"] for r in linkage_rows)
print("confidence:", dict(c))
uniq_src = len({(r['source_table'], r['source_patient_id']) for r in linkage_rows})
print("unique source ids linked:", uniq_src)
n_pat = len({r['canonical_patient_id'] for r in linkage_rows})
print("distinct canonical patients touched:", n_pat)
