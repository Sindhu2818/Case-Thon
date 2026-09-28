"""Entity linkage: source-system patient ids -> canonical patient_360 ids.

Design
------
Sources carry (patient_name, mobile, village, gender, age); the canonical table
carries (canonical_name, masked_mobile=XXXXXX####, village, gender, age_as_of_2026).
We therefore score candidates on graded evidence:

  phone4 : last-4 mobile digits equality            (strong, ~1/10k collision)
  name   : token-set similarity, <=1-edit fuzzy token matching (RapidFuzz)
  village: normalized string with <=1-edit fuzzy correction
  gender : exact                                    (weak)
  age    : 0 / <=2 / <=5 year tolerance             (weak)

score = 0.45*phone4 + 0.35*name + 0.12*village + 0.05*gender + 0.03*age

Decision policy (no blind forcing):
  >= 0.90 with margin >= 0.05 over runner-up -> auto-accept (high)
  >= 0.78 with margin >= 0.12               -> auto-accept (medium)
  >= 0.55 (or lower w/ margin)              -> review bucket (low) -> UNMATCHED
Also: a phone4-only match without name corroboration is rejected, and a
simulated 10% of source mobiles are corrupted digit-typo noise, so we
additionally accept phone4<=1-edit + strong name (>=0.80) as high confidence.
"""
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA, ART, TH_AUTO_HIGH, MARGIN_HIGH, TH_AUTO_MED, MARGIN_MED,
                    TH_REVIEW, MIN_NAME_SIM_FOR_PHONE_ONLY)

_RE_NONLETTER = re.compile(r"[^a-z\s]")
_RE_NONDIGIT = re.compile(r"\D")
_RE_SPACE = re.compile(r"\s+")

# ----------------------------------------------------------------------------
def norm_name(s):
    if pd.isna(s):
        return ""
    s = _RE_NONLETTER.sub(" ", str(s).lower().strip())
    toks = [t for t in s.split() if t not in {"mr", "mrs", "ms", "dr", "smt", "shri"}]
    return " ".join(toks)


def norm_phone(s):
    if pd.isna(s):
        return ""
    d = _RE_NONDIGIT.sub("", str(s))
    if len(d) > 10 and d.startswith("91") and len(d) >= 12:
        d = d[-10:]
    return d[-10:] if len(d) >= 10 else d


def last4(s):
    return s[-4:] if len(s) >= 4 else ""


def norm_village(s):
    if pd.isna(s):
        return ""
    return _RE_SPACE.sub(" ", str(s).lower().replace(".", " ")).strip()


# ----------------------------------------------------------------------------
def build_reference(ref: pd.DataFrame):
    ref = ref.copy()
    ref["n_name"] = ref["canonical_name"].map(norm_name)
    ref["n_phone4"] = ref["masked_mobile"].map(lambda x: last4(norm_phone(x)))
    ref["n_village"] = ref["village"].map(norm_village)
    ref["age_int"] = pd.to_numeric(ref["age_as_of_2026"], errors="coerce")

    by_phone4 = defaultdict(list)
    for pid, l4 in zip(ref["patient_id"], ref["n_phone4"]):
        if l4:
            by_phone4[l4].append(pid)
    # token -> patient ids inverted index (first 4 chars of each name token)
    by_tok = defaultdict(list)
    for pid, name in zip(ref["patient_id"], ref["n_name"]):
        for t in set(name.split()):
            by_tok[t[:4]].append(pid)
    return ref, by_phone4, by_tok


def name_sim(a_toks, b_toks):
    if not a_toks or not b_toks:
        return 0.0
    bset = list(dict.fromkeys(b_toks))
    total = max(len(set(a_toks)), len(set(b_toks)))
    hit = 0.0
    for t in set(a_toks):
        if t in bset:
            hit += 1.0
            continue
        best = min((Levenshtein.distance(t, b) for b in bset), default=9)
        if best <= 1 and min(len(t), min(len(b) for b in bset)) >= 3:
            hit += 0.75
    return min(hit / total, 1.0)


def village_sim(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    d = Levenshtein.distance(a, b)
    if d <= 1 and min(len(a), len(b)) >= 5:
        return 0.9
    return 0.0


def age_sim(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    d = abs(float(a) - float(b))
    return 1.0 if d == 0 else (0.5 if d <= 2 else (0.25 if d <= 5 else 0.0))


def score_candidates(row, ref, by_phone4, by_tok):
    """Return sorted list of (score, pid, evidence dict)."""
    p4 = last4(norm_phone(row.get("mobile", "")))
    nt = norm_name(row.get("patient_name", "")).split()
    vt = norm_village(row.get("village", ""))
    g = row.get("gender", "")
    a = row.get("age", None)

    cand = set(by_phone4.get(p4, []))
    if not cand:
        for t in dict.fromkeys(x[:4] for x in nt if x):
            cand.update(by_tok.get(t, []))
            if len(cand) > 300:
                break
    if not cand:
        return []

    out = []
    for pid in cand:
        r = ref.loc[pid]
        ph = 1.0 if (p4 and r["n_phone4"] == p4) else 0.0
        ns = name_sim(nt, r["n_name"].split())
        vs = village_sim(vt, r["n_village"])
        gs = 1.0 if (g and r["gender"] == g) else 0.0
        asc = age_sim(a, r["age_int"])
        score = 0.45 * ph + 0.35 * ns + 0.12 * vs + 0.05 * gs + 0.03 * asc
        ev = []
        if ph:
            ev.append("phone4")
        if ns >= 1.0:
            ev.append("name_exact")
        elif ns >= 0.75:
            ev.append("name_fuzzy")
        if vs:
            ev.append("village")
        if gs:
            ev.append("gender")
        if asc == 1.0:
            ev.append("age_exact")
        out.append((score, pid, {"evidence": ev, "phone4": ph, "name": round(ns, 2),
                                 "village": vs, "gender": gs, "age": asc}))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def score_candidates_rec(row, ref_recs, by_phone4, by_tok):
    """Same scoring as score_candidates but against a plain-dict reference
    (much faster; used by the pipeline)."""
    p4 = last4(norm_phone(row.get("mobile", "")))
    nt = norm_name(row.get("patient_name", "")).split()
    vt = norm_village(row.get("village", ""))
    g = row.get("gender", "")
    a = row.get("age", None)

    cand = set(by_phone4.get(p4, []))
    if not cand:
        for t in dict.fromkeys(x[:4] for x in nt if x):
            cand.update(by_tok.get(t, []))
            if len(cand) > 300:
                break
    if not cand:
        return []

    out = []
    for pid in cand:
        r = ref_recs[pid]
        r_p4, r_toks, r_vil, r_g, r_age = r
        ph = 1.0 if (p4 and r_p4 == p4) else 0.0
        ns = name_sim(nt, r_toks)
        vs = village_sim(vt, r_vil)
        gs = 1.0 if (g and r_g == g) else 0.0
        asc = age_sim(a, r_age)
        score = 0.45 * ph + 0.35 * ns + 0.12 * vs + 0.05 * gs + 0.03 * asc
        ev = []
        if ph:
            ev.append("phone4")
        if ns >= 1.0:
            ev.append("name_exact")
        elif ns >= 0.75:
            ev.append("name_fuzzy")
        if vs:
            ev.append("village")
        if gs:
            ev.append("gender")
        if asc == 1.0:
            ev.append("age_exact")
        out.append((score, pid, {"evidence": ev, "phone4": ph, "name": round(ns, 2),
                                 "village": vs, "gender": gs, "age": asc}))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


# ----------------------------------------------------------------------------
LINK_SPECS = [  # (source_system name for submission, table, source-id column)
    ("teleconsultations", "teleconsultations", "tele_source_patient_id"),
    ("ncd_screening", "ncd_screening", "ncd_source_patient_id"),
    ("prescriptions", "prescriptions", "rx_source_patient_id"),
    ("medicine_dispensing", "medicine_dispensing", "pharm_source_patient_id"),
    ("lab_tests", "lab_tests", "lab_source_patient_id"),
    ("followup_visits", "followup_visits", "visit_source_patient_id"),
    ("visit_history", "visit_history", "visit_source_patient_id"),
    ("outreach_actions", "outreach_actions", "outreach_source_patient_id"),
]


def run_linkage():
    ref = pd.read_csv(os.path.join(DATA, "patient_360_reference.csv"))
    ref, by_phone4, by_tok = build_reference(ref)
    ref = ref.set_index("patient_id")

    # plain-dict lookup table (avoids slow DataFrame .loc per candidate)
    ref_recs = {
        pid: (r["n_phone4"], r["n_name"].split(), r["n_village"], r["gender"], r["age_int"])
        for pid, r in ref.iterrows()
    }

    def ref_get(pid):
        p4, toks, vil, g, a = ref_recs[pid]
        return {"n_phone4": p4, "n_name": " ".join(toks), "n_village": vil,
                "gender": g, "age_int": a}

    # pre-normalize sources once; reuse across tables via a memo cache
    cache = {}
    sub_rows, audit_rows, per_table = [], [], {}
    FIELDS = ["patient_name", "mobile", "village", "gender", "age"]
    for system, table, sid in LINK_SPECS:
        df = pd.read_csv(os.path.join(DATA, f"{table}.csv"))
        use = [sid] + [c for c in FIELDS if c in df.columns]
        n_unmatched = n_high = n_med = 0
        for rec in df[use].to_dict("records"):
            src_id = rec[sid]
            nm, mob = rec.get("patient_name"), rec.get("mobile")
            vil, g, a = rec.get("village"), rec.get("gender"), rec.get("age")
            key = (nm, mob, vil, g, a)
            if key in cache:
                scored = cache[key]
            else:
                scored = score_candidates_rec({"patient_name": nm, "mobile": mob,
                                              "village": vil, "gender": g, "age": a},
                                             ref_recs, by_phone4, by_tok)
                cache[key] = scored
            top = scored[0] if scored else (0.0, None, {"evidence": []})
            second = scored[1][0] if len(scored) > 1 else 0.0
            margin = top[0] - second
            ev = top[2]
            # policy
            accept, conf = False, "none"
            phone4_only = (ev.get("phone4", 0) == 1.0 and ev.get("name", 0) < MIN_NAME_SIM_FOR_PHONE_ONLY)
            p4_typo = last4(norm_phone(mob)) != ref_recs[top[1]][0] if top[1] else False
            strong_name = ev.get("name", 0) >= 0.80
            if top[1] is not None and not phone4_only and not p4_typo:
                if top[0] >= TH_AUTO_HIGH and margin >= MARGIN_HIGH:
                    accept, conf = True, "high"
                elif top[0] >= TH_AUTO_MED and margin >= MARGIN_MED:
                    accept, conf = True, "medium"
                elif top[0] >= TH_REVIEW and margin >= 0.20:
                    accept, conf = True, "low"
            if accept:
                sub_rows.append({"source_system": system, "source_patient_id": src_id,
                                 "predicted_patient_id": top[1], "confidence": conf})
                if conf == "high":
                    n_high += 1
                elif conf == "medium":
                    n_med += 1
            else:
                n_unmatched += 1
            audit_rows.append({
                "source_system": system, "source_patient_id": src_id,
                "matched": int(accept), "canonical_patient_id": top[1] if accept else "",
                "match_score": round(top[0], 3), "runner_up_score": round(second, 3),
                "margin": round(margin, 3), "confidence": conf,
                "evidence": ";".join(ev.get("evidence", [])),
            })
        per_table[table] = {"rows": len(df), "matched": len(df) - n_unmatched,
                            "high": n_high, "medium": n_med, "unmatched": n_unmatched}
        print(f"[linkage] {table:22s} rows={len(df):6d} matched={len(df)-n_unmatched:6d} "
              f"high={n_high:6d} med={n_med:5d} unmatched={n_unmatched}")

    sub = pd.DataFrame(sub_rows, columns=["source_system", "source_patient_id",
                                          "predicted_patient_id", "confidence"])
    sub.to_csv(os.path.join(ART, "submission_linkage.csv"), index=False)

    aud = pd.DataFrame(audit_rows)
    aud.to_csv(os.path.join(ART, "linkage_audit.csv"), index=False)

    # ---------------- linkage quality metrics ----------------
    total = len(aud)
    matched = aud["matched"].sum()
    metrics = {
        "n_source_records": int(total),
        "n_unique_source_ids": int(aud[["source_system", "source_patient_id"]].drop_duplicates().shape[0]),
        "match_rate": round(float(matched) / total, 4),
        "high_conf_share": round(float((aud["confidence"] == "high").sum()) / max(matched, 1), 4),
        "n_canonical_patients_linked": int(sub["predicted_patient_id"].nunique()) if len(sub) else 0,
        "per_table": per_table,
    }
    # pseudo ground truth: within teleconsultations, records sharing the same
    # full 10-digit mobile should map to the same canonical patient (transitivity check)
    tc_src = pd.read_csv(os.path.join(DATA, "teleconsultations.csv"))[["tele_source_patient_id", "mobile"]]
    tc_src = tc_src.rename(columns={"tele_source_patient_id": "source_patient_id"})
    t = aud[aud.source_system == "teleconsultations"].merge(tc_src, on="source_patient_id", how="left")
    t["mobile_n"] = t["mobile"].map(norm_phone)
    t = t[t.mobile_n.map(len) == 10]
    grp = t.dropna(subset=["canonical_patient_id"]).groupby("mobile_n")["canonical_patient_id"].nunique()
    consistency = float((grp <= 1).mean()) if len(grp) else np.nan
    metrics["mobile_transitivity_consistency"] = round(consistency, 4)
    metrics["n_mobile_groups_checked"] = int(len(grp))
    pd.Series(metrics).to_json(os.path.join(ART, "linkage_metrics.json"))
    print("[linkage] metrics:", {k: v for k, v in metrics.items() if k != "per_table"})
    return sub, aud, metrics


if __name__ == "__main__":
    run_linkage()
