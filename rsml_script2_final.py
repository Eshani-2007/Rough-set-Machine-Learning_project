"""
FUZZY-CONFORMAL VARIABLE PRECISION ROUGH SET MACHINE LEARNING (FC-VPRS-ML)
===========================================================================
Enhanced version of the original VPRS-ML script for ammonium adsorption
capacity classification on hydrochar.

WHAT WAS CHANGED AND WHY (novelty upgrades vs. the original script)
-------------------------------------------------------------------
[N1] ADAPTIVE beta SELECTION (data-driven VPRS precision).
     WHY: In the original script beta = 0.92 was hand-picked. Reviewers of
     ML-for-adsorption papers routinely flag arbitrary hyperparameters as
     "tuning to the test set". Selecting beta by internal stratified
     validation converts a fixed heuristic into a principled, reproducible
     procedure and is itself a small methodological contribution
     ("self-calibrating VPRS").

[N2] FUZZY MEMBERSHIP INFERENCE OVER CRISP ROUGH RULES (fuzzy-rough hybrid).
     WHY: Crisp k-means bins make prediction brittle: a sample 1% past a
     bin edge matches zero rules, which forced the original ad-hoc Hamming
     fallback. Replacing crisp matching at INFERENCE time with triangular
     fuzzy memberships (rules are still induced by classical VPRS) gives a
     graded match degree for every sample, removes the fallback entirely,
     and creates a genuine "fuzzy-rough" hybrid — a recognized but rarely
     applied direction in environmental ML, i.e., real novelty rather than
     an engineering patch.

[N3] SPLIT-CONFORMAL PREDICTION ON TOP OF THE ROUGH-SET SCORES.
     WHY: This is the strongest novelty lever. Rough set theory already
     speaks the language of "boundary regions" (uncertain objects between
     lower and upper approximations). Split-conformal prediction formalizes
     exactly this idea with a FINITE-SAMPLE COVERAGE GUARANTEE: the model
     outputs a SET of plausible capacity classes whose coverage >= 1-alpha
     by construction. To our knowledge, conformalized rough-set classifiers
     have not been reported for adsorption datasets, and the conceptual
     bridge (boundary region <-> conformal prediction set) is an
     attractive, citable framing for the paper.

[N4] ORDINAL-AWARE EVALUATION (quadratic weighted kappa + ordinal MAE).
     WHY: Low/Medium/High capacity classes are ORDERED. Plain accuracy
     treats confusing "Low" with "High" the same as "Low" with "Medium",
     which understates the model's practical usefulness. QWK and ordinal
     MAE are the correct metrics and immediately differentiate the paper
     from the many studies that only report accuracy/F1.

[N5] REPEATED STRATIFIED CROSS-VALIDATION + BASELINE BENCHMARK.
     WHY: A single 80/20 split on a small adsorption dataset produces
     high-variance results that reviewers distrust. Repeated stratified
     K-fold gives mean +/- std, and benchmarking against Decision Tree /
     Random Forest quantifies the interpretability-accuracy trade-off —
     the standard evidence package expected by JHM/CEJ-tier journals.

[N6] PHYSICALLY INTERPRETABLE RULE EXPORT (original units, not z-scores).
     WHY: The original script standardized features BEFORE discretization,
     so bin edges lived in z-score space and rules could not be read as
     chemistry ("BET > 250 m2/g AND O/C < 0.15 -> High"). Scaling is
     mathematically irrelevant for per-feature k-means binning anyway
     (monotone affine transform preserves 1-D k-means partitions), so it
     was removed. Rules are now exported with real bin intervals, which is
     what makes a rough-set paper attractive to a materials audience:
     directly testable mechanistic hypotheses.

[FIX] Stratified train/test split (the original split was unstratified,
     risking class-ratio drift after discretizing y), and the discretizer /
     rule induction remain fit on training data only (no leakage).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import random

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# ============================================================================
# STEP 1 & 2: DATA LOADING & PREPARATION
# ============================================================================
print("=" * 80)
print("FUZZY-CONFORMAL VARIABLE PRECISION ROUGH SET ML (FC-VPRS-ML)")
print("=" * 80)

data_path = Path("hydro_ammina_converted__1_.csv")
df = pd.read_csv(data_path)
df_clean = df.dropna().drop_duplicates().reset_index(drop=True)

conditional_attributes = [
    'C', 'H/C', 'O/C', '(O+N)/C', 'Ash', 'pH_bio',
    'BET', 'V', 'Temp', 'pH', 'C0(mg/g)'
]
target_attribute = 'Q(mg/g)'

X = df_clean[conditional_attributes].copy()
y = df_clean[target_attribute].copy()

# ============================================================================
# STEP 3: TARGET DISCRETIZATION & STRATIFIED SPLIT
# ============================================================================
# [FIX] Discretize y with quantile bins fit on the FULL y only to build a
# stratification key, then re-fit the final discretizer on TRAIN ONLY for
# the actual labels. Stratification prevents class-ratio drift between
# train and test, which is a common silent failure with small datasets.
_strat_key = pd.qcut(y, q=3, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG_SEED, stratify=_strat_key
)

# Final label discretizer: fit on training targets only (no leakage).
discretizer_y = KBinsDiscretizer(n_bins=3, encode='ordinal',
                                 strategy='kmeans', subsample=None)
y_train_classes = discretizer_y.fit_transform(
    y_train.values.reshape(-1, 1)).flatten().astype(int)
y_test_classes = discretizer_y.transform(
    y_test.values.reshape(-1, 1)).flatten().astype(int)

CLASS_NAMES = {0: 'Low-Q', 1: 'Medium-Q', 2: 'High-Q'}

# [N6] NOTE: StandardScaler was deliberately REMOVED. Per-feature 1-D
# k-means binning is invariant to affine rescaling, so scaling added zero
# modeling value while destroying the physical interpretability of the
# exported rules (bin edges were in z-score units). Working in raw units
# lets rules read as chemistry (e.g., "BET in [180, 420] m2/g").

# ============================================================================
# FC-VPRS-ML CLASSIFIER
# ============================================================================

class FuzzyConformalRSML(BaseEstimator, ClassifierMixin):
    """VPRS rule induction + fuzzy inference + adaptive beta.

    Rule INDUCTION is classical (crisp) variable-precision rough sets, so
    the theoretical machinery (lower/upper approximations, core, reduct)
    is untouched and citable. Only the INFERENCE step is fuzzified [N2],
    and beta is selected from data rather than hand-set [N1].
    """

    def __init__(self, n_bins_attributes=4, min_rule_confidence=0.80,
                 min_rule_support=0.04, max_rules_per_class=40,
                 beta='auto', beta_grid=(0.80, 0.85, 0.90, 0.95),
                 val_fraction=0.25, random_state=RNG_SEED):
        self.n_bins_attributes = n_bins_attributes
        self.min_rule_confidence = min_rule_confidence
        self.min_rule_support = min_rule_support
        self.max_rules_per_class = max_rules_per_class
        # [N1] beta='auto' triggers internal validation to choose the VPRS
        # precision level; a float reproduces the original fixed behaviour.
        self.beta = beta
        self.beta_grid = beta_grid
        self.val_fraction = val_fraction
        self.random_state = random_state

    # ------------------------- discretization -------------------------
    def _fit_discretizer(self, X):
        self.feature_discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins_attributes, encode='ordinal',
            strategy='kmeans', subsample=None)
        self.feature_discretizer_.fit(X)
        # Pre-compute bin centers for the fuzzy membership functions [N2].
        self.bin_centers_ = []
        for edges in self.feature_discretizer_.bin_edges_:
            centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
            self.bin_centers_.append(centers)

    def _discretize(self, X):
        return self.feature_discretizer_.transform(X).astype(int)

    def _fuzzy_memberships(self, X):
        """[N2] Triangular membership of each raw value to each bin.

        WHY: crisp binning maps a continuous descriptor (e.g., pH 6.99 vs
        7.01) to different symbols, so near-boundary samples matched no
        rule and needed the ad-hoc Hamming fallback. Triangular memberships
        centered on the k-means bin centers give every sample a graded
        degree of belonging to every bin; rule matching becomes a product
        of memberships (a standard t-norm), so every rule contributes a
        continuous vote and the fallback is no longer needed.
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        memberships = []
        for j in range(d):
            centers = self.bin_centers_[j]
            k = len(centers)
            M = np.zeros((n, k))
            for b in range(k):
                # Width = distance to the nearest neighbouring center
                # (robust to unevenly spaced k-means centers).
                if k == 1:
                    M[:, b] = 1.0
                    continue
                left = centers[b] - centers[b - 1] if b > 0 else centers[b + 1] - centers[b]
                right = centers[b + 1] - centers[b] if b < k - 1 else centers[b] - centers[b - 1]
                x = X[:, j]
                lo = np.clip(1.0 - (centers[b] - x) / max(left, 1e-12), 0, 1)
                hi = np.clip(1.0 - (x - centers[b]) / max(right, 1e-12), 0, 1)
                M[:, b] = np.where(x <= centers[b], lo, hi)
            # Normalize so memberships across bins sum to 1 per sample;
            # samples outside all triangles snap to the nearest bin.
            row_sum = M.sum(axis=1, keepdims=True)
            zero = (row_sum.flatten() == 0)
            if zero.any():
                nearest = np.argmin(np.abs(X[zero, j][:, None] - centers[None, :]), axis=1)
                M[zero, :] = 0.0
                M[zero, nearest] = 1.0
                row_sum = M.sum(axis=1, keepdims=True)
            memberships.append(M / row_sum)
        return memberships  # list of (n_samples, n_bins) arrays, one per feature

    # --------------------- classical VPRS machinery ---------------------
    # (unchanged in spirit from the original script; kept crisp on purpose)
    def _indiscernibility(self, Xd, attrs):
        if len(attrs) == 0:
            return {(): list(range(Xd.shape[0]))}
        eq = {}
        for i in range(Xd.shape[0]):
            sig = tuple(Xd[i, a] for a in attrs)
            eq.setdefault(sig, []).append(i)
        return eq

    def _lower(self, Xd, y, cls, attrs, beta):
        out = set()
        for objs in self._indiscernibility(Xd, attrs).values():
            if sum(1 for o in objs if y[o] == cls) / len(objs) >= beta:
                out.update(objs)
        return out

    def _upper(self, Xd, y, cls, attrs):
        out = set()
        for objs in self._indiscernibility(Xd, attrs).values():
            if any(y[o] == cls for o in objs):
                out.update(objs)
        return out

    @staticmethod
    def _strength(lower, upper):
        return len(lower) / len(upper) if len(upper) else 0.0

    def _core(self, Xd, y, cls, beta):
        n = Xd.shape[1]
        all_attrs = set(range(n))
        full = self._strength(self._lower(Xd, y, cls, list(all_attrs), beta),
                              self._upper(Xd, y, cls, list(all_attrs)))
        core = set()
        for a in all_attrs:
            rest = list(all_attrs - {a})
            if not rest:
                core.add(a)
                continue
            s = self._strength(self._lower(Xd, y, cls, rest, beta),
                               self._upper(Xd, y, cls, rest))
            if s < full:
                core.add(a)
        return core

    def _reduct(self, Xd, y, cls, beta):
        n = Xd.shape[1]
        all_attrs = set(range(n))
        reduct = self._core(Xd, y, cls, beta)
        target = self._strength(self._lower(Xd, y, cls, list(all_attrs), beta),
                                self._upper(Xd, y, cls, list(all_attrs)))
        remaining = all_attrs - reduct
        while remaining and len(reduct) < n:
            cur = self._strength(self._lower(Xd, y, cls, list(reduct), beta),
                                 self._upper(Xd, y, cls, list(reduct)))
            best, best_s = None, cur
            for a in remaining:
                t = reduct | {a}
                s = self._strength(self._lower(Xd, y, cls, list(t), beta),
                                   self._upper(Xd, y, cls, list(t)))
                if s > best_s:
                    best, best_s = a, s
            if best is None or best_s >= target:
                break
            reduct.add(best)
            remaining.remove(best)
        # [ROBUSTNESS FIX] Guard against an EMPTY reduct. When the data is
        # fully rough for a class (approximation strength = 0 for the full
        # attribute set), the greedy search terminates immediately and the
        # original script silently generated zero rules for that class
        # (masked by the Hamming fallback). Falling back to the full
        # attribute set keeps rule induction alive and makes the failure
        # mode explicit instead of invisible.
        if not reduct:
            reduct = set(all_attrs)
        return reduct

    def _rules_for_class(self, Xd, y, cls, reduct, min_conf, min_supp):
        rules = []
        class_size = int(np.sum(y == cls))
        total = len(y)
        attrs = sorted(reduct)
        if not attrs:
            return rules
        for r in range(1, min(3, len(attrs)) + 1):
            for cond_attrs in combinations(attrs, r):
                for vals in np.unique(Xd[:, cond_attrs], axis=0):
                    mask = np.ones(Xd.shape[0], dtype=bool)
                    for i, a in enumerate(cond_attrs):
                        mask &= (Xd[:, a] == vals[i])
                    tot = int(mask.sum())
                    if tot == 0:
                        continue
                    hit = int((mask & (y == cls)).sum())
                    conf = hit / tot
                    supp = hit / total
                    cov = hit / class_size if class_size else 0
                    if conf >= min_conf and supp >= min_supp:
                        rules.append({
                            'condition': tuple(zip(cond_attrs, vals)),
                            'confidence': conf, 'support': supp,
                            'coverage': cov,
                            'quality': 0.6 * conf + 0.2 * supp + 0.2 * cov
                        })
        rules.sort(key=lambda r: r['quality'], reverse=True)
        return rules[:self.max_rules_per_class]

    def _induce(self, Xd, y, beta):
        """[ROBUSTNESS FIX] Adaptive threshold relaxation per class.

        WHY: with strict global thresholds (conf>=0.80, supp>=0.04) a
        minority class can end up with ZERO rules, silently degrading the
        classifier to majority voting for that class — an invisible failure
        in the original script (masked by the Hamming fallback). If a class
        yields no rules, confidence/support are relaxed stepwise until at
        least one rule exists, and the relaxation is logged so the final
        effective thresholds are transparent and reportable in the paper.
        """
        rules, counts = {}, {}
        for cls in np.unique(y):
            reduct = self._reduct(Xd, y, cls, beta)
            conf, supp = self.min_rule_confidence, self.min_rule_support
            cls_rules = self._rules_for_class(Xd, y, cls, reduct, conf, supp)
            while not cls_rules and conf > 0.50:
                conf = round(conf - 0.05, 2)
                supp = max(supp / 2.0, 1.0 / len(y))
                cls_rules = self._rules_for_class(Xd, y, cls, reduct,
                                                  conf, supp)
            if not cls_rules and conf <= 0.50:
                # Last resort: accept the single best rule at any support.
                cls_rules = self._rules_for_class(Xd, y, cls, reduct,
                                                  0.0, 0.0)[:1]
            if conf != self.min_rule_confidence and cls_rules:
                print(f"      [relaxed] class {cls}: conf->{conf}, "
                      f"supp->{supp:.4f} ({len(cls_rules)} rules)")
            rules[cls] = cls_rules
            counts[cls] = int(np.sum(y == cls))
        return rules, counts

    # ------------------------------ fit ------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self._fit_discretizer(X)
        Xd = self._discretize(X)

        if self.beta == 'auto':
            # [N1] Data-driven beta: internal stratified split, choose the
            # precision level that maximizes macro-F1 on held-out data.
            # WHY: turns the most criticized "magic number" of VPRS into a
            # transparent, reproducible model-selection step.
            try:
                Xt, Xv, yt, yv = train_test_split(
                    X, y, test_size=self.val_fraction,
                    random_state=self.random_state, stratify=y)
            except ValueError:
                Xt, Xv, yt, yv = train_test_split(
                    X, y, test_size=self.val_fraction,
                    random_state=self.random_state, stratify=None)
            Xt_d = self._discretize(Xt)
            best_beta, best_f1 = self.beta_grid[0], -1.0
            for b in self.beta_grid:
                rules, counts = self._induce(Xt_d, yt, b)
                pred = self._predict_from_rules(Xv, rules, counts)
                f1 = f1_score(yv, pred, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_beta, best_f1 = b, f1
            self.beta_ = best_beta
            print(f"    [N1] Adaptive beta selected: {self.beta_} "
                  f"(internal macro-F1 = {best_f1:.3f})")
        else:
            self.beta_ = float(self.beta)

        # Re-induce rules on ALL training data with the selected beta.
        self.decision_rules_, self.class_counts_ = self._induce(Xd, y, self.beta_)
        return self

    # ---------------------------- predict ----------------------------
    def _class_scores(self, X, rules, counts):
        """[N2] Fuzzy rule matching: score(class) = sum over rules of
        match_degree * confidence * sqrt(support), where match_degree is
        the PRODUCT (t-norm) of the sample's fuzzy memberships to each
        condition bin. Every sample gets a continuous score for every
        class, so no fallback branch is needed."""
        X = np.asarray(X, dtype=float)
        memberships = self._fuzzy_memberships(X)
        n = X.shape[0]
        classes = sorted(counts.keys())
        S = np.zeros((n, len(classes)))
        for ci, cls in enumerate(classes):
            for rule in rules[cls]:
                deg = np.ones(n)
                for attr_idx, val in rule['condition']:
                    deg *= memberships[attr_idx][:, int(val)]
                S[:, ci] += deg * rule['confidence'] * np.sqrt(rule['support'])
        # Prior-based tiny tiebreak toward the majority class when all
        # scores are exactly zero (can only happen if a class has 0 rules).
        priors = np.array([counts[c] for c in classes], dtype=float)
        S += 1e-9 * priors / priors.sum()
        return S, classes

    def _predict_from_rules(self, X, rules, counts):
        S, classes = self._class_scores(X, rules, counts)
        return np.array([classes[i] for i in S.argmax(axis=1)])

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._predict_from_rules(X, self.decision_rules_, self.class_counts_)

    def predict_proba(self, X):
        """Normalized fuzzy scores; required by the conformal layer [N3]."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        S, _ = self._class_scores(X, self.decision_rules_, self.class_counts_)
        row_sum=S.sum(axis=1,keepdims=True)
        row_sum[row_sum==0]=1.0
        return S/row_sum

    # ----------------------- interpretability -----------------------
    def export_rules(self, feature_names, class_names=None, top_k=5):
        """[N6] Human-readable rules with PHYSICAL bin intervals.

        WHY: this is what a materials-science reader actually uses — each
        rule is a directly testable mechanistic hypothesis (e.g., high BET
        + low O/C -> High capacity), which strengthens the Discussion
        section far more than abstract rule indices in z-score space.
        """
        lines = []
        edges = self.feature_discretizer_.bin_edges_
        for cls, rules in self.decision_rules_.items():
            label = class_names.get(cls, cls) if class_names else cls
            lines.append(f"\n--- Top rules for class: {label} "
                         f"(beta = {self.beta_}) ---")
            for rk, rule in enumerate(rules[:top_k], 1):
                conds = []
                for attr_idx, b in rule['condition']:
                    lo, hi = edges[attr_idx][int(b)], edges[attr_idx][int(b) + 1]
                    conds.append(f"{feature_names[attr_idx]} in "
                                 f"[{lo:.3g}, {hi:.3g}]")
                lines.append(
                    f"  R{rk}: IF " + " AND ".join(conds) +
                    f"  => {label}  "
                    f"(conf={rule['confidence']:.2f}, "
                    f"supp={rule['support']:.3f}, cov={rule['coverage']:.2f})")
        return "\n".join(lines)


# ============================================================================
# [N3] SPLIT-CONFORMAL WRAPPER
# ============================================================================
class ConformalRSML:
    """Split-conformal prediction sets on top of any scorer with
    predict_proba.

    WHY THIS IS THE HEADLINE NOVELTY: rough sets describe uncertainty
    qualitatively via the boundary region; conformal prediction makes that
    quantitative with a finite-sample guarantee P(y in set) >= 1 - alpha,
    assuming exchangeability. The output "this hydrochar is {Medium, High}"
    is the statistically rigorous analogue of a rough-set boundary object —
    a one-to-one conceptual mapping that gives the paper a clear, defensible
    contribution beyond yet-another-classifier.
    """

    def __init__(self, base_model, alpha=0.10, cal_fraction=0.25,
                 random_state=RNG_SEED):
        self.base_model = base_model
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        # Proper-train / calibration split (calibration never seen by rules).
        try:
            Xt, Xc, yt, yc = train_test_split(
                X, y, test_size=self.cal_fraction,
                random_state=self.random_state, stratify=y)
        except ValueError:
            Xt, Xc, yt, yc = train_test_split(
                X, y, test_size=self.cal_fraction,
                random_state=self.random_state, stratify=None)
        self.model_ = clone(self.base_model).fit(Xt, yt)
        # Nonconformity = 1 - normalized score of the true class.
        proba = self.model_.predict_proba(Xc)
        cls_index = {c: i for i, c in enumerate(sorted(np.unique(yt)))}
        alpha_scores = 1.0 - proba[np.arange(len(yc)),
                                   [cls_index[c] for c in yc]]
        n = len(alpha_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat_ = np.quantile(alpha_scores, min(q_level, 1.0),
                                 method='higher')
        self.classes_ = sorted(cls_index.keys())
        return self

    def predict_set(self, X):
        proba = self.model_.predict_proba(np.asarray(X, dtype=float))
        sets = []
        for row in proba:
            s = [c for i, c in enumerate(self.classes_)
                 if 1.0 - row[i] <= self.qhat_]
            if not s:  # guarantee non-empty sets
                s = [self.classes_[int(np.argmax(row))]]
            sets.append(s)
        return sets

    def predict(self, X):
        return self.model_.predict(X)


# ============================================================================
# STEP 5: TRAINING, POINT PREDICTION, CONFORMAL SETS
# ============================================================================
print(f"\n[STEP 5] Training FC-VPRS-ML ...")

base = FuzzyConformalRSML(
    n_bins_attributes=4,
    min_rule_confidence=0.80,
    min_rule_support=0.04,
    max_rules_per_class=50,
    beta='auto'          # [N1] data-driven instead of hand-set 0.92
)

conformal = ConformalRSML(base, alpha=0.10)   # 90% target coverage [N3]
conformal.fit(X_train.values, y_train_classes)

y_pred = conformal.predict(X_test.values)
pred_sets = conformal.predict_set(X_test.values)

# ---------------------------- point metrics ----------------------------
acc = accuracy_score(y_test_classes, y_pred)
prec = precision_score(y_test_classes, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test_classes, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test_classes, y_pred, average='weighted', zero_division=0)

# [N4] Ordinal-aware metrics: capacity classes are ORDERED, so report
# quadratic weighted kappa and ordinal MAE alongside standard metrics.
qwk = cohen_kappa_score(y_test_classes, y_pred, weights='quadratic')
ord_mae = float(np.mean(np.abs(y_test_classes - y_pred)))

# [N3] Conformal diagnostics: empirical coverage should be >= 1 - alpha;
# average set size measures informativeness (smaller = sharper model).
coverage = float(np.mean([yt in s for yt, s in zip(y_test_classes, pred_sets)]))
avg_set_size = float(np.mean([len(s) for s in pred_sets]))

cm = confusion_matrix(y_test_classes, y_pred)

print(f"\n" + "=" * 60)
print("FINAL METRICS REPORT (test set)")
print("=" * 60)
print(f"  Accuracy                 : {acc:.4f}")
print(f"  Weighted precision       : {prec:.4f}")
print(f"  Weighted recall          : {rec:.4f}")
print(f"  Weighted F1              : {f1:.4f}")
print(f"  [N4] Quadratic wtd kappa : {qwk:.4f}")
print(f"  [N4] Ordinal MAE (bins)  : {ord_mae:.4f}")
print(f"  [N3] Conformal coverage  : {coverage:.4f} (target >= 0.90)")
print(f"  [N3] Avg prediction-set size: {avg_set_size:.2f} / 3 classes")
print("=" * 60)

print("\nCONFUSION MATRIX:")
print(pd.DataFrame(cm,
                   index=[f'Actual {CLASS_NAMES[i]}' for i in range(cm.shape[0])],
                   columns=[f'Pred {CLASS_NAMES[i]}' for i in range(cm.shape[1])]))

# ------------------- [N6] interpretable rule export -------------------
print("\n[N6] PHYSICALLY INTERPRETABLE DECISION RULES (original units):")
print(conformal.model_.export_rules(conditional_attributes,
                                    class_names=CLASS_NAMES, top_k=5))

# ============================================================================
# STEP 6: [N5] REPEATED STRATIFIED CV + BASELINE BENCHMARK
# ============================================================================
# WHY: a single split on a small dataset is statistically fragile; repeated
# stratified K-fold reports mean +/- std, and interpretable/black-box
# baselines quantify the interpretability-accuracy trade-off that the
# Discussion section must address.
print("\n[STEP 6] Repeated stratified 5-fold CV (2 repeats) benchmark ...")

models = {
    'FC-VPRS-ML (ours)': FuzzyConformalRSML(beta='auto',
                                            max_rules_per_class=50),
    'VPRS fixed beta=0.92': FuzzyConformalRSML(beta=0.92,
                                               max_rules_per_class=50),
    'Decision Tree': DecisionTreeClassifier(max_depth=4,
                                            random_state=RNG_SEED),
    'Random Forest': RandomForestClassifier(n_estimators=300,
                                            random_state=RNG_SEED),
}

# Labels for CV are built once from the full-data quantile key used only
# for fold stratification; per-fold labels are re-derived inside the loop
# from a discretizer fit on that fold's training targets (leakage-safe).
from collections import Counter

X_all = X.values.astype(float)
y_all = y.values.astype(float)
strat_all = pd.qcut(y, q=3, labels=False, duplicates='drop').values

min_class = min(Counter(strat_all).values())
n_splits = max(2, min(5, min_class))

print(f"Using {n_splits}-fold repeated stratified CV")

cv = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=2,
    random_state=RNG_SEED
)

results = {name: {'f1': [], 'qwk': []} for name in models}

for fold_i, (tr, te) in enumerate(cv.split(X_all, strat_all)):
    dy = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans',
                          subsample=None)
    ytr = dy.fit_transform(y_all[tr].reshape(-1, 1)).flatten().astype(int)
    yte = dy.transform(y_all[te].reshape(-1, 1)).flatten().astype(int)
    for name, mdl in models.items():
        m = clone(mdl)
        m.fit(X_all[tr], ytr)
        p = m.predict(X_all[te])
        results[name]['f1'].append(
            f1_score(yte, p, average='weighted', zero_division=0))
        results[name]['qwk'].append(
            cohen_kappa_score(yte, p, weights='quadratic'))

print("\nBENCHMARK (mean +/- std over 10 folds):")
print(f"{'Model':<26}{'Weighted F1':<20}{'QWK':<20}")
for name, res in results.items():
    print(f"{name:<26}"
          f"{np.mean(res['f1']):.3f} +/- {np.std(res['f1']):.3f}      "
          f"{np.mean(res['qwk']):.3f} +/- {np.std(res['qwk']):.3f}")

print("\nDone. Report point metrics, conformal coverage/set size, the rule "
      "table, and the CV benchmark together — this is the full evidence "
      "package for the novelty claims [N1]-[N6].")
