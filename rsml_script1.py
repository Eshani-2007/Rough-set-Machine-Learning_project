"""
OPTIMIZED VARIABLE PRECISION ROUGH SET MACHINE LEARNING (VPRS-ML)
====================================================================
Includes Hamming Distance Fallback & Support-Weighted Rule Scoring
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score,
    recall_score
)
from itertools import combinations
import random

# Reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# STEP 1 & 2: DATA LOADING & PREPARATION
# ============================================================================
print("="*80)
print("OPTIMIZED VARIABLE PRECISION ROUGH SET MACHINE LEARNING")
print("="*80)

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
# STEP 3 & 4: DISCRETIZATION & SCALING
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

discretizer_y = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans', subsample=None)
y_train_classes = discretizer_y.fit_transform(y_train.values.reshape(-1, 1)).flatten().astype(int)
y_test_classes = discretizer_y.transform(y_test.values.reshape(-1, 1)).flatten().astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=conditional_attributes)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=conditional_attributes)

# ============================================================================
# PROFESSIONAL ROUGH SET MACHINE LEARNING CLASSIFIER
# ============================================================================

class ProfessionalRSML(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_bins_attributes=4, min_rule_confidence=0.80, 
                 min_rule_support=0.04, max_rules_per_class=40, beta=0.92):
        self.n_bins_attributes = n_bins_attributes
        self.min_rule_confidence = min_rule_confidence
        self.min_rule_support = min_rule_support
        self.max_rules_per_class = max_rules_per_class
        self.beta = beta 
        
        self.decision_rules = {}
        self.class_counts = {}
        self.feature_discretizer = None
        
    def _discretize_features(self, X, fit=True):
        if fit:
            self.feature_discretizer = KBinsDiscretizer(
                n_bins=self.n_bins_attributes, encode='ordinal', 
                strategy='kmeans', subsample=None
            )
            return self.feature_discretizer.fit_transform(X).astype(int)
        else:
            return self.feature_discretizer.transform(X).astype(int)
    
    def _compute_indiscernibility_classes(self, X_discrete, attribute_indices):
        if len(attribute_indices) == 0:
            return {(): list(range(X_discrete.shape[0]))}
        equivalence_classes = {}
        for sample_idx in range(X_discrete.shape[0]):
            signature = tuple(X_discrete[sample_idx, attr_idx] for attr_idx in attribute_indices)
            if signature not in equivalence_classes:
                equivalence_classes[signature] = []
            equivalence_classes[signature].append(sample_idx)
        return equivalence_classes
    
    def _compute_lower_approximation(self, X_discrete, y, decision_class, attribute_indices):
        ind_classes = self._compute_indiscernibility_classes(X_discrete, attribute_indices)
        lower_approx = set()
        for equiv_class_objects in ind_classes.values():
            matches = sum(1 for obj_idx in equiv_class_objects if y[obj_idx] == decision_class)
            if (matches / len(equiv_class_objects)) >= self.beta:
                lower_approx.update(equiv_class_objects)
        return lower_approx
    
    def _compute_upper_approximation(self, X_discrete, y, decision_class, attribute_indices):
        ind_classes = self._compute_indiscernibility_classes(X_discrete, attribute_indices)
        upper_approx = set()
        for equiv_class_objects in ind_classes.values():
            if any(y[obj_idx] == decision_class for obj_idx in equiv_class_objects):
                upper_approx.update(equiv_class_objects)
        return upper_approx
    
    def _compute_approximation_strength(self, lower_approx, upper_approx):
        if len(upper_approx) == 0: return 0.0
        return len(lower_approx) / len(upper_approx)
    
    def _compute_core(self, X_discrete, y, decision_class):
        n_features = X_discrete.shape[1]
        all_attrs = set(range(n_features))
        all_lower = self._compute_lower_approximation(X_discrete, y, decision_class, list(all_attrs))
        all_upper = self._compute_upper_approximation(X_discrete, y, decision_class, list(all_attrs))
        all_strength = self._compute_approximation_strength(all_lower, all_upper)
        
        core = set()
        for attr in all_attrs:
            reduced_attrs = list(all_attrs - {attr})
            if len(reduced_attrs) == 0:
                core.add(attr)
                continue
            reduced_lower = self._compute_lower_approximation(X_discrete, y, decision_class, reduced_attrs)
            reduced_upper = self._compute_upper_approximation(X_discrete, y, decision_class, reduced_attrs)
            reduced_strength = self._compute_approximation_strength(reduced_lower, reduced_upper)
            if reduced_strength < all_strength:
                core.add(attr)
        return core
    
    def _compute_reduct_greedy(self, X_discrete, y, decision_class):
        n_features = X_discrete.shape[1]
        all_attrs = set(range(n_features))
        core = self._compute_core(X_discrete, y, decision_class)
        reduct = core.copy()
        
        target_lower = self._compute_lower_approximation(X_discrete, y, decision_class, list(all_attrs))
        target_upper = self._compute_upper_approximation(X_discrete, y, decision_class, list(all_attrs))
        target_strength = self._compute_approximation_strength(target_lower, target_upper)
        
        remaining = all_attrs - reduct
        while remaining and len(reduct) < n_features:
            best_attr = None
            best_strength = self._compute_approximation_strength(
                self._compute_lower_approximation(X_discrete, y, decision_class, list(reduct)),
                self._compute_upper_approximation(X_discrete, y, decision_class, list(reduct))
            )
            for attr in remaining:
                test_reduct = reduct | {attr}
                test_lower = self._compute_lower_approximation(X_discrete, y, decision_class, list(test_reduct))
                test_upper = self._compute_upper_approximation(X_discrete, y, decision_class, list(test_reduct))
                test_strength = self._compute_approximation_strength(test_lower, test_upper)
                
                if test_strength > best_strength:
                    best_strength = test_strength
                    best_attr = attr
            if best_attr is None or best_strength >= target_strength: break
            reduct.add(best_attr)
            remaining.remove(best_attr)
        return reduct
    
    def _generate_decision_rules(self, X_discrete, y, decision_class, reduct):
        rules = []
        class_size = np.sum(y == decision_class)
        total_size = len(y)
        reduct_attrs = sorted(list(reduct))
        if len(reduct_attrs) == 0: return rules
            
        max_r = min(3, len(reduct_attrs)) 
        for r in range(1, max_r + 1):
            for condition_attrs in combinations(reduct_attrs, r):
                unique_combinations = np.unique(X_discrete[:, condition_attrs], axis=0)
                for vals in unique_combinations:
                    attr_matches = np.ones(X_discrete.shape[0], dtype=bool)
                    for i, attr_idx in enumerate(condition_attrs):
                        attr_matches &= (X_discrete[:, attr_idx] == vals[i])
                        
                    class_matches = (y == decision_class)
                    matches_total = np.sum(attr_matches)
                    matches_class = np.sum(attr_matches & class_matches)
                    
                    if matches_total == 0: continue
                        
                    confidence = matches_class / matches_total
                    support = matches_class / total_size
                    coverage = matches_class / class_size if class_size > 0 else 0
                    quality = (confidence * 0.6 + support * 0.2 + coverage * 0.2)
                    
                    if (confidence >= self.min_rule_confidence and support >= self.min_rule_support):
                        condition_dict = tuple(zip(condition_attrs, vals))
                        rules.append({
                            'condition': condition_dict,
                            'confidence': confidence,
                            'support': support,
                            'coverage': coverage,
                            'quality': quality
                        })
        rules.sort(key=lambda r: r['quality'], reverse=True)
        return rules[:self.max_rules_per_class]
    
    def fit(self, X, y):
        print("    Computing Information System and Rule Matrices...")
        X_discrete = self._discretize_features(X, fit=True)
        unique_classes = np.unique(y)
        for decision_class in unique_classes:
            reduct = self._compute_reduct_greedy(X_discrete, y, decision_class)
            rules = self._generate_decision_rules(X_discrete, y, decision_class, reduct)
            self.decision_rules[decision_class] = rules
            self.class_counts[decision_class] = np.sum(y == decision_class)
        return self
    
    def predict(self, X):
        X_discrete = self._discretize_features(X.values if isinstance(X, pd.DataFrame) else X, fit=False)
        predictions = []
        majority_class = max(self.class_counts.keys(), key=lambda x: self.class_counts[x])
        
        for sample_idx in range(X_discrete.shape[0]):
            sample = X_discrete[sample_idx]
            class_scores = {}
            matched_any = False
            
            # 1. Standard Rule Matching
            for decision_class, rules in self.decision_rules.items():
                score = 0
                for rule in rules:
                    match = True
                    for attr_idx, val in rule['condition']:
                        if sample[attr_idx] != val:
                            match = False
                            break
                    if match:
                        # Mathematical Upgrade: Support Weighting
                        score += rule['confidence'] * np.sqrt(rule['support'])
                        matched_any = True
                class_scores[decision_class] = score
            
            # 2. Hamming Distance Fallback for Unclassified Samples
            if not matched_any:
                best_dist = float('inf')
                best_fallback_class = majority_class
                
                for decision_class, rules in self.decision_rules.items():
                    for rule in rules:
                        dist = 0
                        for attr_idx, val in rule['condition']:
                            if sample[attr_idx] != val:
                                dist += 1
                        
                        # Tie-breaker using rule quality
                        adjusted_dist = dist - (rule['quality'] * 0.1)
                        if adjusted_dist < best_dist:
                            best_dist = adjusted_dist
                            best_fallback_class = decision_class
                
                predictions.append(best_fallback_class)
            else:
                predictions.append(max(class_scores, key=class_scores.get))
                
        return np.array(predictions)

# ============================================================================
# STEP 5: TRAINING AND EVALUATION
# ============================================================================
print(f"\n[STEP 5] Training and Test Set Evaluation...")

prsml = ProfessionalRSML(
    n_bins_attributes=4, # Increased feature resolution
    min_rule_confidence=0.80, # Very strict confidence
    min_rule_support=0.04,
    max_rules_per_class=50,
    beta=0.92 # Extremely tight VPRS boundary
)

prsml.fit(X_train_scaled_df, y_train_classes)
y_pred_rsml = prsml.predict(X_test_scaled_df)

acc_rsml = accuracy_score(y_test_classes, y_pred_rsml)
precision_rsml = precision_score(y_test_classes, y_pred_rsml, average='weighted', zero_division=0)
recall_rsml = recall_score(y_test_classes, y_pred_rsml, average='weighted', zero_division=0)
f1_rsml = f1_score(y_test_classes, y_pred_rsml, average='weighted', zero_division=0)
cm = confusion_matrix(y_test_classes, y_pred_rsml)

print(f"\n" + "="*50)
print(f"FINAL METRICS REPORT")
print(f"="*50)
print(f"  ✓ Accuracy:  {acc_rsml:.4f} ({acc_rsml*100:.2f}%)")
print(f"  ✓ Precision: {precision_rsml:.4f}  <-- Significant increase expected")
print(f"  ✓ Recall:    {recall_rsml:.4f}")
print(f"  ✓ F1-Score:  {f1_rsml:.4f}")
print(f"="*50)

print(f"\nCONFUSION MATRIX:")
print("Predicted ->")
print(pd.DataFrame(cm, 
                   index=[f'Actual Class {i}' for i in range(cm.shape[0])],
                   columns=[f'Pred Class {i}' for i in range(cm.shape[1])]))