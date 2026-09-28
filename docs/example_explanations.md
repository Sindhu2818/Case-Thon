# Example patient explanations (temporal validation episodes)

Language is deliberately associational: factors are the ones the model weighed most for this episode; they are not causal claims.

### Episode E0001371 (patient P000877)
- Predicted risk: **74%** (dropout)
- Likely failure stage: **Medicine not collected** - MATCH vs true stage 'Medicine not collected'
- Associated factors (SHAP, not causal): `dist_x_medicine`=36.10 (pushes risk higher); `n_advised`=3.00 (pushes risk higher); `distance_km`=36.10 (pushes risk higher); `test_advised`=1.00 (pushes risk higher)
- Validated outcome: actual=dropout -> correct

### Episode E0003945 (patient P002482)
- Predicted risk: **71%** (dropout)
- Likely failure stage: **Medicine not collected**
- Associated factors (SHAP, not causal): `dist_x_medicine`=36.50 (pushes risk higher); `distance_km`=36.50 (pushes risk higher); `n_advised`=2.00 (pushes risk higher); `review_advised`=1.00 (pushes risk higher)
- Validated outcome: actual=completed -> false alarm

### Episode E0003843 (patient P002416)
- Predicted risk: **70%** (dropout)
- Likely failure stage: **Test not completed** - miss vs true stage 'Medicine not collected'
- Associated factors (SHAP, not causal): `dist_x_medicine`=33.30 (pushes risk higher); `n_advised`=3.00 (pushes risk higher); `distance_km`=33.30 (pushes risk higher); `test_advised`=1.00 (pushes risk higher)
- Validated outcome: actual=dropout -> correct

### Episode E0000420 (patient P000241)
- Predicted risk: **69%** (dropout)
- Likely failure stage: **Review not attended** - miss vs true stage 'Medicine not collected'
- Associated factors (SHAP, not causal): `n_advised`=3.00 (pushes risk higher); `dist_x_medicine`=22.90 (pushes risk higher); `test_advised`=1.00 (pushes risk higher); `distance_km`=22.90 (pushes risk higher)
- Validated outcome: actual=dropout -> correct

### Episode E0002971 (patient P001885)
- Predicted risk: **49%** (likely complete)
- Likely failure stage: **Completed care journey** - miss vs true stage 'Medicine not collected'
- Associated factors (SHAP, not causal): `n_advised`=1.00 (pushes risk lower); `dist_x_medicine`=28.20 (pushes risk higher); `distance_km`=28.20 (pushes risk higher); `review_advised`=0.00 (pushes risk lower)
- Validated outcome: actual=dropout -> missed dropout (false negative)
