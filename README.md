# DDA-Neural-Sentinel: Predictive Player Retention via Dynamic Difficulty Adjustment

## Executive Summary
This project predicts mobile puzzle game "Rage Quits" to trigger Dynamic Difficulty Adjustment (DDA). The core objective was building a robust, production-ready pipeline capable of handling highly imbalanced, chaotic telemetry data (3 million+ rows). 

Initially, a standard Deep Learning architecture (TensorFlow) using SMOTE achieved 95%+ accuracy. However, rigorous evaluation revealed this was a mathematical illusion. This repository documents the teardown of that failing model and the architectural pivot to an XGBoost pipeline optimized strictly for Precision-Recall AUC (PR-AUC).

## The "Fake 95%" Trap & Why the Initial Model Failed
Standard classification metrics are practically useless for highly imbalanced mobile game events. 

+ **The Illusion:** The original neural network achieved 97.7% accuracy by naturally biasing toward the majority class ("No Quit"). It simply predicted the player would continue playing every single time. 
+ **The SMOTE Failure:** To fix the imbalance, SMOTE was applied. However, on messy tabular sequence data, SMOTE generated overlapping synthetic noise. The model memorized this noise rather than learning true mathematical boundaries.
+ **The Real-World Reality:** When subjected to simulated production chaos (missing `LvlA_user_end` events, out-of-order logs, and network duplicates across the TLC), the Deep Learning model collapsed. Validation PR-AUC dropped to near-zero, and Recall fell below 5%. It was acting as a fire alarm that never rang.

## Methodology & The Academic Upgrade

### 1. Telemetry Cleansing & Signal Extraction
Real-world player drop-offs are not clean. Missing end-level API calls were explicitly reclassified as `timeout_crashes` rather than simply dropped via `.dropna()`. Drop percentages (Level Drop %, Interruption Drop %) were calculated using adjusted denominators to mathematically prevent ZeroDivisionErrors on orphaned sessions.

### 2. Psychological Feature Engineering
Raw cumulative counts (`cum_levels_played`) failed to capture player frustration. The feature matrix was rebuilt to include psychological proxies:
+ `rolling_rage_3`: Localized failure clusters measuring immediate fail streaks.
+ `action_velocity`: A time-series calculation detecting frantic, rapid inputs indicative of player "tilt."

### 3. The Algorithm Pivot (XGBoost vs. Deep Learning)
Deep Learning was abandoned. Neural Networks structurally underperform on small, noisy tabular data. The architecture was pivoted to **XGBoost**:
+ **Tree-Based Dominance:** Decision trees natively slice through tabular noise without memorizing it.
+ **Mathematical Imbalance Handling:** Instead of synthesizing fake data with SMOTE, XGBoost's native `scale_pos_weight` was used. The algorithm is mathematically penalized 50x harder for missing a true positive, forcing it to detect the minority class based on pure data integrity.

## Final Evaluation Metrics
The model is evaluated strictly on PR-AUC to satisfy rigorous statistical standards, completely ignoring generic accuracy.

+ **PR-AUC:** `0.7560` (Strong predictive power for severe class imbalance).
+ **Recall (Class 1):** `0.92` (Successfully catches 92% of players *before* they rage quit, allowing DDA to trigger).
+ **Precision (Class 1):** `0.80` (80% of DDA triggers are necessary, protecting the game's economy from false adjustments).

## Conclusion
This pipeline proves that advanced feature engineering and strict metric selection (PR-AUC) heavily outweigh complex, black-box Deep Learning architectures when deploying real-time predictive models in the gaming industry.
