# ML Score Calculation Explanation

## Overview
The ML score is calculated differently for **buy analysis** vs **sell analysis** to reflect the appropriate signal strength for each type of recommendation.

## Buy Analysis (Stock Recommendation)
**Formula:** `ml_score = int(ml_probability * 100)`

**Logic:** Higher ML probability of price increase = Higher buy signal strength

**Example:**
- ML Probability: 0.271 (27.1% chance of price increase)
- ML Score: `int(0.271 * 100) = 27/100`
- Interpretation: Weak buy signal (27/100)

## Sell Analysis (Sell Point Estimation)
**Formula:** `sell_signal_strength = int((1 - ml_probability) * 100)`

**Logic:** Higher probability of price decline = Higher sell signal strength

**Example:**
- ML Probability: 0.271 (27.1% chance of price increase)
- Decline Probability: `1 - 0.271 = 0.729` (72.9% chance of decline)
- Sell Signal Strength: `int(0.729 * 100) = 72/100`
- Interpretation: Strong sell signal (72/100)

## Why This Makes Sense

### Buy Analysis
- We want to buy when there's a high probability of price increase
- ML Probability 0.271 = 27.1% chance of rise = Weak buy signal (27/100)

### Sell Analysis
- We want to sell when there's a high probability of price decline
- ML Probability 0.271 = 27.1% chance of rise = 72.9% chance of decline = Strong sell signal (72/100)

## Display Improvements

### Before (Confusing)
```
ML Probability: 0.271
ML Score: 72.00/100  # Why is this high when probability is low?
```

### After (Clear)
```
ML Probability: 0.271 (27.1% chance of rise)
Sell Signal Strength: 72.00/100 (ML-based)  # 72.9% chance of decline
ML Insights: ML moderately predicts decline (72.9% chance of decline, 27.1% chance of rise)
```

## Summary
- **Buy Analysis:** ML Score directly reflects probability of price increase
- **Sell Analysis:** Sell Signal Strength reflects probability of price decline
- **Both are correct** - they just serve different purposes
- The high sell signal strength (72/100) for low ML probability (27.1%) is **correct** because it indicates a strong probability of decline

## Files Modified
- `chinese_stock_analyzer.py`: Improved comments and variable naming
- `main.py`: Updated display labels for clarity
- `test_ml_score_explanation.py`: Created explanation script
- `ML_SCORE_EXPLANATION.md`: This documentation 