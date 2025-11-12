"""
Position Sizing Example Script

Demonstrates how to use Kelly Criterion position sizing with LSTM predictions.
Shows comparison between fixed position sizing and Kelly-based dynamic sizing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.position_sizing.kelly import KellyCriterion
from src.strategies.position_sizing.lstm_integration import LSTMPositionSizer
from src.strategies.position_sizing.config import (
    get_conservative_config,
    get_moderate_config,
    PositionSizingConfig
)
from src.strategies.position_sizing.utils import (
    calculate_sharpe_ratio,
    calculate_max_drawdown
)


def generate_sample_data(n_samples=500):
    """
    Generate sample LSTM predictions and actual returns for demonstration

    Returns:
        dict: Contains predictions, actual returns, and confidence scores
    """
    np.random.seed(42)

    # Simulate actual returns (with some drift and volatility)
    actual_returns = np.random.normal(0.001, 0.02, n_samples)

    # Simulate LSTM predictions (correlated with actual, but noisy)
    # Higher skill = higher correlation
    skill = 0.60  # 60% of prediction is signal, 40% is noise
    predictions = skill * actual_returns + (1 - skill) * np.random.normal(0, 0.015, n_samples)

    # Simulate confidence scores (higher when predictions are stronger)
    confidence = np.clip(
        0.5 + 0.3 * np.abs(predictions) / np.std(predictions),
        0.3,
        0.95
    )

    return {
        'predictions': predictions,
        'actual_returns': actual_returns,
        'confidence': confidence
    }


def example_1_basic_kelly():
    """Example 1: Basic Kelly Criterion usage"""

    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Kelly Criterion")
    print("="*70)

    # Initialize Kelly calculator
    kelly = KellyCriterion(kelly_fraction=0.25)  # Quarter Kelly

    # Example: 55% win rate, average wins are 1.5x average losses
    win_probability = 0.55
    win_loss_ratio = 1.5

    position_size = kelly.calculate_kelly_fraction(win_probability, win_loss_ratio)

    print(f"\nGiven:")
    print(f"  Win Probability: {win_probability:.1%}")
    print(f"  Win/Loss Ratio: {win_loss_ratio:.2f}")
    print(f"  Kelly Fraction: {kelly.kelly_fraction} (Quarter Kelly)")

    print(f"\nOptimal Position Size: {position_size:.2%} of capital")

    # Show effect of different Kelly fractions
    print(f"\nComparison of Kelly Fractions:")
    for frac in [0.25, 0.50, 1.00]:
        k = KellyCriterion(kelly_fraction=frac)
        size = k.calculate_kelly_fraction(win_probability, win_loss_ratio)
        print(f"  {frac:.2f} Kelly → {size:.2%} position size")


def example_2_lstm_integration():
    """Example 2: LSTM Integration with Confidence Scaling"""

    print("\n" + "="*70)
    print("EXAMPLE 2: LSTM Integration with Confidence Scaling")
    print("="*70)

    # Generate sample data
    data = generate_sample_data(n_samples=500)

    # Split into calibration and test sets
    calibration_size = 300
    hist_preds = data['predictions'][:calibration_size]
    hist_returns = data['actual_returns'][:calibration_size]
    test_preds = data['predictions'][calibration_size:]
    test_returns = data['actual_returns'][calibration_size:]
    test_confidence = data['confidence'][calibration_size:]

    # Initialize position sizer
    config = get_conservative_config()
    sizer = LSTMPositionSizer(
        config=config,
        historical_predictions=hist_preds,
        historical_returns=hist_returns
    )

    # Show calibration results
    summary = sizer.get_sizing_summary()
    print(f"\nCalibrated Parameters:")
    print(f"  Win Probability: {summary['win_probability']:.1%}")
    print(f"  Win/Loss Ratio: {summary['win_loss_ratio']:.2f}")

    # Size positions for test set
    print(f"\nExample Position Sizes:")
    for i in range(min(5, len(test_preds))):
        size = sizer.size_position(
            prediction=test_preds[i],
            confidence=test_confidence[i],
            recent_returns=hist_returns[-60:]  # Last 60 days for volatility
        )
        print(f"  Prediction: {test_preds[i]:+.2%}, "
              f"Confidence: {test_confidence[i]:.1%} → "
              f"Position: {size:.1%}")


def example_3_backtest_comparison():
    """Example 3: Backtest - Fixed vs Kelly Sizing"""

    print("\n" + "="*70)
    print("EXAMPLE 3: Backtest Comparison - Fixed vs Kelly Position Sizing")
    print("="*70)

    # Generate longer sample data
    data = generate_sample_data(n_samples=1000)

    # Split data
    calibration_size = 300
    hist_preds = data['predictions'][:calibration_size]
    hist_returns = data['actual_returns'][:calibration_size]
    test_preds = data['predictions'][calibration_size:]
    test_returns = data['actual_returns'][calibration_size:]
    test_confidence = data['confidence'][calibration_size:]

    # Initialize position sizers
    config = PositionSizingConfig(kelly_fraction=0.25)
    kelly_sizer = LSTMPositionSizer(
        config=config,
        historical_predictions=hist_preds,
        historical_returns=hist_returns
    )

    # Fixed position size (15% always)
    fixed_position = 0.15

    # Run backtest
    print("\nRunning backtest...")

    fixed_equity = [1.0]  # Start with $1
    kelly_equity = [1.0]

    for i in range(len(test_preds)):
        # Fixed sizing
        fixed_return = fixed_position * np.sign(test_preds[i]) * test_returns[i]
        fixed_equity.append(fixed_equity[-1] * (1 + fixed_return))

        # Kelly sizing
        kelly_size = kelly_sizer.size_position(
            prediction=test_preds[i],
            confidence=test_confidence[i],
            recent_returns=np.concatenate([hist_returns[-60:], test_returns[:i]])[-60:]
        )
        kelly_return = kelly_size * np.sign(test_preds[i]) * test_returns[i]
        kelly_equity.append(kelly_equity[-1] * (1 + kelly_return))

    # Calculate metrics
    fixed_returns = np.diff(fixed_equity) / fixed_equity[:-1]
    kelly_returns = np.diff(kelly_equity) / kelly_equity[:-1]

    fixed_sharpe = calculate_sharpe_ratio(fixed_returns)
    kelly_sharpe = calculate_sharpe_ratio(kelly_returns)

    fixed_mdd = calculate_max_drawdown(fixed_returns)
    kelly_mdd = calculate_max_drawdown(kelly_returns)

    # Print results
    print(f"\nBacktest Results:")
    print(f"\n{'Strategy':<20} {'Final Value':<15} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 55)
    print(f"{'Fixed (15%)':<20} ${fixed_equity[-1]:.2f}{'':<10} "
          f"{fixed_sharpe:>6.2f}{'':<4} {fixed_mdd:>6.1%}")
    print(f"{'Kelly (Dynamic)':<20} ${kelly_equity[-1]:.2f}{'':<10} "
          f"{kelly_sharpe:>6.2f}{'':<4} {kelly_mdd:>6.1%}")

    improvement = ((kelly_equity[-1] / fixed_equity[-1]) - 1) * 100
    print(f"\nKelly Improvement: {improvement:+.1f}%")

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fixed_equity, label='Fixed 15%', linewidth=2)
    plt.plot(kelly_equity, label='Kelly Dynamic', linewidth=2)
    plt.title('Equity Curves')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Calculate drawdowns
    fixed_cummax = np.maximum.accumulate(fixed_equity)
    fixed_dd = [(e - cm) / cm for e, cm in zip(fixed_equity, fixed_cummax)]

    kelly_cummax = np.maximum.accumulate(kelly_equity)
    kelly_dd = [(e - cm) / cm for e, cm in zip(kelly_equity, kelly_cummax)]

    plt.plot(fixed_dd, label='Fixed 15%', linewidth=2)
    plt.plot(kelly_dd, label='Kelly Dynamic', linewidth=2)
    plt.title('Drawdown')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent.parent / "models" / "position_sizing_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {save_path}")
    plt.show()


def example_4_volatility_regimes():
    """Example 4: Position Sizing Across Volatility Regimes"""

    print("\n" + "="*70)
    print("EXAMPLE 4: Position Sizing Across Volatility Regimes")
    print("="*70)

    from src.strategies.position_sizing.utils import detect_volatility_regime, get_regime_scaling_factor

    # Simulate different volatility scenarios
    low_vol_returns = np.random.normal(0.001, 0.01, 60)  # 10% annual vol
    medium_vol_returns = np.random.normal(0.001, 0.02, 60)  # 20% annual vol
    high_vol_returns = np.random.normal(0.001, 0.04, 60)  # 40% annual vol

    scenarios = {
        'Low Volatility': low_vol_returns,
        'Medium Volatility': medium_vol_returns,
        'High Volatility': high_vol_returns
    }

    # Initialize sizer
    config = PositionSizingConfig()
    sizer = LSTMPositionSizer(config=config)

    print(f"\nBase Position Size (before regime adjustment): 10.0%")
    print(f"\nPosition Sizing Across Regimes:")
    print(f"{'Scenario':<20} {'Annualized Vol':<18} {'Regime':<12} {'Position Size':<15}")
    print("-" * 70)

    for scenario_name, returns in scenarios.items():
        # Calculate volatility
        vol = np.std(returns) * np.sqrt(252)

        # Detect regime
        regime = detect_volatility_regime(vol)
        scaling_factor = get_regime_scaling_factor(regime)

        # Base position
        base_position = 0.10  # 10%

        # Adjusted position
        adjusted_position = base_position * scaling_factor
        adjusted_position = min(adjusted_position, 0.30)  # Cap at 30%

        print(f"{scenario_name:<20} {vol:<18.1%} {regime.value.upper():<12} "
              f"{adjusted_position:<15.1%} (×{scaling_factor:.1f})")


def main():
    """Run all examples"""

    print("\n" + "="*70)
    print(" POSITION SIZING WITH KELLY CRITERION - EXAMPLES")
    print("="*70)

    # Run examples
    example_1_basic_kelly()
    example_2_lstm_integration()
    example_3_backtest_comparison()
    example_4_volatility_regimes()

    print("\n" + "="*70)
    print(" All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
