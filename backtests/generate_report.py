import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from datetime import datetime
import quantstats as qs

# Extend QuantStats to support custom reporting if needed
qs.extend_pandas()

def generate_report(trades_csv, summary_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(trades_csv)
    with open(summary_json, 'r') as f:
        summary = json.load(f)

    if df.empty:
        print("No trades found in CSV.")
        return

    # Convert times
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time')

    # Calculate daily returns for QuantStats
    # We need a series of daily returns.
    # Since we have trade-level data, we'll aggregate by day.
    initial_capital = summary.get('portfolio', {}).get('initial_capital_usdt', 10000)

    # Calculate equity curve
    df['cumulative_pnl'] = df['pnl_usdt'].cumsum()
    df['equity'] = initial_capital + df['cumulative_pnl']

    # Create daily equity series
    daily_equity = df.set_index('exit_time')['equity'].resample('D').last().ffill()

    # Add initial capital at the start if not present
    start_date = daily_equity.index[0] - pd.Timedelta(days=1)
    daily_equity[start_date] = initial_capital
    daily_equity = daily_equity.sort_index()

    daily_returns = daily_equity.pct_change().dropna()

    # 1. QuantStats HTML Report
    print("Generating QuantStats HTML report...")
    report_html = os.path.join(output_dir, 'quantstats_report.html')
    qs.reports.html(daily_returns, output=report_html, title='Strategy Backtest Report')

    # 2. Custom Matplotlib Plots
    print("Generating custom plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(15, 20))

    # A. Equity Curve
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(daily_equity.index, daily_equity.values, label='Equity (USDT)', color='blue')
    ax1.set_title('Cumulative Equity Curve', fontsize=16)
    ax1.set_ylabel('Equity (USDT)')
    ax1.legend()

    # B. Drawdown
    ax2 = plt.subplot(4, 1, 2)
    rolling_max = daily_equity.cummax()
    drawdown = (daily_equity - rolling_max) / rolling_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    ax2.set_title('Drawdown Chart', fontsize=16)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(-1, 0.05)

    # C. Monthly Returns Heatmap (using Seaborn)
    ax3 = plt.subplot(4, 1, 3)
    monthly_ret = qs.stats.monthly_returns(daily_returns) * 100
    sns.heatmap(monthly_ret, annot=True, fmt=".1f", cmap='RdYlGn', center=0, ax=ax3)
    ax3.set_title('Monthly Returns (%)', fontsize=16)

    # D. Symbol Performance
    ax4 = plt.subplot(4, 1, 4)
    symbol_pnl = df.groupby('symbol')['pnl_usdt'].sum().sort_values()
    # Plot top 10 and bottom 10
    top_bottom = pd.concat([symbol_pnl.head(10), symbol_pnl.tail(10)])
    colors = ['red' if x < 0 else 'green' for x in top_bottom.values]
    top_bottom.plot(kind='barh', ax=ax4, color=colors)
    ax4.set_title('Top 10 & Bottom 10 Symbols by PnL', fontsize=16)
    ax4.set_xlabel('Total PnL (USDT)')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'performance_summary.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Saved summary plot to {plot_file}")

    # 3. Additional Analysis: Long vs Short
    ls_perf = df.groupby('side')['pnl_usdt'].agg(['sum', 'count', lambda x: (x > 0).mean()])
    ls_perf.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    print("\nLong vs Short Performance:")
    print(ls_perf)

    ls_perf.to_csv(os.path.join(output_dir, 'long_short_performance.csv'))

    print(f"\nReport generation complete. Files are in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", default="artifacts/backtests/full/trades.csv")
    parser.add_argument("--summary", default="artifacts/backtests/full/summary.json")
    parser.add_argument("--out", default="artifacts/reports/latest")
    args = parser.parse_args()

    generate_report(args.trades, args.summary, args.out)
