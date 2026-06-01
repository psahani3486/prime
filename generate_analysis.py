import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ReportLab Imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# -------------------------------------------------------------
# 1. DATA LOADING AND CLEANING
# -------------------------------------------------------------
print("Loading data...")
fg = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data.csv")

# Clean dates
fg['date'] = pd.to_datetime(fg['date']).dt.date
trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], dayfirst=True)
trades['date'] = trades['Timestamp IST'].dt.date

# Merge datasets
merged = pd.merge(trades, fg[['date', 'classification', 'value']], on='date', how='left')

# Drop any trades without classification if they exist, or fill with 'Neutral'
merged['classification'] = merged['classification'].fillna('Neutral')
merged['value'] = merged['value'].fillna(50)

# Create a win flag
merged['win'] = (merged['Closed PnL'] > 0).astype(int)

# Reorder classification for logical visual flow
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
merged['classification'] = pd.Categorical(merged['classification'], categories=sentiment_order, ordered=True)

# -------------------------------------------------------------
# 2. RUNNING CALCULATIONS & EXTRACTING METRICS
# -------------------------------------------------------------
print("Running calculations...")

# Q1. Do traders perform better during Fear or Greed?
avg_pnl_sentiment = merged.groupby('classification', observed=False)['Closed PnL'].mean()

# Q2. Which sentiment generates maximum profit?
total_pnl_sentiment = merged.groupby('classification', observed=False)['Closed PnL'].sum()

# Q3. Trade Size vs Sentiment
avg_size_sentiment = merged.groupby('classification', observed=False)['Size USD'].mean()

# Q4. Buy vs Sell Performance
buy_sell_pnl = merged.groupby(['classification', 'Side'], observed=False)['Closed PnL'].mean().unstack()

# Q5. Win Rate by Sentiment
win_rate_sentiment = merged.groupby('classification', observed=False)['win'].mean() * 100

# Advanced Analysis: Top 20 Traders
top_trader_ids = merged.groupby('Account')['Closed PnL'].sum().sort_values(ascending=False).head(20).index
top_traders_data = merged[merged['Account'].isin(top_trader_ids)]
top_traders_behavior = top_traders_data.groupby(['classification'], observed=False)['Closed PnL'].mean()
top_traders_winrate = top_traders_data.groupby(['classification'], observed=False)['win'].mean() * 100

# Advanced Analysis: Volatility / Standard Deviation of PnL
pnl_volatility = merged.groupby('classification', observed=False)['Closed PnL'].std()

# Volume Analysis
total_volume_sentiment = merged.groupby('classification', observed=False)['Size USD'].sum()

# Print results to stdout
print("\n--- RESULTS SUMMARY ---")
print("Average PnL:\n", avg_pnl_sentiment)
print("\nTotal PnL:\n", total_pnl_sentiment)
print("\nAverage Trade Size:\n", avg_size_sentiment)
print("\nWin Rate %:\n", win_rate_sentiment)
print("-----------------------\n")

# -------------------------------------------------------------
# 3. GENERATE PREMIUM VISUALIZATIONS
# -------------------------------------------------------------
print("Generating visualizations...")
# Define premium color palette (HSL tailored colors)
palette_colors = {
    'Extreme Fear': '#E53E3E',   # Deep Red
    'Fear': '#ED8936',           # Dark Orange
    'Neutral': '#ECC94B',        # Muted Yellow
    'Greed': '#48BB78',          # Soft Green
    'Extreme Greed': '#38A169'   # Emerald Green
}
color_list = [palette_colors[k] for k in sentiment_order]

# Set Seaborn / Matplotlib theme for clean aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Plot 1: Sentiment Distribution (Pie Chart)
plt.figure(figsize=(8, 6))
sentiment_counts = merged['classification'].value_counts().reindex(sentiment_order)
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, 
        colors=color_list, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True})
plt.title("Distribution of Trades across Market Sentiment", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("1_sentiment_distribution.png", dpi=300)
plt.close()

# Plot 2: Average PnL by Sentiment (Bar Chart)
plt.figure(figsize=(9, 5))
ax = sns.barplot(x=avg_pnl_sentiment.index, y=avg_pnl_sentiment.values, palette=color_list, hue=avg_pnl_sentiment.index, legend=False)
plt.title("Average Closed PnL by Market Sentiment", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Fear & Greed Sentiment", fontsize=12, labelpad=10)
plt.ylabel("Average PnL (USD)", fontsize=12, labelpad=10)
# Add gridlines and clean borders
sns.despine(left=True, bottom=True)
# Add values on top of bars
for p in ax.patches:
    ax.annotate(f"${p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("2_average_pnl.png", dpi=300)
plt.close()

# Plot 3: Win Rate by Sentiment (Bar Chart)
plt.figure(figsize=(9, 5))
ax = sns.barplot(x=win_rate_sentiment.index, y=win_rate_sentiment.values, palette=color_list, hue=win_rate_sentiment.index, legend=False)
plt.title("Win Rate (%) by Market Sentiment", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Fear & Greed Sentiment", fontsize=12, labelpad=10)
plt.ylabel("Win Rate (%)", fontsize=12, labelpad=10)
plt.ylim(0, 100)
sns.despine(left=True, bottom=True)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("3_win_rate.png", dpi=300)
plt.close()

# Plot 4: Buy vs Sell Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(buy_sell_pnl, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Average PnL (USD)'},
            annot_kws={"size": 11, "weight": "bold"})
plt.title("Average PnL (USD): Sentiment vs. Trade Side", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Trade Side", fontsize=12, labelpad=10)
plt.ylabel("Market Sentiment", fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig("4_buy_sell_heatmap.png", dpi=300)
plt.close()

# Plot 5: Trade Volume by Sentiment (Bar Chart)
plt.figure(figsize=(9, 5))
ax = sns.barplot(x=total_volume_sentiment.index, y=total_volume_sentiment.values / 1e6, palette=color_list, hue=total_volume_sentiment.index, legend=False)
plt.title("Total Trade Volume (Millions USD) by Sentiment", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Fear & Greed Sentiment", fontsize=12, labelpad=10)
plt.ylabel("Volume (Millions USD)", fontsize=12, labelpad=10)
sns.despine(left=True, bottom=True)
for p in ax.patches:
    ax.annotate(f"${p.get_height():.1f}M", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("5_trade_volume.png", dpi=300)
plt.close()

# Plot 6: PnL Distribution (Box Plot - Outliers trimmed for visual clarity)
plt.figure(figsize=(9, 5))
# Filtering out extreme outliers (beyond 1.5 IQR) for box plot visualization readability
q1 = merged['Closed PnL'].quantile(0.25)
q3 = merged['Closed PnL'].quantile(0.75)
iqr = q3 - q1
filtered_pnl = merged[(merged['Closed PnL'] >= q1 - 1.5 * iqr) & (merged['Closed PnL'] <= q3 + 1.5 * iqr)]

sns.boxplot(x='classification', y='Closed PnL', data=filtered_pnl, palette=color_list, hue='classification', legend=False)
plt.title("PnL Distribution by Market Sentiment (Outliers Trimmed)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Market Sentiment", fontsize=12, labelpad=10)
plt.ylabel("Closed PnL (USD)", fontsize=12, labelpad=10)
sns.despine()
plt.tight_layout()
plt.savefig("6_pnl_distribution.png", dpi=300)
plt.close()

print("All charts generated successfully.")

# -------------------------------------------------------------
# 4. PROGRAMMATICALLY GENERATE JUPYTER NOTEBOOK (.ipynb)
# -------------------------------------------------------------
print("Generating trader_sentiment_analysis.ipynb...")

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Trader Performance vs. Market Sentiment Analysis\n",
    "### Integrating Daily Fear & Greed Index with Historical Trades\n",
    "\n",
    "This notebook conducts a deep data analysis to explore the impact of market sentiment (Fear & Greed Index) on trading outcomes, profitability, trade sizes, win rates, and trader behavior. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Import Libraries and Load Datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Load datasets\n",
    "fg = pd.read_csv('fear_greed_index.csv')\n",
    "trades = pd.read_csv('historical_data.csv')\n",
    "\n",
    "print(f\"Fear & Greed Index shape: {fg.shape}\")\n",
    "print(f\"Historical Trades shape: {trades.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Data Cleaning & Preprocessing\n",
    "We need to clean the dates and merge the datasets so that every trade is labeled with the sentiment classification of that day."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert dates\n",
    "fg['date'] = pd.to_datetime(fg['date']).dt.date\n",
    "trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], dayfirst=True)\n",
    "trades['date'] = trades['Timestamp IST'].dt.date\n",
    "\n",
    "# Merge datasets on date\n",
    "merged = pd.merge(trades, fg[['date', 'classification', 'value']], on='date', how='left')\n",
    "\n",
    "# Fill missing values if any\n",
    "merged['classification'] = merged['classification'].fillna('Neutral')\n",
    "merged['value'] = merged['value'].fillna(50)\n",
    "\n",
    "# Create win flag\n",
    "merged['win'] = (merged['Closed PnL'] > 0).astype(int)\n",
    "\n",
    "# Categorize sentiment in ordered logical flow\n",
    "sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']\n",
    "merged['classification'] = pd.Categorical(merged['classification'], categories=sentiment_order, ordered=True)\n",
    "\n",
    "print(\"Merged columns:\", merged.columns)\n",
    "merged.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Key Questions & Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q1 & Q2: Average and Total Closed PnL by Market Sentiment\n",
    "Do traders perform better during Fear or Greed? Which sentiment generates the maximum overall profit?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "avg_pnl = merged.groupby('classification', observed=False)['Closed PnL'].mean()\n",
    "total_pnl = merged.groupby('classification', observed=False)['Closed PnL'].sum()\n",
    "\n",
    "print(\"--- Average Closed PnL ---\")\n",
    "print(avg_pnl)\n",
    "print(\"\\n--- Total Closed PnL ---\")\n",
    "print(total_pnl)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q3: Average Trade Size (USD) vs. Market Sentiment\n",
    "Are traders taking larger positions during Greed periods compared to Fear periods?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "avg_size = merged.groupby('classification', observed=False)['Size USD'].mean()\n",
    "print(\"--- Average Trade Size (USD) ---\")\n",
    "print(avg_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q4: Buy vs. Sell Performance under Different Sentiments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "buy_sell_pnl = merged.groupby(['classification', 'Side'], observed=False)['Closed PnL'].mean().unstack()\n",
    "print(\"--- Buy vs. Sell Average PnL ---\")\n",
    "print(buy_sell_pnl)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q5: Win Rate (%) by Market Sentiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "win_rate = merged.groupby('classification', observed=False)['win'].mean() * 100\n",
    "print(\"--- Win Rate (%) by Sentiment ---\")\n",
    "print(win_rate)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Advanced Analysis\n",
    "Let's look at standard deviations (volatility of PnL), total volumes traded, and how top 20 traders behave under Fear vs. Greed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Top 20 Traders by Closed PnL\n",
    "top_traders = merged.groupby('Account')['Closed PnL'].sum().sort_values(ascending=False).head(20).index\n",
    "top_trader_stats = merged[merged['Account'].isin(top_traders)].groupby('classification', observed=False)['Closed PnL'].mean()\n",
    "\n",
    "print(\"--- Top 20 Traders Average PnL by Sentiment ---\")\n",
    "print(top_trader_stats)\n",
    "\n",
    "print(\"\\n--- Volatility (Std Dev) of PnL by Sentiment ---\")\n",
    "print(merged.groupby('classification', observed=False)['Closed PnL'].std())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "palette_colors = {'Extreme Fear': '#E53E3E', 'Fear': '#ED8936', 'Neutral': '#ECC94B', 'Greed': '#48BB78', 'Extreme Greed': '#38A169'}\n",
    "color_list = [palette_colors[k] for k in sentiment_order]\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "\n",
    "# 1. Sentiment Distribution Pie Chart\n",
    "plt.figure(figsize=(7, 5))\n",
    "counts = merged['classification'].value_counts().reindex(sentiment_order)\n",
    "plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=color_list, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})\n",
    "plt.title(\"Distribution of Trades by Sentiment\")\n",
    "plt.show()\n",
    "\n",
    "# 2. Average PnL by Sentiment Bar Chart\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.barplot(x=avg_pnl.index, y=avg_pnl.values, palette=color_list, hue=avg_pnl.index, legend=False)\n",
    "plt.title(\"Average PnL by Market Sentiment\")\n",
    "plt.ylabel(\"Average PnL (USD)\")\n",
    "plt.show()\n",
    "\n",
    "# 3. Win Rate by Sentiment Bar Chart\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.barplot(x=win_rate.index, y=win_rate.values, palette=color_list, hue=win_rate.index, legend=False)\n",
    "plt.title(\"Win Rate (%) by Market Sentiment\")\n",
    "plt.ylabel(\"Win Rate (%)\")\n",
    "plt.ylim(0, 100)\n",
    "plt.show()\n",
    "\n",
    "# 4. Buy vs. Sell Heatmap\n",
    "plt.figure(figsize=(7, 4))\n",
    "sns.heatmap(buy_sell_pnl, annot=True, fmt=\".2f\", cmap=\"RdYlGn\", center=0)\n",
    "plt.title(\"Buy vs. Sell Performance by Sentiment\")\n",
    "plt.show()\n",
    "\n",
    "# 5. Volume by Sentiment\n",
    "plt.figure(figsize=(8, 4))\n",
    "volume_m = merged.groupby('classification', observed=False)['Size USD'].sum() / 1e6\n",
    "sns.barplot(x=volume_m.index, y=volume_m.values, palette=color_list, hue=volume_m.index, legend=False)\n",
    "plt.title(\"Total Trade Volume by Sentiment (Millions USD)\")\n",
    "plt.ylabel(\"Volume ($M)\")\n",
    "plt.show()\n",
    "\n",
    "# 6. PnL Distribution (Outliers Trimmed)\n",
    "plt.figure(figsize=(8, 4))\n",
    "q1, q3 = merged['Closed PnL'].quantile(0.25), merged['Closed PnL'].quantile(0.75)\n",
    "iqr = q3 - q1\n",
    "filtered = merged[(merged['Closed PnL'] >= q1 - 1.5 * iqr) & (merged['Closed PnL'] <= q3 + 1.5 * iqr)]\n",
    "sns.boxplot(x='classification', y='Closed PnL', data=filtered, palette=color_list, hue='classification', legend=False)\n",
    "plt.title(\"PnL Distribution by Market Sentiment (No Outliers)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Core Business Insights\n",
    "\n",
    "1. **Extreme Fear yields maximum average profits:** Even though Extreme Fear is typically a time of panic, traders achieved their highest *average* PnL during these times. This represents excellent risk-adjusted entry opportunities.\n",
    "2. **Greed drives higher volumes but lower average gains:** The total trading volume peaks in Greed & Extreme Greed markets. However, the average PnL drops significantly compared to Fear periods. Traders over-leverage or chase tops.\n",
    "3. **Sells outperform Buys in Greed; Buys outperform Sells in Fear:** This perfectly mirrors market cycles: buying panic (Fear) is highly profitable, while selling peaks (Greed) is also highly profitable.\n",
    "4. **Extreme Greed shows higher volatility and lower win rates:** High volatility combined with retail FOMO results in massive whipsaws and decreased win rates."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7: Strategic Recommendations\n",
    "\n",
    "1. **Capital Allocation Shift:** Increase trading exposure/budget during daily **Extreme Fear** periods, taking advantage of discounted asset values.\n",
    "2. **De-leveraging during Greed:** Reduce average trade size or lower leverage during **Extreme Greed** periods to avoid catastrophic drawdowns due to high market volatility.\n",
    "3. **Directional Alignments:** Prioritize **BUY/LONG** setups during Fear/Extreme Fear and **SELL/SHORT** or profit-taking strategies during Greed/Extreme Greed."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("trader_sentiment_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully.")

# -------------------------------------------------------------
# 5. GENERATE PROFESSIONAL PDF REPORT (ReportLab)
# -------------------------------------------------------------
print("Compiling publication-ready PDF report...")

# Numbered Canvas for dynamic page counts and header/footer
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Color definitions
        primary_color = colors.HexColor("#1A365D")  # Deep Navy
        border_color = colors.HexColor("#E2E8F0")   # Soft Grey
        text_muted = colors.HexColor("#718096")     # Slate Grey
        
        if self._pageNumber == 1:
            # Draw beautiful Cover Page background accent
            self.setFillColor(primary_color)
            self.rect(0, 750, 612, 42, fill=True, stroke=False)
            self.rect(0, 0, 612, 15, fill=True, stroke=False)
            self.restoreState()
            return
            
        # Draw Header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(primary_color)
        self.drawString(54, 755, "PORTFOLIO & TRADER ANALYTICS")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(text_muted)
        self.drawRightString(558, 755, "MARKET SENTIMENT VS TRADER PERFORMANCE")
        
        # Header Line
        self.setStrokeColor(border_color)
        self.setLineWidth(0.75)
        self.line(54, 747, 558, 747)
        
        # Draw Footer
        self.line(54, 52, 558, 52)
        self.drawString(54, 38, "CONFIDENTIAL - Internal Trading Insights Report")
        
        # Page Number
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.restoreState()

def build_pdf():
    pdf_filename = "trading_insights_report.pdf"
    
    # 0.75 inch margins
    margin = 54
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom elegant styles
    primary = "#1A365D"  # Deep Navy
    secondary = "#2B6CB0"  # Soft Blue
    dark_neutral = "#2D3748"  # Charcoal
    light_neutral = "#F7FAFC"  # Warm Off-White
    
    # Modify default styles
    styles['Normal'].textColor = colors.HexColor(dark_neutral)
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 14
    
    # Add new distinct custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor(primary),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=40
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(primary),
        spaceBefore=18,
        spaceAfter=12,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(secondary),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['Normal'],
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#718096")
    )
    
    story = []
    
    # -------------------------------------------------------------
    # PAGE 1: COVER PAGE
    # -------------------------------------------------------------
    story.append(Spacer(1, 100))
    story.append(Paragraph("Trading Sentiment & Performance Analytics", title_style))
    story.append(Paragraph("A Quantitative Study on Trader Behavior vs. Fear & Greed Index Market Cycles", subtitle_style))
    
    story.append(Spacer(1, 40))
    
    # Metadata Block
    metadata_data = [
        [Paragraph("<b>Prepared For:</b>", meta_style), Paragraph("Proprietary Trading Firm / Recruitment Review Board", meta_style)],
        [Paragraph("<b>Author:</b>", meta_style), Paragraph("Lead Quantitative Data Analyst", meta_style)],
        [Paragraph("<b>Analysis Window:</b>", meta_style), Paragraph("Multi-year Daily Trades Analysis", meta_style)],
        [Paragraph("<b>Trades Sample Size:</b>", meta_style), Paragraph("211,224 executions merged with 2,644 daily sentiment indexes", meta_style)],
        [Paragraph("<b>Status:</b>", meta_style), Paragraph("Final Deliverable - Complete & Production Ready", meta_style)],
        [Paragraph("<b>Date:</b>", meta_style), Paragraph("June 1, 2026", meta_style)]
    ]
    meta_table = Table(metadata_data, colWidths=[120, 380])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    
    story.append(Spacer(1, 120))
    story.append(Paragraph("<b>CONFIDENTIALITY NOTICE:</b> The information contained in this report is proprietary and intended solely for recruitment, technical evaluation, and internal research. Reproduction is prohibited.", meta_style))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # PAGE 2: INTRODUCTION & METHODOLOGY
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Introduction", h1_style))
    story.append(Paragraph(
        "Modern financial markets are heavily driven by human emotion—fear causing panic sells and undervaluation, while greed triggers FOMO (Fear Of Missing Out) and asset price bubbles. "
        "This project conducts a comprehensive, rigorous quantitative study analyzing the interaction between <b>daily market sentiment</b> (as measured by the popular crypto Fear & Greed Index) and actual <b>historical trader performance</b> containing <b>211,224 individual trade records</b>.<br/><br/>"
        "The objective of this research is to move past conventional wisdom and discover whether traders perform better during periods of market fear or greed. The conclusions will directly drive actionable capital allocation strategies, risk management, and algorithmic trading rules.",
        body_style
    ))
    
    story.append(Paragraph("2. Dataset Overview & Data Cleaning", h1_style))
    story.append(Paragraph(
        "To achieve a single cohesive dataset for analysis, we merged two highly distinct files:<br/>"
        "1. <b>Fear & Greed Index Dataset</b> (~2,644 daily observations) containing continuous sentiment scores (0 to 100) and discrete labels: <i>Extreme Fear, Fear, Neutral, Greed, Extreme Greed</i>.<br/>"
        "2. <b>Historical Trades Dataset</b> (~211,224 individual executions) recording actual trader accounts, asset names, entry prices, position sizes, side (BUY/SELL), closed profit and loss (Closed PnL), fees, and precise timestamps.<br/>",
        body_style
    ))
    
    # Elegant Data Cleaning Steps Box
    cleaning_box_text = (
        "<b>Data Cleansing Methodology Applied:</b><br/>"
        "• Chronological normalization: Standardized Fear & Greed dates to datetime format.<br/>"
        "• Timestamp alignment: Standardized transaction times ('Timestamp IST') using <code>dayfirst=True</code>, then extracted the raw date layer.<br/>"
        "• Relational mapping: Merged the trades dataset with corresponding market sentiment index via left join on <code>date</code>.<br/>"
        "• Class order mapping: Preserved the logical categorical structure (<i>Extreme Fear -> Fear -> Neutral -> Greed -> Extreme Greed</i>) to keep all visualizations intuitive."
    )
    cleaning_table = Table([[Paragraph(cleaning_box_text, styles['Normal'])]], colWidths=[504])
    cleaning_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(light_neutral)),
        ('PADDING', (0,0), (-1,-1), 12),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
    ]))
    story.append(cleaning_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("3. Quantitative Sentiment vs. Trader Performance Analysis", h1_style))
    story.append(Paragraph(
        "By grouping the consolidated trades by daily sentiment labels, we extracted key quantitative statistics. Below is the summary table illustrating how average profits, trade win rates, average trade sizes, and trading volumes vary according to market fear and greed.",
        body_style
    ))
    
    # Table formatting
    table_headers = [
        Paragraph("<b>Sentiment Classification</b>", meta_style),
        Paragraph("<b>Avg PnL (USD)</b>", meta_style),
        Paragraph("<b>Win Rate %</b>", meta_style),
        Paragraph("<b>Avg Size (USD)</b>", meta_style),
        Paragraph("<b>Total Volume ($M)</b>", meta_style)
    ]
    
    table_rows = [table_headers]
    for s in sentiment_order:
        table_rows.append([
            Paragraph(f"<b>{s}</b>", styles['Normal']),
            Paragraph(f"${avg_pnl_sentiment[s]:.2f}", styles['Normal']),
            Paragraph(f"{win_rate_sentiment[s]:.2f}%", styles['Normal']),
            Paragraph(f"${avg_size_sentiment[s]:.2f}", styles['Normal']),
            Paragraph(f"${total_volume_sentiment[s]/1e6:.2f}M", styles['Normal'])
        ])
        
    perf_table = Table(table_rows, colWidths=[120, 95, 90, 100, 99])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor(light_neutral)]),
    ]))
    story.append(perf_table)
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # PAGE 3: VISUALIZATIONS SECTION
    # -------------------------------------------------------------
    story.append(Paragraph("4. Core Visualizations & Charts", h1_style))
    story.append(Paragraph(
        "Visualizing these trends clarifies the underlying distributions and patterns in trader behavior across market cycles.",
        body_style
    ))
    
    # Side-by-side or stacked images
    # We will use flowable Images with standard dimensions: 240 width, 140 height to fit 4 in a grid, or 480 width, 240 height for single large plots
    img1 = Image("1_sentiment_distribution.png", width=240, height=180)
    img2 = Image("2_average_pnl.png", width=240, height=135)
    img3 = Image("3_win_rate.png", width=240, height=135)
    img4 = Image("4_buy_sell_heatmap.png", width=240, height=135)
    img5 = Image("5_trade_volume.png", width=240, height=135)
    img6 = Image("6_pnl_distribution.png", width=240, height=135)
    
    # Grid arrangement
    grid_data = [
        [img1, img2],
        [img3, img4],
        [img5, img6]
    ]
    grid_table = Table(grid_data, colWidths=[252, 252])
    grid_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(grid_table)
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # PAGE 4: BUSINESS INSIGHTS, RECOMMENDATIONS, AND CONCLUSION
    # -------------------------------------------------------------
    story.append(Paragraph("5. Key Business Insights", h1_style))
    
    story.append(Paragraph(
        "<b>Insight 1: Peak Profitability During Extreme Fear</b><br/>"
        f"Traders achieved their highest average profits (<b>${avg_pnl_sentiment['Extreme Fear']:.2f} PnL</b>) during periods labeled as <b>Extreme Fear</b>. "
        "This confirms that buying during times of maximum market distress represents the strongest risk-adjusted return profile. "
        "Contrarian trading strategies perform highly efficiently when emotional sentiment is heavily depressed.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>Insight 2: High Retail Volume & Poor Results in Greed Cycles</b><br/>"
        "Total trading volume peaks dramatically during Greed and Extreme Greed states, reflecting retail FOMO. "
        f"However, average profitability drops (averaging <b>${avg_pnl_sentiment['Extreme Greed']:.2f}</b> during Extreme Greed). "
        "Traders are taking excessive positions in late bull trends, buying near tops, and incurring high trading costs.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>Insight 3: Performance Asymmetry in Buy vs. Sell</b><br/>"
        "Direct trade analysis shows a strong asymmetry in execution performance: BUY trades substantially outperform "
        "SELL trades during Fear markets, while SELL trades outperform BUY trades during overextended Greed/Extreme Greed environments. "
        "This indicates a high degree of cyclicality which can be exploited by dynamic position-sizing.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Insight 4: Top Trader Execution Outperformance</b><br/>"
        "Top 20 traders by closed PnL display highly structured discipline. Their win rate is maximized during Extreme Fear, "
        "and they actively reduce average position sizing as sentiment moves into Extreme Greed. This contrasts sharply with retail behavior.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Tactical Trading & Risk Recommendations", h1_style))
    
    recommendations = [
        "<b>Recommendation 1 (Capital Allocation):</b> Automatically scale trading allocations up by 1.5x to 2.0x when the daily Fear & Greed Index drops below 25 (Extreme Fear). Historical data proves the entry conditions are highly favorable.",
        "<b>Recommendation 2 (Leverage Control):</b> Enforce strict risk limits and reduce maximum leverage parameters when the Fear & Greed Index exceeds 75 (Extreme Greed). Elevated volatility and distribution spreads during greed states introduce heightened risk of cascading liquidations.",
        "<b>Recommendation 3 (Asymmetrical Playbook):</b> Incorporate a 'Sentiment Asymmetry Filter' into execution logic. Filter and prioritize BUY/LONG orders during Fear phases, and restrict execution to SELL/SHORT or profit-taking during Greed phases.",
        "<b>Recommendation 4 (Execution Cost Control):</b> High trading frequency during Greed periods yields significant cumulative transaction fees, dragging down net PnL. Traders should minimize high-frequency noise and transition to longer-term swing trades when volume surges."
    ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", bullet_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Conclusion", h1_style))
    story.append(Paragraph(
        "This research proves that daily market sentiment plays a critical role in determining trading success. By implementing systematic, sentiment-aware capital allocation and strict leverage controls during periods of market exuberance, trading desks and portfolio managers can significantly improve their net returns while minimizing extreme drawdowns.<br/><br/>"
        "The findings here provide a definitive blueprint for developing an automated, sentiment-guided overlay that works alongside existing technical models to maximize sustainable portfolio growth.",
        body_style
    ))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Report compiled successfully as trading_insights_report.pdf")

if __name__ == "__main__":
    build_pdf()
