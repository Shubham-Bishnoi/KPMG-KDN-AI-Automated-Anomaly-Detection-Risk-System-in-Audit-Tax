import time
start_time = time.time()

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid blocking
import polars as ps  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os
import pandas as pd
import json
import seaborn as sns
import folium
from folium.plugins import FeatureGroupSubGroup
import warnings

# --- File Path Configuration ---
BASE_DATA_PATH = "/Users/shubhambishnoi/Downloads/KPMG-KDN-AI-Automated-Anomaly-Detection-Risk-System-in-Audit-Tax/data"
CARDS_DATA_FILE = os.path.join(BASE_DATA_PATH, "cards_data.csv")
MCC_CODES_FILE = os.path.join(BASE_DATA_PATH, "mcc_codes.json")
TRAIN_FRAUD_LABELS_FILE = os.path.join(BASE_DATA_PATH, "train_fraud_labels.json")
TRANSACTIONS_DATA_FILE = os.path.join(BASE_DATA_PATH, "transactions_data.csv")
USERS_DATA_FILE = os.path.join(BASE_DATA_PATH, "users_data.csv")

# Set up directory for saving figures
FIGURE_DIR = os.path.join(BASE_DATA_PATH, "reports", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

########################################
### 1. Cards Data Processing (Polars) ###
########################################
print("Loading cards data from:", CARDS_DATA_FILE)
df_cards = ps.read_csv(CARDS_DATA_FILE)
print("Cards data loaded. Number of rows:", df_cards.shape[0])
print("Cards data (head):")
print(df_cards.head())

print("Cleaning 'credit_limit' column...")
df_cards = df_cards.with_columns(
    ps.col('credit_limit')
    .str.replace(r'[\$,]', '', literal=False)
    .cast(ps.Float64)
)
print("Converting 'acct_open_date' to Date format (%m/%Y)...")
df_cards = df_cards.with_columns(
    ps.col("acct_open_date").str.strptime(ps.Date, format="%m/%Y").alias("acct_open_date")
)
print("Cards data after cleaning (head):")
print(df_cards.head())

print("Unique value counts for string columns:")
string_columns = df_cards.select(ps.selectors.string()).columns
for col in string_columns:
    unique_count = df_cards[col].n_unique()
    print(f"  {col}: {unique_count} unique values")

print("Filtering out cards with 'card_on_dark_web' == 'Yes'...")
df_cards = df_cards.filter(ps.col('card_on_dark_web') != 'Yes')
print("Cards data after filtering (head):")
print(df_cards.head())

########################################
### 2. Graphical Analysis on Cards Data #
########################################

# --- Stacked Bar Chart: Card Brands by Card Type ---
print("Plotting stacked bar chart: Distribution of Card Brands by Type...")
card_counts = (
    df_cards
    .group_by(["card_brand", "card_type"])
    .agg(ps.len().alias("count"))
)
brand_labels = sorted(card_counts["card_brand"].unique().to_list())
type_labels = sorted(card_counts["card_type"].unique().to_list())
brand_type_counts = {brand: {ct: 0 for ct in type_labels} for brand in brand_labels}
for row in card_counts.iter_rows(named=True):
    brand_type_counts[row["card_brand"]][row["card_type"]] = row["count"]
data_matrix = np.array([[brand_type_counts[brand].get(ct, 0) for ct in type_labels] for brand in brand_labels])
colors = plt.cm.Paired(np.linspace(0, 1, len(type_labels)))
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(brand_labels))
bars = []
for i, ct in enumerate(type_labels):
    bar_group = ax.bar(brand_labels, data_matrix[:, i], bottom=bottom, color=colors[i],
                       edgecolor="black", label=ct)
    bars.append(bar_group)
    bottom += data_matrix[:, i]
ax.set_xlabel("Card Brand", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Card Brands by Type", fontsize=14, fontweight="bold")
ax.legend(title="Card Type", loc="upper right")
plt.xticks(rotation=45)
# Annotate each bar segment
for group in bars:
    for bar in group:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, int(height),
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
stacked_bar_file = os.path.join(FIGURE_DIR, "stacked_bar_card_brands.png")
plt.savefig(stacked_bar_file)
print("Saved stacked bar chart to:", stacked_bar_file)
plt.close()

# --- Bar Chart: Number of Users by Card Type ---
print("Plotting bar chart: Number of Users by Card Type...")
card_type_counts = (
    df_cards
    .group_by("card_type")
    .agg(ps.len().alias("count"))
    .sort("count", descending=True)
)
card_types = card_type_counts["card_type"].to_list()
counts = card_type_counts["count"].to_list()
plt.figure(figsize=(8, 5))
plt.bar(card_types, counts, color="skyblue", edgecolor="black")
plt.xlabel("Card Type", fontsize=12)
plt.ylabel("Number of Users", fontsize=12)
plt.title("Number of Users by Card Type", fontsize=14, fontweight="bold")
for i, count in enumerate(counts):
    plt.text(i, count + 0.1, str(count), ha="center", fontsize=12, fontweight="bold")
plt.xticks(rotation=30)
bar_chart_file = os.path.join(FIGURE_DIR, "bar_chart_users_by_card_type.png")
plt.savefig(bar_chart_file)
print("Saved bar chart to:", bar_chart_file)
plt.close()

# --- Pie Chart: Card Brand Distribution ---
print("Plotting pie chart: Proportion of Different Card Brands...")
card_brand_counts = (
    df_cards
    .group_by("card_brand")
    .agg(ps.len().alias("count"))
    .sort("count", descending=True)
)
brands = card_brand_counts["card_brand"].to_list()
counts = card_brand_counts["count"].to_list()
colors = plt.cm.Paired.colors[:len(brands)]
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=brands, autopct="%1.1f%%", colors=colors,
        startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title("Proportion of Different Card Brands", fontsize=14, fontweight="bold")
pie_chart_file = os.path.join(FIGURE_DIR, "pie_chart_card_brands.png")
plt.savefig(pie_chart_file)
print("Saved pie chart to:", pie_chart_file)
plt.close()

# --- Histogram: Distribution of Credit Limits ---
print("Plotting histogram: Distribution of Credit Limits...")
credit_limits = df_cards["credit_limit"].to_list()
plt.figure(figsize=(10, 6))
plt.hist(credit_limits, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
plt.xlabel("Credit Limit", fontsize=12)
plt.ylabel("Number of Users", fontsize=12)
plt.title("Distribution of Credit Limits", fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
hist_credit_file = os.path.join(FIGURE_DIR, "hist_credit_limits.png")
plt.savefig(hist_credit_file)
print("Saved histogram of credit limits to:", hist_credit_file)
plt.close()

# --- Box Plot: Credit Limits per Card Type ---
print("Plotting box plot: Credit Limit Distribution by Card Type...")
card_types_unique = df_cards["card_type"].unique().to_list()
credit_limit_data = [df_cards.filter(df_cards["card_type"] == ct)["credit_limit"].to_list() for ct in card_types_unique]
plt.figure(figsize=(8, 6))
plt.boxplot(credit_limit_data, labels=card_types_unique, patch_artist=True,
            boxprops=dict(facecolor="skyblue", alpha=0.7))
plt.xlabel("Card Type", fontsize=12)
plt.ylabel("Credit Limit", fontsize=12)
plt.title("Credit Limit Distribution by Card Type", fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
box_card_type_file = os.path.join(FIGURE_DIR, "boxplot_credit_limits_by_card_type.png")
plt.savefig(box_card_type_file)
print("Saved box plot for credit limits by card type to:", box_card_type_file)
plt.close()

# --- Box Plot: Credit Limits per Card Brand ---
print("Plotting box plot: Credit Limit Distribution by Card Brand...")
card_brands_unique = df_cards["card_brand"].unique().to_list()
credit_limit_data = [df_cards.filter(df_cards["card_brand"] == brand)["credit_limit"].to_list() for brand in card_brands_unique]
plt.figure(figsize=(8, 6))
plt.boxplot(credit_limit_data, labels=card_brands_unique, patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7))
plt.xlabel("Card Brand", fontsize=12)
plt.ylabel("Credit Limit", fontsize=12)
plt.title("Credit Limit Distribution by Card Brand", fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
box_card_brand_file = os.path.join(FIGURE_DIR, "boxplot_credit_limits_by_card_brand.png")
plt.savefig(box_card_brand_file)
print("Saved box plot for credit limits by card brand to:", box_card_brand_file)
plt.close()

# --- Histogram: Number of Cards Issued ---
print("Plotting histogram: Distribution of Number of Cards Issued...")
num_cards = df_cards["num_cards_issued"].to_list()
plt.figure(figsize=(8, 6))
plt.hist(num_cards, bins=range(1, max(num_cards) + 2), color="skyblue", edgecolor="black", alpha=0.7, align='left')
plt.xlabel("Number of Cards Issued", fontsize=12)
plt.ylabel("Number of Users", fontsize=12)
plt.title("Distribution of Number of Cards Issued", fontsize=14, fontweight="bold")
plt.xticks(range(1, max(num_cards) + 1))
plt.grid(axis="y", linestyle="--", alpha=0.7)
hist_cards_file = os.path.join(FIGURE_DIR, "hist_number_of_cards_issued.png")
plt.savefig(hist_cards_file)
print("Saved histogram of number of cards issued to:", hist_cards_file)
plt.close()

# --- Time-Series Analysis: Account Opening Dates ---
print("Plotting time-series analysis: Trend of Account Openings Over Time...")
yearly_counts = df_cards.group_by(
   df_cards["acct_open_date"].dt.year().alias("year")
).agg(
   ps.len().alias("num_accounts")
).sort("year")
years = yearly_counts["year"].to_list()
num_accounts = yearly_counts["num_accounts"].to_list()
plt.figure(figsize=(10, 6))
plt.plot(years, num_accounts, marker="o", linestyle="-", color="royalblue", linewidth=2)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Accounts Opened", fontsize=12)
plt.title("Trend of Account Openings Over Time", fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(years, rotation=45)
time_series_file = os.path.join(FIGURE_DIR, "time_series_account_openings.png")
plt.savefig(time_series_file)
print("Saved time-series analysis plot to:", time_series_file)
plt.close()

######################################
### 3. Merging & Cleaning Formats  ###
######################################
print("Merging transactions data with card and user data...")
# Load datasets using pandas for merging
transactions_df = pd.read_csv(TRANSACTIONS_DATA_FILE)
card_data = pd.read_csv(CARDS_DATA_FILE)
client_data = pd.read_csv(USERS_DATA_FILE)

print("Renaming columns for merging...")
client_data.rename(columns={'id': 'client_id'}, inplace=True)
card_data.rename(columns={'id': 'card_id'}, inplace=True)
transactions_df.rename(columns={'id': 'transactions_id'}, inplace=True)

print("Merging transactions with card data...")
transactions_df_merged = transactions_df.merge(card_data, on='card_id', how='inner')
transactions_df_merged.drop(columns=['client_id_y'], inplace=True)
transactions_df_merged.rename(columns={'client_id_x': 'client_id'}, inplace=True)
print("Merging with client data...")
transactions_df_merged = transactions_df_merged.merge(client_data, on='client_id', how='inner')
print("Merged transactions data shape:", transactions_df_merged.shape)

print("Cleaning formats in merged transactions data...")
transactions_df_merged['amount'] = transactions_df_merged['amount'].str.replace('$', '').astype(float)
transactions_df_merged['credit_limit'] = transactions_df_merged['credit_limit'].str.replace('$', '').astype(float)
transactions_df_merged['per_capita_income'] = transactions_df_merged['per_capita_income'].str.replace('$', '').astype(float)
transactions_df_merged['yearly_income'] = transactions_df_merged['yearly_income'].str.replace('$', '').astype(float)
transactions_df_merged['total_debt'] = transactions_df_merged['total_debt'].str.replace('$', '').astype(float)

transactions_df_merged['expires'] = pd.to_datetime(transactions_df_merged['expires'], format='%m/%Y')
transactions_df_merged['acct_open_date'] = pd.to_datetime(transactions_df_merged['acct_open_date'], format='%m/%Y')

merged_output_file = os.path.join(BASE_DATA_PATH, "transactions_df_merged.csv")
transactions_df_merged.to_csv(merged_output_file, index=False)
print("Cleaned merged transactions data saved to:", merged_output_file)

####################################
### 4. Financial Transactions EDA  ###
####################################
print("Starting Financial Transactions EDA...")

# For mapping and further EDA, load datasets using pandas
users_data = pd.read_csv(USERS_DATA_FILE)
transactions_data = pd.read_csv(TRANSACTIONS_DATA_FILE)
cards_data = pd.read_csv(CARDS_DATA_FILE)
print("Users data shape:", users_data.shape)
print("Transactions data shape:", transactions_data.shape)
print("Cards data shape:", cards_data.shape)

####################################
### EDA on Users Data
####################################
print("Performing EDA on Users Data...")
users_data['per_capita_income'] = users_data['per_capita_income'].replace({'\$': ''}, regex=True).astype(int)
users_data['yearly_income'] = users_data['yearly_income'].replace({'\$': ''}, regex=True).astype(int)
users_data['total_debt'] = users_data['total_debt'].replace({'\$': ''}, regex=True).astype(int)
print("Cleaned Users Data (head):")
print(users_data.head(5))
print("Missing values in Users Data:\n", users_data.isnull().sum())

print("Plotting histogram for Age Distribution by Gender...")
sns.histplot(data=users_data, x='current_age', hue='gender', multiple='stack', kde=True)
plt.title("Age Distribution by Gender")
age_hist_file = os.path.join(FIGURE_DIR, "users_age_distribution.png")
plt.savefig(age_hist_file)
print("Saved age distribution histogram to:", age_hist_file)
plt.close()

print("Plotting histogram for Credit Score Distribution by Gender...")
sns.histplot(data=users_data, x='credit_score', hue='gender', multiple='stack', kde=True)
plt.title("Credit Score Distribution by Gender")
credit_score_hist_file = os.path.join(FIGURE_DIR, "users_credit_score_distribution.png")
plt.savefig(credit_score_hist_file)
print("Saved credit score histogram to:", credit_score_hist_file)
plt.close()

print("Plotting scatterplot for Yearly Income vs Total Debt...")
sns.scatterplot(data=users_data, x='yearly_income', y='total_debt', hue='num_credit_cards', alpha=0.5)
plt.title("Yearly Income vs Total Debt")
scatter_income_debt_file = os.path.join(FIGURE_DIR, "users_income_vs_debt.png")
plt.savefig(scatter_income_debt_file)
print("Saved scatterplot to:", scatter_income_debt_file)
plt.close()

print("Plotting 3D scatter for Users Data...")
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(users_data['yearly_income'], users_data['total_debt'], users_data['credit_score'], c='red', marker='o', alpha=0.3)
ax.set_xlabel('Yearly Income')
ax.set_ylabel('Total Debt')
ax.set_zlabel('Credit Score')
plt.title("3D Scatter: Income, Debt, and Credit Score")
scatter3d_file = os.path.join(FIGURE_DIR, "users_3d_scatter.png")
plt.savefig(scatter3d_file)
print("Saved 3D scatter plot to:", scatter3d_file)
plt.close()

print("Creating folium map for Users Data...")
m = folium.Map(location=[users_data["latitude"].mean(), users_data["longitude"].mean()], zoom_start=5)
fg_main = folium.FeatureGroup(name="Income Categories").add_to(m)
fg_above = FeatureGroupSubGroup(fg_main, "Above Per Capita Income").add_to(m)
fg_below = FeatureGroupSubGroup(fg_main, "Below Per Capita Income").add_to(m)
fg_equal = FeatureGroupSubGroup(fg_main, "Equal to Per Capita Income").add_to(m)
for _, row in users_data.iterrows():
    if row["yearly_income"] > row["per_capita_income"]:
        color, group = "green", fg_above
    elif row["yearly_income"] < row["per_capita_income"]:
        color, group = "red", fg_below
    else:
        color, group = "blue", fg_equal
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.2,
        popup=f"Income: ${row['yearly_income']}<br>Per Capita: ${row['per_capita_income']}"
    ).add_to(group)
folium.LayerControl(collapsed=False).add_to(m)
map_output_file = os.path.join(BASE_DATA_PATH, "users_income_map.html")
m.save(map_output_file)
print("Folium map saved to:", map_output_file)

####################################
### Analysis on Classification Dataset
####################################
print("Performing analysis on classification dataset (Fraud Status)...")
# Merge fraud info with transaction data
with open(TRAIN_FRAUD_LABELS_FILE, 'r') as file:
    fraud = json.load(file)
fraud_id = list(fraud['target'].keys())
fraud_status = list(fraud['target'].values())
df1 = pd.DataFrame({"id": fraud_id, "Status": fraud_status})
print("Fraud Data (head):")
print(df1.head(3))

# Convert the 'id' column in df1 to numeric to match transactions_data
df1['id'] = pd.to_numeric(df1['id'], errors='coerce')

# Now merge using the correctly typed 'id' column
df = pd.merge(transactions_data, df1, on='id', how='inner')
print("Transactions with Fraud Data (head):")
print(df.head(3))


# Merge MCC codes
with open(MCC_CODES_FILE, 'r') as file:
    codes = json.load(file)
df2 = pd.DataFrame({'mcc': list(codes.keys()), 'Name': list(codes.values())})

# Convert both columns to string
df['mcc'] = df['mcc'].astype(str)
df2['mcc'] = df2['mcc'].astype(str)

df = pd.merge(df, df2, on='mcc', how='inner')
print("Transactions merged with MCC Codes (head):")
print(df.head(3))


# Process amount column
df['amount'] = df['amount'].str.replace('$', '')
df['amount'] = pd.to_numeric(df['amount'])
df.rename(columns={'amount': 'Amount($)'}, inplace=True)

print("Missing values summary for transactions:")
miss = []
rows = df.shape[0]
for col in df.columns:
    missing = df[col].isnull().sum()
    miss.append(missing)
res = pd.DataFrame({'Column Name': df.columns, 'Missing Values': miss})
print(res)

print("Filling missing error values with 'Errorless'...")
df['errors'].fillna('Errorless', inplace=True)
print("Filling missing merchant_state values with mode...")
df['merchant_state'].fillna(df['merchant_state'].mode()[0], inplace=True)
print("Plotting box plot for 'zip' column to inspect outliers...")
plt.figure()
df['zip'].plot(kind='box')
box_zip_file = os.path.join(FIGURE_DIR, "boxplot_zip.png")
plt.savefig(box_zip_file)
print("Saved box plot for 'zip' to:", box_zip_file)
plt.close()
print("Filling missing zip values with mean...")
df['zip'].fillna(df['zip'].mean(), inplace=True)
print("Missing values after cleaning:")
print(df.isnull().sum())

print("Sorting Amount values:")
amt = df['Amount($)'].sort_values(ascending=True)
print(amt.head())

amt_below = df[df['Amount($)'] <= 0]
print("Shape of transactions with non-positive amounts:", amt_below.shape)

print("Assigning Payment Type based on Amount...")
amount = df['Amount($)']
payment_status = ['Debit' if val < 0 else 'Credit' for val in amount]
df['Payment_Type'] = payment_status

print("Renaming columns for clarity...")
df.rename(columns={"id": "ID", 'date': 'Date', 'client_id': 'CID', 'Amount($)': 'Amount',
                   'use_chip': 'UseChip', 'merchant_id': 'MID', 'merchant_city': 'MCity',
                   'merchant_state': 'MState', 'zip': 'Pincode', 'mcc': 'MCC',
                   'errors': 'Error', 'Status': 'Fraud_Status', 'Name': 'Category'}, inplace=True)
print("Transactions Data after renaming (head):")
print(df.head(3))

print("Analysis on Fraud Status column:")
per = df['Fraud_Status'].value_counts(normalize=True) * 100
legit = per.get(0, 0)
fraud = per.get(1, 0)
percentages = [legit, fraud]
labels = ['Legitimate', 'Fraud']
plt.figure(figsize=(8, 5))
plt.bar(labels, percentages, color=['#1e8449', '#cb4335'])
plt.title('Percentage of Transactions by Fraud Status')
plt.xlabel('Transaction Type')
plt.ylabel('Percentage')
for i, v in enumerate(percentages):
    plt.text(i, v + 1, f"{round(v, 2)}%", ha='center')
fraud_bar_file = os.path.join(FIGURE_DIR, "fraud_status_bar.png")
plt.savefig(fraud_bar_file)
print("Saved fraud status bar chart to:", fraud_bar_file)
plt.close()

print("Statistical distribution of Amount variable:")
print(df['Amount'].quantile([0.0, 0.25, 0.5, 0.75, 1]))

print("Cleaning Amount values by removing negative sign...")
df['Amount'] = df['Amount'].astype('str').str.replace('-', '')
df['Amount'] = pd.to_numeric(df['Amount'])
print("Distribution of Amount after cleaning:")
print(df['Amount'].quantile([0.0, 0.25, 0.5, 0.75, 1]))

amt = df[df['Amount'] > 71.00]
print("Fraud Status counts for transactions with Amount > 71:")
print(amt['Fraud_Status'].value_counts())

amt_fraud = amt[amt['Fraud_Status'] == 'Yes']
amt_legit = amt[amt['Fraud_Status'] == 'No']
print("Fraud transaction summary stats (Amount):")
print("Mean:", amt_fraud['Amount'].mean())
print(amt_fraud['Amount'].quantile([0, 0.25, 0.5, 0.75, 1]))
print("Legitimate transaction summary stats (Amount):")
print("Mean:", amt_legit['Amount'].mean())
print(amt_legit['Amount'].quantile([0, 0.25, 0.5, 0.75, 1]))

fraud = df[df['Fraud_Status'] == 'Yes']
legit = df[df['Fraud_Status'] == 'No']
print("Total categories present:", df['Category'].nunique())
print("Categories involved in fraud transactions:", fraud['Category'].nunique())

print("Plotting bar chart for Transactions by Category...")
plt.figure(figsize=(10, 6))
df['Category'].value_counts().plot(kind='bar')
plt.title('Transactions by Category')
cat_bar_file = os.path.join(FIGURE_DIR, "transactions_by_category.png")
plt.savefig(cat_bar_file)
print("Saved transactions by category bar chart to:", cat_bar_file)
plt.close()

print("Plotting bar chart for Top 20 Categories with Maximum Transactions...")
plt.figure(figsize=(10, 6))
df['Category'].value_counts().head(20).plot(kind='bar')
plt.title('Top 20 Categories with Maximum Transactions')
top20_bar_file = os.path.join(FIGURE_DIR, "top20_categories.png")
plt.savefig(top20_bar_file)
print("Saved top 20 categories bar chart to:", top20_bar_file)
plt.close()

print("Plotting bar chart for Fraudulent Transactions by Category...")
plt.figure(figsize=(10, 6))
fraud['Category'].value_counts().plot(kind='bar')
plt.title('Fraudulent Transactions by Category')
fraud_cat_bar_file = os.path.join(FIGURE_DIR, "fraud_transactions_by_category.png")
plt.savefig(fraud_cat_bar_file)
print("Saved fraudulent transactions by category bar chart to:", fraud_cat_bar_file)
plt.close()

print("Plotting bar chart for Legitimate Transactions by Category...")
plt.figure(figsize=(10, 6))
legit['Category'].value_counts().sort_values(ascending=False).head(20).plot(kind='bar')
plt.title('Legitimate Transactions by Category')
legit_cat_bar_file = os.path.join(FIGURE_DIR, "legit_transactions_by_category.png")
plt.savefig(legit_cat_bar_file)
print("Saved legitimate transactions by category bar chart to:", legit_cat_bar_file)
plt.close()

print("All steps executed successfully.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
