import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Load Dataset
# ============================
df = pd.read_csv("Housing.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail())

print("\n" + "=" * 60)
print("SHAPE OF DATASET")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns)

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

if df.isnull().values.any():
    print("Dataset contains missing values.")
else:
    print("Dataset contains NO missing values.")

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("UNIQUE VALUES IN EACH COLUMN")
print("=" * 60)

for column in df.columns:
    print(f"\n{column}")
    print(df[column].unique())

print("\n" + "=" * 60)
print("VALUE COUNTS OF CATEGORICAL COLUMNS")
print("=" * 60)

categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts())

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
print(df.corr(numeric_only=True))

print("\n" + "=" * 60)
print("MOST EXPENSIVE HOUSE")
print("=" * 60)
print(df.loc[df["price"].idxmax()])

print("\n" + "=" * 60)
print("CHEAPEST HOUSE")
print("=" * 60)
print(df.loc[df["price"].idxmin()])

print("\nAverage House Price:", df["price"].mean())
print("Maximum Area:", df["area"].max())
print("Minimum Area:", df["area"].min())

print("\nNumber of 4 Bedroom Houses:",
      (df["bedrooms"] == 4).sum())

print("\nNumber of Houses with Air Conditioning:",
      (df["airconditioning"] == "yes").sum())

print("\nNumber of Houses with Parking >= 2:",
      (df["parking"] >= 2).sum())

print("\n" + "=" * 60)
print("TOP 5 MOST EXPENSIVE HOUSES")
print("=" * 60)
print(df.sort_values("price", ascending=False).head())

print("\n" + "=" * 60)
print("TOP 5 CHEAPEST HOUSES")
print("=" * 60)
print(df.sort_values("price").head())

# ============================
# Visualizations
# ============================

# Histogram of Price
plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=20)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Scatter Plot: Area vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs Price")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()