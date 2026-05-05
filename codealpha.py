import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Iris.csv")
df = df.drop("Id", axis=1)

# Style
sns.set(style="whitegrid")

# =========================
# 1. Pairplot 🔥
# =========================
sns.pairplot(df, hue="Species")
plt.show()

# =========================
# 2. Heatmap (Correlation)
# =========================
plt.figure()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 3. Boxplot
# =========================
plt.figure()
sns.boxplot(x="Species", y="SepalLengthCm", data=df)
plt.title("Sepal Length by Species")
plt.show()

# =========================
# 4. Violin Plot 🔥
# =========================
plt.figure()
sns.violinplot(x="Species", y="PetalLengthCm", data=df)
plt.title("Petal Length Distribution")
plt.show()

# =========================
# 5. Histogram
# =========================
plt.figure()
df["SepalWidthCm"].hist()
plt.title("Sepal Width Distribution")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()

# =========================
# 6. Scatter Plot
# =========================
plt.figure()
sns.scatterplot(x="SepalLengthCm", y="PetalLengthCm", hue="Species", data=df)
plt.title("Sepal vs Petal Length")
plt.show()

# =========================
# 7. Count Plot
# =========================
plt.figure()
sns.countplot(x="Species", data=df)
plt.title("Count of Each Species")
plt.show()

# =========================
# 8. KDE Plot 🔥
# =========================
plt.figure()
sns.kdeplot(data=df, x="PetalWidthCm", hue="Species", fill=True)
plt.title("Petal Width Density")
plt.show()