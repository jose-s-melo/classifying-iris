import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.datasets import load_iris
import statsmodels.api as sm

iris = load_iris(as_frame=True)
df = iris.frame

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

variables = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]


def normal_test(df, variables):
    for var in variables:
        stat, p = shapiro(df[var])

        print(stat)
        print(f"\n{var}")
        print(f"Shapiro-Wilk p-value = {p:.4f}")

        if p > 0.05:
            print("~= Normal")
        else:
            print("~= Não normal")

def qq_plot(df, axes, variables):
    for ax, var in zip(axes.flatten(), variables):
        sm.qqplot(
            df[var],
            line="s",
            ax=ax
        )
        ax.set_title(f"QQ-Plot: {var}")


qq_plot(df=df, axes=axes, variables=variables)
normal_test(df=df, variables=variables)


plt.tight_layout()
plt.savefig("iris_qqplots.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo em iris_qqplots.png")