from matplotlib import pyplot as plt
import numpy as np
from case_studies import *

# uncomment for environments like WSL that have no gui
import matplotlib
matplotlib.use('agg')


def wolfe_search(f, df, x, p, alpha0, c1, c2):
    """
    Implements the wolfe line-search algorithm which returns a point fulfilling the strong wolfe conditions.

    The algorithm selects a point along the line x+alpha*p for alpha>0
    Arguments:
    f: A function f(x) returning the value of the objective function at x
    df: A function df(x) which returns the gradient of f at x
    x: The current iterate
    p: The search direction
    alpha0: initial guess of alpha
    c_1,c_2: parameters of the wolfe condition
    
    Returns:
    alpha: the chosen step length
    brackets: a list of brackets [l,u] showing what the current search range is.
    """
    l = 0        # lower bracket
    u = alpha0   # upper bracket
    brackets = []
    brackets.append([l, u])  # add the initial bracket
    
    return u,brackets # just to return something.


# ── Setup problem ──────────────────────────────────────────────────────
# Try out different functions and starting points.
x = 0.1 * np.ones(3)
p = -df3(x)
alpha0 = 0.00001
c1 = 1.0e-3
c2 = 0.9

plot_upper_range = 0.2 / np.max(np.abs(p))  # max alpha to plot

# ── Run algorithm ──────────────────────────────────────────────────────
alpha, brackets = wolfe_search(f3, df3, x, p, alpha0, c1, c2)
print(f"Selected alpha = {alpha:.6e}")
print(f"Number of bracket iterations: {len(brackets)}")
print("Brackets:", brackets)


#A LOT OF PLOTTING CODE BELOW

# ── Precompute plotting data ───────────────────────────────────────────
alphas = np.linspace(0, plot_upper_range, 400)
ys = np.array([f3(x + a * p) for a in alphas])

f0 = f3(x)
slope0 = np.inner(df3(x), p)  # directional derivative at alpha=0

# Wolfe sufficient-decrease line: f0 + c1 * alpha * slope0
armijo_line = f0 + c1 * alphas * slope0

# ── Create figure ──────────────────────────────────────────────────────
fig, (ax_main, ax_brackets) = plt.subplots(
    2, 1,
    figsize=(10, 7), dpi=150,
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)
fig.suptitle("Wolfe Line Search", fontsize=14, fontweight="bold")

# ── Top panel: function, Armijo line, selected point ──────────
ax_main.plot(alphas, ys, "b-", linewidth=1.5, label=r"$\phi(\alpha) = f(\mathbf{x}+\alpha\,\mathbf{p})$")
ax_main.plot(alphas, armijo_line, "--", color="orange", linewidth=1,
             label=rf"Armijo line  $f_0 + c_1\,\alpha\,\nabla f^T p$  ($c_1$={c1:.0e})")

# Selected point
f_alpha = f3(x + alpha * p)
ax_main.scatter([alpha], [f_alpha], c="red", s=90, zorder=5, edgecolors="k",
                label=rf"Selected  $\alpha^*$={alpha:.4e}")

# Mark the starting point
ax_main.scatter([0], [f0], c="black", marker="D", s=50, zorder=5,
                label=rf"$f_0$={f0:.4f}")

ax_main.set_ylabel(r"$f$ value")
ax_main.legend(loc="best", fontsize=8, framealpha=0.9)
ax_main.grid(True, alpha=0.3)

# ── Bottom panel: bracket evolution ────────────────────────────────────
n_brackets = len(brackets)
for i, (lo, hi) in enumerate(brackets):
    color = plt.cm.viridis(i / max(n_brackets - 1, 1))
    y = n_brackets - i  # stack from top to bottom
    ax_brackets.barh(
        y, hi - lo, left=lo, height=0.7,
        color=color, edgecolor="k", linewidth=0.5, alpha=0.8,
    )


# Mark the final selected alpha
ax_brackets.axvline(alpha, color="red", linestyle="-", linewidth=1.2, label=r"$\alpha^*$")

ax_brackets.set_xlabel(r"$\alpha$")
ax_brackets.set_ylabel("Iteration")
ax_brackets.set_yticks(range(1, n_brackets + 1))
ax_brackets.set_yticklabels([f"{n_brackets - i}" for i in range(n_brackets)], fontsize=7)
ax_brackets.set_title("Bracket evolution  (top = first iteration)", fontsize=9)
ax_brackets.grid(True, axis="x", alpha=0.3)
ax_brackets.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results_line_search.png")