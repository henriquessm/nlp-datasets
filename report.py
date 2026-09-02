"""Gera a imagem final consolidada (relatorio_final.png) e o relatorio em PDF
(relatorio_final.pdf) a partir dos resultados dos experimentos do main.ipynb.

Uso no notebook:

    import report
    report.build(results, ".")

onde ``results`` segue o formato montado na secao "Relatorio Final" do notebook.
"""
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------ paleta
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BASE_C = SERIES[0]     # Baseline
MED_C = SERIES[1]      # Medium
HARD_C = SERIES[2]     # Hard
VHARD_C = "#4a3aa7"    # Very Hard
LOSS_C = "#d03b3b"     # perda vs Cross Validation
GAIN_C = "#2a78d6"     # ganho vs Cross Validation
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
# uma rampa por dataset de teste: tom claro (Baseline) ao escuro (Very Hard)
RAMPS = [["#86b6ef", "#3987e5", "#2a78d6", "#1c5cab"],   # IMDB
         ["#f6a983", "#f08a5a", "#eb6834", "#b94a22"],   # Yelp
         ["#7fd7b4", "#41c396", "#1baf7a", "#14855c"]]   # Twitter
SHORT = ["Base", "Med", "Hard", "V.Hard"]

ORDER = ["imdb", "yelp", "twitter"]
LABEL = {"imdb": "IMDB", "yelp": "Yelp", "twitter": "Twitter"}
NAMES = [LABEL[n] for n in ORDER]

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.grid": False,
    "font.size": 9,
})


def miles(n):
    return f"{n:,}".replace(",", ".")


# ------------------------------------------------------------- componentes
def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)


def caption(ax, text, y=-0.30, fontsize=8.5):
    """Legenda curta abaixo do grafico explicando a logica do experimento."""
    ax.text(0, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, color=MUTED)


def bars(ax, names, values, title, colors=None, label=None):
    colors = colors or [BASE_C] * len(names)
    x = np.arange(len(names))
    ax.bar(x, values, width=0.55, color=colors, zorder=3, label=label)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=INK2, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Acuracia", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10, loc="left")
    style_axes(ax)


def pair_bars(ax, baseline, values, label, color, title, fontsize=9):
    """Baseline e challenge lado a lado, na cor do dataset de teste (mesmas cores do
    Cross Validation). O baseline e marcado com estrela e escrito embaixo da barra."""
    x = np.arange(len(NAMES))
    w = 0.34
    for g in range(len(NAMES)):
        for k, (vals, lab) in enumerate([(baseline, "★ Baseline"), (values, label)]):
            xi = x[g] + (k - 0.5) * w
            bar = ax.bar(xi, vals[g], width=w - 0.03, color=SERIES[g], zorder=3)[0]
            if k == 0:
                bar.set_hatch("//")
                bar.set_edgecolor("#ffffff")
                bar.set_linewidth(0)
            ax.text(xi, vals[g] + 0.02, f"{vals[g]:.2f}", ha="center", va="bottom",
                    fontsize=fontsize + 0.5, color=INK, fontweight="bold")
            ax.text(xi, vals[g] - 0.03, lab, ha="center", va="top", rotation=90,
                    fontsize=fontsize, color="#ffffff", fontweight="bold", zorder=4)
        ax.text(x[g], -0.04, f"Teste: {NAMES[g]}", ha="center", va="top",
                fontsize=fontsize + 1.5, color=INK, fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(NAMES) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acuracia", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10, loc="left")
    style_axes(ax)


def grouped_bars(ax, matrix, title, mark_baseline=True):
    """matrix[i][j] = acuracia treinando em ORDER[i] e testando em ORDER[j]."""
    x = np.arange(len(NAMES))
    w = 0.26
    diagonal = []
    for j, cname in enumerate(NAMES):
        vals = [matrix[i][j] for i in range(len(NAMES))]
        b = ax.bar(x + (j - 1) * w, vals, width=w - 0.02, color=SERIES[j],
                   label=f"Teste: {cname}", zorder=3)
        diagonal.append(b[j])
        for xi, v in zip(x + (j - 1) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Treino: {n}" for n in NAMES], color=INK2, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acuracia", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), labelcolor=INK2)
    if mark_baseline:
        for patch in diagonal:
            patch.set_edgecolor(INK)
            patch.set_linewidth(1.6)
    style_axes(ax)


def heatmap(ax, matrix, title, mark_baseline=True):
    m = np.array(matrix, dtype=float)
    vmin, vmax = 0.4, 1.0
    ax.imshow(m, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels([f"Teste\n{c}" for c in NAMES], color=INK2, fontsize=9.5)
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels([f"Treino\n{r}" for r in NAMES], color=INK2, fontsize=9.5)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            dark = (m[i, j] - vmin) / (vmax - vmin) > 0.55
            ink = "#ffffff" if dark else INK
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color=ink)
            if mark_baseline and i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=ink, linewidth=2.5, zorder=5))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10, loc="left")


def delta_bars(ax, cross, medium, title, fontsize=8, legend_y=-0.22):
    """Agrupado por dataset de teste, uma barra por dataset de treino (mesmas cores do
    Cross Validation), com a fatia vermelha do quanto falta para o Cross Validation.
    Quando nao falta nada, a barra vermelha nao aparece."""
    x = np.arange(len(NAMES))
    w = 0.26
    for i, tname in enumerate(NAMES):
        med = [medium[i][j] for j in range(3)]
        falta = [max(cross[i][j] - medium[i][j], 0.0) for j in range(3)]
        deltas = [(medium[i][j] - cross[i][j]) * 100 for j in range(3)]
        pos = x + (i - 1) * w
        ax.bar(pos, med, width=w - 0.02, color=SERIES[i], zorder=3,
               label=f"Treino: {tname}")
        ax.bar(pos, falta, bottom=med, width=w - 0.02, color=LOSS_C, zorder=3)
        for xi, m, f, d in zip(pos, med, falta, deltas):
            color = LOSS_C if d < 0 else (GAIN_C if d > 0 else MUTED)
            ax.text(xi, m + f + 0.015, f"{d:+.2f}", ha="center", va="bottom",
                    fontsize=fontsize - 0.5, color=color, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, color=LOSS_C))
    labels.append("Falta para o Cross Validation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Teste: {n}" for n in NAMES], color=INK2, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Acuracia", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10, loc="left")
    ax.legend(handles, labels, frameon=False, fontsize=fontsize, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, legend_y), labelcolor=INK2)
    style_axes(ax)


def table(ax, rows, col_labels, fontsize=9.5, title=None):
    ax.axis("off")
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=8, loc="left")
    t = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="upper center")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_facecolor(SURFACE if r % 2 else "#f7f7f5")
            cell.set_text_props(color=INK2)
        if c == 0 and r > 0:
            cell.set_text_props(color=INK, fontweight="bold")
    return t


# ------------------------------------------------------------------ figuras
CAP_BASE = "Baseline: treino e teste no mesmo dataset"
CAP_STAR = "★ Baseline: treino e teste no mesmo dataset"
CAP_DIAG = "Diagonal destacada = Baseline (treino e teste no mesmo dataset)"
CAP_LOSS = ("Fatia vermelha = quanto falta para a acuracia do Cross Validation (treino completo)\n"
            "Numero acima da barra = diferenca em pontos percentuais")


class Report:
    def __init__(self, results):
        R = self.R = results
        self.baseline = [R["baseline"][n] for n in ORDER]
        self.cross = [[R["cross"][t][s] for s in ORDER] for t in ORDER]
        self.medium = [[R["medium"][t][s] for s in ORDER] for t in ORDER]
        self.hard = [R["hard"][s] for s in ORDER]
        self.very = [R["very_hard"][s] for s in ORDER]
        # dataset que definiu o tamanho de treino usado no challenge Medium
        self.menor = LABEL[min(ORDER, key=lambda n: R["sizes"][n]["train"])]
        self.cap_medium = ("Medium: todos os treinos reduzidos a "
                           + miles(R["smallest_size"]) + " exemplos, tamanho do "
                           + self.menor + " (o menor dos tres)")
        self.cap_hard = ("Hard: treino com os 3 datasets concatenados ("
                         + miles(R["hard_train_size"]) + " exemplos)")
        self.cap_vhard = "Very Hard: treino com os outros 2 datasets, sem ver o de teste"
        self.comparativo_series = [
            (self.baseline, "Baseline", BASE_C),
            ([self.medium[i][i] for i in range(3)], "Medium", MED_C),
            (self.hard, "Hard", HARD_C),
            (self.very, "Very Hard", VHARD_C),
        ]

    def comparativo(self, ax, title, fontsize=8):
        """Uma cor por dataset de teste; dentro do grupo o tom escurece do Baseline
        ao Very Hard."""
        w = 0.21
        for g in range(3):
            for k, (vals, lab, _) in enumerate(self.comparativo_series):
                xi = g + (k - 1.5) * w
                ax.bar(xi, vals[g], width=w - 0.02, color=RAMPS[g][k], zorder=3)
                ax.text(xi, vals[g] + 0.015, f"{vals[g]:.2f}", ha="center", va="bottom",
                        fontsize=fontsize + 0.5, color=INK, fontweight="bold")
                ax.text(xi, -0.03, SHORT[k], ha="center", va="top", fontsize=fontsize,
                        color=INK2, transform=ax.get_xaxis_transform())
            ax.text(g, -0.16, f"Teste: {NAMES[g]}", ha="center", va="top", fontsize=11,
                    color=INK, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xticks([])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Acuracia", color=INK2)
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10, loc="left")
        style_axes(ax)

    def imagem_final(self, path=None):
        fig = plt.figure(figsize=(17, 12))
        gs = GridSpec(3, 6, figure=fig, hspace=0.9, wspace=0.9,
                      left=0.055, right=0.975, top=0.89, bottom=0.10)
        fig.suptitle("Relatorio Final: Sentiment Analysis Cross Dataset",
                     x=0.055, y=0.965, ha="left", fontsize=22, fontweight="bold", color=INK)
        fig.text(0.055, 0.93,
                 "TF-IDF + Regressao Logistica  |  IMDB, Yelp Polarity, Twitter Financial News",
                 ha="left", fontsize=12, color=INK2)

        ax = fig.add_subplot(gs[0, 0:2])
        bars(ax, NAMES, self.baseline, "1. Baseline")
        caption(ax, CAP_BASE)

        ax = fig.add_subplot(gs[0, 2:4])
        heatmap(ax, self.cross, "2. Cross Validation (3x3)")
        caption(ax, CAP_DIAG, y=-0.34)

        ax = fig.add_subplot(gs[0, 4:6])
        grouped_bars(ax, self.cross, "2. Cross Validation (barras)")
        caption(ax, "Barra com contorno preto = Baseline", y=-0.42)

        ax = fig.add_subplot(gs[1, 0:2])
        heatmap(ax, self.medium, "3. Medium: treinos igualados")
        caption(ax, self.cap_medium, y=-0.34)

        ax = fig.add_subplot(gs[1, 2:4])
        delta_bars(ax, self.cross, self.medium, "3. Medium: perda vs Cross Validation",
                   fontsize=8)
        caption(ax, CAP_LOSS, y=-0.44)

        ax = fig.add_subplot(gs[1, 4:6])
        pair_bars(ax, self.baseline, self.hard, "Hard", HARD_C, "4. Hard vs Baseline")
        caption(ax, self.cap_hard + "\n" + CAP_STAR, y=-0.30)

        ax = fig.add_subplot(gs[2, 0:2])
        pair_bars(ax, self.baseline, self.very, "Very Hard", VHARD_C,
                  "5. Very Hard vs Baseline")
        caption(ax, self.cap_vhard + "\n" + CAP_STAR, y=-0.30)

        ax = fig.add_subplot(gs[2, 2:6])
        self.comparativo(ax, "6. Comparativo por dataset de teste")
        caption(ax, "Uma cor por dataset de teste, do tom claro (Baseline) ao escuro (Very Hard)\n"
                "Base: treino e teste no mesmo dataset  |  Med: treinos igualados ao menor "
                "dataset  |  Hard: treino nos 3 juntos  |  V.Hard: treino nos outros 2",
                y=-0.32)

        if path:
            fig.savefig(path, dpi=170)
        return fig

    # ---------------------------------------------------------------- pdf
    @staticmethod
    def _page(title, subtitle=None):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.945, title, fontsize=19, fontweight="bold", color=INK, ha="left")
        if subtitle:
            fig.text(0.06, 0.905, subtitle, fontsize=11, color=INK2, ha="left")
        return fig

    @staticmethod
    def _save(pdf, fig):
        pdf.savefig(fig)
        debug = os.environ.get("REPORT_DEBUG")
        if debug:
            Report._page_n = getattr(Report, "_page_n", 0) + 1
            fig.savefig(os.path.join(debug, f"page{Report._page_n}.png"), dpi=110)
        plt.close(fig)

    def pdf(self, path):
        R = self.R
        with PdfPages(path) as pdf:
            # 1. datasets + baseline
            fig = self._page(
                "Relatorio Final: Sentiment Analysis Cross Dataset",
                "TF-IDF + Regressao Logistica  |  IMDB, Yelp Polarity, Twitter Financial News")
            ax = fig.add_axes([0.06, 0.58, 0.38, 0.20])
            table(ax, [[LABEL[n], miles(R["sizes"][n]["train"]), miles(R["sizes"][n]["test"])]
                       for n in ORDER], ["Dataset", "Treino", "Teste"], title="Datasets")
            ax = fig.add_axes([0.06, 0.24, 0.38, 0.20])
            f1 = R.get("baseline_f1", {})
            if f1:
                table(ax, [[LABEL[n], f"{R['baseline'][n]:.2f}", f"{f1[n]:.2f}"] for n in ORDER],
                      ["Dataset", "Acuracia", "F1 macro"], title="Baseline")
            else:
                table(ax, [[LABEL[n], f"{R['baseline'][n]:.2f}"] for n in ORDER],
                      ["Dataset", "Acuracia"], title="Baseline")
            ax = fig.add_axes([0.55, 0.20, 0.39, 0.56])
            bars(ax, NAMES, self.baseline, "Baseline")
            caption(ax, CAP_BASE, y=-0.14)
            self._save(pdf, fig)

            # 2. cross validation
            fig = self._page("Cross Validation", "Treino em 1 dataset, teste nos 3")
            ax = fig.add_axes([0.06, 0.20, 0.36, 0.58])
            heatmap(ax, self.cross, "Acuracia (treino x teste)")
            caption(ax, CAP_DIAG, y=-0.16)
            ax = fig.add_axes([0.56, 0.20, 0.38, 0.58])
            grouped_bars(ax, self.cross, "Acuracia por dataset de teste")
            caption(ax, "Barra com contorno preto = Baseline", y=-0.20)
            self._save(pdf, fig)

            # 3. medium
            fig = self._page("Challenge Medium",
                             "Todos os treinos reduzidos ao tamanho do menor dataset ("
                             + self.menor + "), n = " + miles(R["smallest_size"]))
            ax = fig.add_axes([0.06, 0.32, 0.32, 0.48])
            heatmap(ax, self.medium, "Acuracia com treinos igualados")
            caption(ax, CAP_DIAG, y=-0.18)
            ax = fig.add_axes([0.51, 0.32, 0.43, 0.48])
            delta_bars(ax, self.cross, self.medium, "Perda vs Cross Validation",
                       fontsize=9, legend_y=-0.16)
            caption(ax, CAP_LOSS, y=-0.36)
            self._save(pdf, fig)

            # 4. hard + very hard
            fig = self._page("Challenges Hard e Very Hard",
                             "Cada challenge comparado com o Baseline do mesmo dataset de teste")
            ax = fig.add_axes([0.06, 0.24, 0.38, 0.54])
            pair_bars(ax, self.baseline, self.hard, "Hard", HARD_C, "Hard vs Baseline")
            caption(ax, self.cap_hard + "\n" + CAP_STAR, y=-0.26)
            ax = fig.add_axes([0.56, 0.24, 0.38, 0.54])
            pair_bars(ax, self.baseline, self.very, "Very Hard", VHARD_C,
                      "Very Hard vs Baseline")
            caption(ax, self.cap_vhard + "\n" + CAP_STAR, y=-0.26)
            self._save(pdf, fig)


def build(results, outdir=".", png="relatorio_final.png", pdf="relatorio_final.pdf"):
    """Gera a imagem consolidada e o PDF; retorna os caminhos gerados."""
    rep = Report(results)
    png_path = os.path.join(outdir, png)
    pdf_path = os.path.join(outdir, pdf)
    fig = rep.imagem_final(png_path)
    plt.close(fig)
    rep.pdf(pdf_path)
    return png_path, pdf_path


def _flatten(raw):
    """Converte um results.json com classification_report completo no formato de build()."""
    acc = lambda r: r["accuracy"]
    return {
        "sizes": raw["sizes"],
        "smallest_size": raw["smallest_size"],
        "hard_train_size": raw["hard_train_size"],
        "baseline": {n: acc(raw["baseline"][n]) for n in ORDER},
        "baseline_f1": {n: raw["baseline"][n]["macro avg"]["f1-score"] for n in ORDER},
        "cross": {t: {s: acc(raw["cross"][t][s]) for s in ORDER} for t in ORDER},
        "medium": {t: {s: acc(raw["medium"][t][s]) for s in ORDER} for t in ORDER},
        "hard": {s: acc(raw["hard"][s]) for s in ORDER},
        "very_hard": {s: acc(raw["very_hard"][s]) for s in ORDER},
    }


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    raw = json.load(open(src, encoding="utf-8"))
    print(*build(_flatten(raw), out), sep="\n")
