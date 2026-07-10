import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import shapiro, ttest_rel, friedmanchisquare, wilcoxon, spearmanr, pearsonr
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from itertools import combinations

CARTELLA_OUT = "C:/Users/tomas/Desktop/universita/Magistrale/tesi/Analisi Statistiche Finali/"

def retrieve_data():
    dati = pd.read_csv(r"C:/Users/tomas/Desktop/universita/Magistrale/tesi/ipotesi dataset.CSV", sep = ";", encoding = "utf-8")
    fissazioni = dati.drop(columns = ['Ultima Visita (anni)', 'Ultima Visita Anni Minimi', 'VDU',  'Anamnesi', 'Utilizzo di Lac', 'RX', 'Valori Autoref.','Acuità Visiva Naturale da LONTANO','Acuità Visiva con correzione da LONTANO', 'Distanza (cm)', 'Saccadi'])
    saccadi = dati.drop(columns = ['Ultima Visita (anni)', 'Ultima Visita Anni Minimi', 'VDU',  'Anamnesi', 'Utilizzo di Lac', 'RX', 'Valori Autoref.','Acuità Visiva Naturale da LONTANO','Acuità Visiva con correzione da LONTANO', 'Distanza (cm)', 'Fissazioni'])
    fissazioni = fissazioni.dropna()
    saccadi = saccadi.dropna()
    return fissazioni, saccadi

def crea_dataset(df, colonna_json):
    righe = []
    for _, row in df.iterrows():
        if pd.notna(row[colonna_json]) and row[colonna_json] != "":
            dati = json.loads(row[colonna_json])
            for valori in dati.values():
                righe.append({
                    "ID": row["ID"],
                    "font_size": valori["font_size"],
                    "numero": valori["numero"],
                    "durata_totale_sec": valori["durata_totale_sec"],
                    "durata_media_sec": valori["durata_media_sec"]
                })
    return pd.DataFrame(righe)

def descriptive_analisys(df, feature):
    statistiche = df.groupby("font_size")[feature].agg(media="mean", deviazione_standard="std", mediana="median", minimo="min", massimo="max", varianza="var").reset_index()
    statistiche["q1"] = df.groupby("font_size")[feature].quantile(0.25).values
    statistiche["q3"] = df.groupby("font_size")[feature].quantile(0.75).values
    statistiche["iqr"] = statistiche["q3"] - statistiche["q1"]
    return statistiche

def grafici(df, df_raw, nome_misura, feature):
    """
    df: dataframe aggregato prodotto da descriptive_analisys()
    df_raw: dataframe originale
    nome_misura: 'Fissazioni' oppure 'Saccadi'
    feature: 'numero' oppure 'durata_totale_sec'
    """

    df = df.sort_values("font_size", ascending=False)
    df_raw = df_raw.sort_values("font_size", ascending=False)

    ordine_font = sorted(df_raw["font_size"].unique(), reverse=True)

    if feature == "numero":
        y_label = f"Numero di {nome_misura.lower()}"
        nome_feature = "numero"
    elif feature == "durata_totale_sec":
        y_label = f"Durata totale {nome_misura.lower()} (s)"
        nome_feature = "durata totale"
    else:
        y_label = feature
        nome_feature = feature

    titolo_base = f"{nome_misura} - {nome_feature}"

    # 1. Grafico a linea delle medie
    plt.figure(figsize=(14, 5))
    plt.plot(df["font_size"], df["media"], marker="o")
    plt.xlabel("Dimensione font")
    plt.ylabel(y_label)
    plt.title(f"{nome_misura}: media di {nome_feature} per dimensione del font")
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)

    percorso_completo = os.path.join(CARTELLA_OUT,titolo_base + " linea medie.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Barplot semplice delle medie
    plt.figure(figsize=(14, 5))
    sns.barplot(data=df, x="font_size", y="media", hue="font_size", order=ordine_font, hue_order=ordine_font, legend=False)
    plt.xlabel("Dimensione font")
    plt.ylabel(y_label)
    plt.title(f"{nome_misura}: media di {nome_feature} per dimensione del font")

    percorso_completo = os.path.join(CARTELLA_OUT, titolo_base + " bar plot.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Grafico delle medie con deviazione standard
    plt.figure(figsize=(14, 5))
    plt.bar(df["font_size"], df["media"], yerr=df["deviazione_standard"], capsize=5)
    plt.xlabel("Dimensione font")
    plt.ylabel(y_label)
    plt.title(f"{nome_misura}: media di {nome_feature} con deviazione standard")
    plt.gca().invert_xaxis()

    percorso_completo = os.path.join(CARTELLA_OUT, titolo_base + " barre errore.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches="tight")
    plt.show()

    # 4. Boxplot reale usando il dataframe originale
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df_raw, x="font_size", y=feature, order=ordine_font)
    plt.xlabel("Dimensione font")
    plt.ylabel(y_label)
    plt.title(f"{nome_misura}: boxplot di {nome_feature} per dimensione del font")

    percorso_completo = os.path.join( CARTELLA_OUT, titolo_base + " boxplot reale.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches="tight")
    plt.show()

def shapiro_test(fissazioni, saccadi, feature):
    pvalue_fix = []
    pvalue_sac = []
    print("Fissazioni")
    for font in fissazioni["font_size"].unique():
        valori = fissazioni[fissazioni["font_size"] == font][feature]
        _, p = shapiro(valori)
        pvalue_fix.append(p)
        print(f"Font {font}: p-value = {p:.4f}")

    print("\nSaccadi")
    for font in saccadi["font_size"].unique():
        valori = saccadi[saccadi["font_size"] == font][feature]
        _, p = shapiro(valori)
        pvalue_sac.append(p)
        print(f"Font {font}: p-value = {p:.4f}")

    rispetta_shapiro = all(n < 0.05 for n in pvalue_fix) and all(n < 0.05 for n in pvalue_sac)
    if rispetta_shapiro:
        print("Shapiro rispettato: normalità rispettata")
    else:
        print("Shapiro non rispettato: normalità non rispettata")
    
    return rispetta_shapiro

def anova_misure_ripetute(df, nome, feature):
    print(f"\nANOVA a misure ripetute - {nome}")
    anova = AnovaRM(data=df, depvar=feature, subject="ID", within=["font_size"]).fit()
    print(anova)
    return anova

def posthoc_ttest_appaiato(df, nome, feature, metodo_correzione="bonferroni"):
    print(f"\nPost-hoc t-test appaiato - {nome}")
    print(f"Correzione multipla: {metodo_correzione}")

    # Trasformo il dataframe in formato wide:
    # righe = soggetti, colonne = font_size, valori = numero
    df_wide = df.pivot(index="ID", columns="font_size", values= feature)

    # Tengo solo soggetti che hanno tutte le condizioni
    df_wide = df_wide.dropna()

    font_sizes = sorted(df_wide.columns, reverse=True)

    risultati = []

    for font1, font2 in combinations(font_sizes, 2):
        valori1 = df_wide[font1]
        valori2 = df_wide[font2]

        stat, p = ttest_rel(valori1, valori2)

        risultati.append({
            "confronto": f"{font1} vs {font2}",
            "font_1": font1,
            "font_2": font2,
            "t_statistic": stat,
            "p_value": p
        })
    risultati_df = pd.DataFrame(risultati)

    # Correzione per confronti multipli
    reject, p_corrected, _, _ = multipletests(risultati_df["p_value"], method=metodo_correzione)
    risultati_df["p_value_corretto"] = p_corrected
    risultati_df["significativo"] = reject
    print(risultati_df)
    percorso_csv = os.path.join(CARTELLA_OUT, f"Posthoc t-test appaiato {nome} {metodo_correzione}.csv"    )
    risultati_df.to_csv(percorso_csv, sep=";", index=False, encoding="utf-8")
    return risultati_df

def friedman_misure_ripetute(df, nome, feature):
    print(f"\nTest di Friedman per {nome}")

    df_wide = df.pivot(index="ID", columns="font_size", values= feature)
    df_wide = df_wide.dropna()

    font_sizes = sorted(df_wide.columns, reverse=True)

    dati_font = [df_wide[font] for font in font_sizes]

    stat, p = friedmanchisquare(*dati_font)

    print(f"Statistica Friedman: {stat:.4f}")
    print(f"p-value: {p:.4f}")
    if p < 0.05:
        print("Risultato significativo: ci sono differenze tra almeno due font size.")
    else:
        print("Risultato non significativo: non emergono differenze significative tra i font size.")

    return p

def wilcoxon_significativita(df, stringa, feature):
    df_wide = df.pivot(index="ID", columns="font_size", values= feature)
    df_wide = df_wide.dropna()

    _, p = wilcoxon(df_wide[14], df_wide[7])

    print("p-value:", p)
    if p < 0.05:
        print("Risultato significativo per le " + stringa + ".")
    else:
        print("Risultato non significativo per le " + stringa + ".")

def posthoc_wilcoxon_appaiato(df, nome, feature, metodo_correzione="fdr_bh"):
    fonts = sorted(df["font_size"].unique(), reverse=True)
    risultati = []

    for font_1, font_2 in combinations(fonts, 2):
        dati_1 = df[df["font_size"] == font_1].sort_values("ID")[["ID", feature]]
        dati_2 = df[df["font_size"] == font_2].sort_values("ID")[["ID", feature]]

        dati = pd.merge(
            dati_1,
            dati_2,
            on="ID",
            suffixes=("_1", "_2")
        ).dropna()

        stat, p = wilcoxon(
            dati[f"{feature}_1"],
            dati[f"{feature}_2"]
        )

        risultati.append({
            "confronto": f"{font_1} vs {font_2}",
            "font_1": font_1,
            "font_2": font_2,
            "w_statistic": stat,
            "p_value": p
        })

    risultati_df = pd.DataFrame(risultati)

    risultati_df["p_value_corretto"] = multipletests(
        risultati_df["p_value"],
        method=metodo_correzione
    )[1]

    risultati_df["significativo"] = risultati_df["p_value_corretto"] < 0.05

    print(f"Post-hoc Wilcoxon appaiato - {nome}")
    print(f"Correzione multipla: {metodo_correzione}")
    print(risultati_df)

    return risultati_df

def correlazione_spearman(fissazioni, saccadi):
    print()
    print("INIZIO CALCOLO CORRELAZIONE DI SPEARMAN")
    variabili = ["numero", "durata_totale_sec", "durata_media_sec"]

    for var in variabili:
        rho_fix, p_fix = spearmanr(fissazioni["font_size"], fissazioni[var])
        rho_sac, p_sac = spearmanr(saccadi["font_size"], saccadi[var])

        print(f"\nVariabile: {var}")

        print("Fissazioni")
        print(f"rho = {rho_fix:.4f}, p = {p_fix:.4f}")

        print("Saccadi")
        print(f"rho = {rho_sac:.4f}, p = {p_sac:.4f}")

def r_quadro(fissazioni, saccadi):
    print()
    print("INIZIO CALCOLO R²")
    variabili = ["numero", "durata_totale_sec", "durata_media_sec"]

    for var in variabili:
        rho_fix, p_fix = pearsonr(fissazioni["font_size"], fissazioni[var])
        rho_sac, p_sac = pearsonr(saccadi["font_size"], saccadi[var])

        print(f"\nVariabile: {var}")

        print("Fissazioni")
        print(f"R² = {rho_fix**2:.4f}")
        print(f"p-value = {p_fix:.4f}")

        print("Saccadi")
        print(f"R² = {rho_sac**2:.4f}")
        print(f"p-value = {p_sac:.4f}")

#durata_totale_sec
feature = "numero"
fissazioni_JSON_form, saccadi_JSON_form = retrieve_data()
fissazioni = crea_dataset(fissazioni_JSON_form, "Fissazioni")
saccadi = crea_dataset(saccadi_JSON_form, "Saccadi")
print(fissazioni.columns)
print("FEATURE ANALIZZATA: "+ feature)
statistiche_descrittive_fix = descriptive_analisys(fissazioni, feature)
statistiche_descrittive_sax = descriptive_analisys(saccadi, feature)
print("STATISTICA DESCRITTIVA DI " + feature + " RIGUARDO LE FISSAZIONI")
print(statistiche_descrittive_fix)
print("STATISTICA DESCRITTIVA DI " + feature + " RIGUARDO LE SACCADI")
print(statistiche_descrittive_sax)
#grafici(statistiche_descrittive_fix, fissazioni, "Fissazioni", feature)
#grafici(statistiche_descrittive_sax, saccadi, "Saccadi", feature)

risultato = shapiro_test(fissazioni, saccadi, feature)

if risultato:
    print("\nNormalità rispettata: procedo con ANOVA a misure ripetute")
    anova_fix = anova_misure_ripetute(fissazioni, "Fissazioni", feature)
    anova_sac = anova_misure_ripetute(saccadi, "Saccadi", feature)

    posthoc_fix_bonferroni = posthoc_ttest_appaiato(fissazioni, "Fissazioni", feature, metodo_correzione="bonferroni")
    posthoc_sac_bonferroni = posthoc_ttest_appaiato(saccadi, "Saccadi", feature, metodo_correzione="bonferroni")

    print("\n")
    posthoc_fix_fdr = posthoc_ttest_appaiato(fissazioni, "Fissazioni", feature, metodo_correzione="fdr_bh")
    print("\n")
    posthoc_sac_fdr = posthoc_ttest_appaiato(saccadi, "Saccadi", feature, metodo_correzione="fdr_bh", )

else:
    print("\nNormalità non rispettata: procedo con Friedman.")
    friedman_fix = friedman_misure_ripetute(fissazioni, "Fissazioni", feature)
    fredman_sac = friedman_misure_ripetute(saccadi, "Saccadi", feature)

    print("\nSignificatività con Wilcoxon")
    wilcoxon_fix = wilcoxon_significativita(fissazioni, "Fissazioni", feature)
    wilcoxon_sac = wilcoxon_significativita(saccadi, "Saccadi", feature)

    print("\n")
    posthoc_fix_fdr = posthoc_wilcoxon_appaiato(fissazioni, "Fissazioni", feature, metodo_correzione="fdr_bh")
    print("\n")
    posthoc_sac_fdr = posthoc_wilcoxon_appaiato(saccadi, "Saccadi", feature, metodo_correzione="fdr_bh")

correlazione_spearman(fissazioni, saccadi)
r_quadro(fissazioni, saccadi)