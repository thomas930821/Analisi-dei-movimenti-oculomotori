import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from EyeTrackLib.calculate_streams.saccades_and_fixations import calculate_saccades_and_fixations
from scipy.signal import find_peaks
import peyemmv as mmv
from detectors import fixation_detection
import pymovements as pm
from detectors import saccade_detection

# mappatura paragrafo → font size (da definire con i valori reali)
FONT_MAP = {
    1: 14,
    2: 13,
    3: 12,
    4: 11,
    5: 10,
    6:  9,
    7:  8,
    8:  7
}
N_JUMPS_EXPECTED = 87
LINES_PER_PARAGRAPH = 11

#CARTELLA_OUT = "C:/Users/tomas/Desktop/universita\Magistrale/tesi/Esperimento su persone/000/20260430T152737Z, 30 cm/immagini/"
#CARTELLA_OUT = "C:/Users/tomas/Desktop/universita\Magistrale/tesi/Esperimento su persone/000/20260430T153924Z, 35 cm/immagini/"
#CARTELLA_OUT = "C:/Users/tomas/Desktop/universita\Magistrale/tesi/Esperimento su persone/000/20260430T155518Z, 40 cm/immagini/"
CARTELLA_OUT = "C:/Users/tomas/Desktop/universita/Magistrale/tesi/Esperimento su persone/004/immagini/"

CARTELLA_OUT_FIX = "C:/Users/tomas/Desktop/universita/Magistrale/tesi\paragoni/fissazioni_paragoni/"
CARTELLA_OUT_REG = "C:/Users/tomas/Desktop/universita/Magistrale/tesi\paragoni/regressioni_paragoni/"
CARTELLA_OUT_SAC = "C:/Users/tomas/Desktop/universita/Magistrale/tesi\paragoni/saccadi_paragoni/"
CARTELLA_OUT_PROX = "C:/Users/tomas/Desktop/universita/Magistrale/tesi\paragoni/proxy/"


def read_json():
    #eye_tracker_JSON = pd.read_json(r"C:/Users/tomas/Desktop/universita/Magistrale/tesi/Esperimento su persone/000/20260430T152737Z, 30 cm/gazedata/gazedata", lines = True)
    #eye_tracker_JSON = pd.read_json(r"C:/Users/tomas/Desktop/universita/Magistrale/tesi/Esperimento su persone/000/20260430T153924Z, 35 cm/gazedata/gazedata", lines = True)
    #eye_tracker_JSON = pd.read_json(r"C:/Users/tomas/Desktop/universita/Magistrale/tesi/Esperimento su persone/000/20260430T155518Z, 40 cm/gazedata/gazedata", lines = True)
    eye_tracker_JSON = pd.read_json(r"C:/Users/tomas/Desktop/universita/Magistrale/tesi/Esperimento su persone/004/gazedata/gazedata", lines = True)
    columns = eye_tracker_JSON.columns
    columns = eye_tracker_JSON.columns
    #eye_tracker = pd.DataFrame(eye_tracker_JSON, columns= columns)
    eye_tracker = pd.json_normalize(eye_tracker_JSON.to_dict(orient="records"))
    #print(eye_tracker.columns)
    return eye_tracker

def plot_distribution(gaze_new, stringa):
    sns.scatterplot(gaze_new, x="timestamp", y="gaze2d_y", hue = "timestamp")
    plt.ylim(0, 1)
    plt.title(stringa)
    percorso_completo = os.path.join(CARTELLA_OUT, stringa + "_y.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
    plt.show()

    sns.scatterplot(gaze_new, x="timestamp", y="gaze2d_x", hue = "timestamp")
    plt.ylim(0, 1)
    plt.title(stringa)
    percorso_completo = os.path.join(CARTELLA_OUT, stringa+ "_x.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
    plt.show()

    sns.scatterplot(gaze_new, x="gaze2d_x", y="gaze2d_y", hue="timestamp")
    plt.xlim(0.4, 0.6)
    plt.ylim(0.8, 0.2)   # spesso utile perché negli schermi y=0 è in alto
    plt.title(stringa)
    percorso_completo = os.path.join(CARTELLA_OUT, stringa+ "_x_y.png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
    plt.show()

def gaze_info(gaze):
    gaze = gaze.drop(columns = ['type', 'data.eyeleft.gazeorigin', 'data.eyeleft.gazedirection', 'data.eyeleft.pupildiameter', 'data.eyeright.gazeorigin',
        'data.eyeright.gazedirection', 'data.eyeright.pupildiameter'])
    
    #tolgo inconsistenze di occhi chiusi e dati non presi
    gaze = gaze.dropna(subset=["data.gaze2d", "timestamp", "data.gaze3d"])
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)

    gaze_new = gaze.copy()
    gaze_new[["gaze2d_x", "gaze2d_y"]] = pd.DataFrame( gaze["data.gaze2d"].tolist(), index=gaze.index )

    #verifico distribuzione iniziale prima di eventuali vincoli
    stringa = "distribuzione dei dati prima del refining"
    plot_distribution(gaze_new, stringa)

    #va prima analizzato il plot per poter descrivere i limiti inferiori e superiori di x e y
    mask = (gaze_new["gaze2d_x"] >= 0.2) & (gaze_new["gaze2d_x"] <= 0.7)
    gaze_new = gaze_new[mask].copy()
    mask = (gaze_new["gaze2d_y"] >= 0.2) & (gaze_new["gaze2d_y"] <= 0.8)
    gaze_new = gaze_new[mask].copy()
    return gaze_new

def analyse_noise(gaze):
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)
    
    # velocità in unità schermo per secondo
    dt = gaze["timestamp"].diff()
    dx = gaze["gaze2d_x"].diff()
    dy = gaze["gaze2d_y"].diff()
    
    gaze["speed"] = (dx**2 + dy**2)**0.5 / dt
    
    # distribuzione delle velocità
    print(gaze["speed"].describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.99]))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # istogramma velocità
    axes[0].hist(gaze["speed"].dropna(), bins=100, edgecolor="none")
    axes[0].set_xlabel("velocità (unità/s)")
    axes[0].set_ylabel("conteggio")
    axes[0].set_title("Distribuzione velocità oculare")
    axes[0].set_yscale("log")  # log per vedere anche la coda
    
    # velocità nel tempo
    axes[1].plot(gaze["timestamp"], gaze["speed"], linewidth=0.5)
    axes[1].set_xlabel("timestamp")
    axes[1].set_ylabel("velocità")
    axes[1].set_title("Velocità nel tempo")
    
    plt.tight_layout()
    plt.show()
    
    return gaze

def filter_by_speed(gaze, percentile=99):
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)
    
    dt = gaze["timestamp"].diff()
    dx = gaze["gaze2d_x"].diff()
    dy = gaze["gaze2d_y"].diff()
    gaze["speed"] = (dx**2 + dy**2)**0.5 / dt

    # soglia = percentile scelto, calcolata dai dati stessi → generica per tutti i dataset
    threshold = gaze["speed"].quantile(percentile / 100)
    print(f"Soglia al {percentile}° percentile: {threshold:.4f} unità/s")
    
    n_before = len(gaze)
    gaze_clean = gaze[gaze["speed"] < threshold].copy()
    n_after = len(gaze_clean)
    print(f"Punti rimossi: {n_before - n_after} ({100*(n_before-n_after)/n_before:.1f}%)")
    
    return gaze_clean

def noise_removal(gaze):
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)
    gaze["x_smooth"] = gaze["gaze2d_x"].rolling(window=15, center=True, min_periods=1).median()
    gaze["y_smooth"] = gaze["gaze2d_y"].rolling(window=25, center=True, min_periods=1).median()
    return gaze

def calibrate_and_validate_jumps(jumps, gaze, n_expected, window_sec=1.5):
    # aggiunge sempre jump_idx, anche se il numero di salti è già quello atteso
    jumps = [
        {**jump, "jump_idx": jump_idx}
        for jump_idx, jump in enumerate(jumps, start=1)
    ]



    annotated = []

    for jump in jumps:
        t_start = jump["timestamp_inizio"]
        t_end   = jump["timestamp_fine"]

        pre  = gaze[(gaze["timestamp"] >= t_start - window_sec) & (gaze["timestamp"] < t_start)]
        post = gaze[(gaze["timestamp"] >  t_end) & (gaze["timestamp"] <= t_end + window_sec)]

        if len(pre) < 5 or len(post) < 5:
            dx, dy = 0.0, 0.0
        else:
            dx = post["x_smooth"].median() - pre["x_smooth"].median()
            dy = post["y_smooth"].median() - pre["y_smooth"].median()

        annotated.append({
            **jump,
            "dx_validated": dx,
            "dy_validated": dy
        })

    annotated_sorted = sorted(annotated, key=lambda j: j["dx_validated"])

    if len(annotated_sorted) >= n_expected:
        valid = annotated_sorted[:n_expected]
        valid = sorted(valid, key=lambda j: j["timestamp_inizio"])
        threshold_used = valid[-1]["dx_validated"]

        print(f"Soglia dx calibrata automaticamente: {threshold_used:.4f}")
        print(f"Salti validi selezionati: {len(valid)} / {len(annotated)}")
    else:
        valid = sorted(annotated_sorted, key=lambda j: j["timestamp_inizio"])
        print(f"⚠️ Trovati solo {len(valid)} candidati, mancano {n_expected - len(valid)} salti")

    return valid

def jumps_(id_x, gaze, search_sec=1.0):
    jumps = []
    search_samples = int(search_sec * 50)  # 50Hz → 50 campioni per secondo
    for id in id_x:
        if id + search_samples >= len(gaze):
            continue
        inizio = gaze.iloc[id]

        # cerca il minimo di x_smooth nella finestra dopo il salto
        # è il punto più a sinistra = fine del ritorno oculare
        window = gaze.iloc[id : id + search_samples]
        x_min_idx = window["x_smooth"].idxmin()
        fine = gaze.loc[x_min_idx]
        jumps.append({
            "timestamp_inizio": inizio["timestamp"],
            "timestamp_fine":   fine["timestamp"],
            "dist_x": fine["x_smooth"] - inizio["x_smooth"],
            "dist_y": fine["y_smooth"] - inizio["y_smooth"]
        })
    print("trovati", len(jumps), "salti")
    return jumps

def remove_duplicate_jumps(jumps, min_gap_sec=2.5):
    if not jumps:
        return jumps
    
    cleaned = [jumps[0]]
    for jump in jumps[1:]:
        prev = cleaned[-1]
        gap = jump["timestamp_inizio"] - prev["timestamp_inizio"]
        if gap < min_gap_sec:
            if abs(jump["dist_x"]) > abs(prev["dist_x"]):
                cleaned[-1] = jump
        else:
            cleaned.append(jump)
    print(f"Dopo rimozione duplicati: {len(cleaned)} salti")
    return cleaned

def find_jumps(gaze_new, target):
    gaze = gaze_new.copy()
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)
    FREQ_HZ = 50.0
    N = 15
    min_distance = int(2.5 * FREQ_HZ)
    gaze["dx_future"] = gaze["x_smooth"].shift(-N) - gaze["x_smooth"]
    best_jumps = None
    best_diff  = float("inf")

    for prominence in np.arange(0.02, 0.20, 0.005):
        id_x, _ = find_peaks(-gaze["dx_future"], prominence=prominence, distance=min_distance)
        jumps = jumps_(id_x, gaze)
        jumps = remove_duplicate_jumps(jumps, min_gap_sec=2.5)
        diff = abs(len(jumps) - target)
        if diff < best_diff:
            best_diff  = diff
            best_jumps = jumps
            best_prom  = prominence
        if diff == 0:
            break  # trovato esattamente target, stop
    print(f"Prominence ottimale: {best_prom:.3f} → {len(best_jumps)} salti (target={target})")
    if best_diff > 0:
        print(f"⚠️  Distanza dal target: {best_diff} salti")
    return gaze, best_jumps

def plot_detected_jumps(gaze, jumps, stringa):
    plt.figure(figsize=(14, 5))

    sns.scatterplot(data=gaze, x="timestamp", y="gaze2d_x", hue="timestamp", legend=False)

    for jump in jumps:
        plt.axvline(jump["timestamp_inizio"], linestyle="--")
        plt.axvline(jump["timestamp_fine"], linestyle=":")

    plt.title("gaze2d_x " + stringa)
    plt.ylim(0, 1)
    percorso_completo = os.path.join(CARTELLA_OUT, "gaze2d_x " + stringa + ".png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(14, 5))

    sns.scatterplot(data=gaze, x="timestamp", y="gaze2d_y", hue="timestamp", legend=False)

    for jump in jumps:
        plt.axvline(jump["timestamp_inizio"], linestyle="--")
        plt.axvline(jump["timestamp_fine"], linestyle=":")

    plt.title("gaze2d_y "+ stringa)
    plt.ylim(0, 1)
    percorso_completo = os.path.join(CARTELLA_OUT, "gaze2d_y "+ stringa+ ".png")
    plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
    plt.show()

def delete_inconsistencies_between_lines(gaze_new, jumps):
    paragraphs = gaze_new.copy()

    indices_to_drop = []
    for dictt in jumps:
        inizio = dictt["timestamp_inizio"]
        fine = dictt["timestamp_fine"]
        mask = (paragraphs["timestamp"] >= inizio) & (paragraphs["timestamp"] <= fine)
        indices_to_drop.extend(paragraphs.index[mask].tolist())

    paragraphs.drop(indices_to_drop, inplace=True, errors = "ignore")
    #paragraphs.to_csv("C:/Users/tomas/Desktop/universita/Magistrale/tesi/paragrafi.csv", sep=',', index=False, encoding='utf-8')
    return paragraphs

def find_lines(gaze_paragraphs, jumps_validated):
    gaze = gaze_paragraphs.copy()
    gaze["line"] = np.nan
    gaze["line_uncertain"] = False

    jumps_validated = sorted(jumps_validated, key=lambda j: j["timestamp_inizio"])

    previous_time = gaze["timestamp"].min()
    previous_jump_idx = 0

    for jump in jumps_validated:
        current_jump_idx = int(jump["jump_idx"])
        inizio = jump["timestamp_inizio"]

        mask = (gaze["timestamp"] >= previous_time) & (gaze["timestamp"] < inizio)

        if current_jump_idx == previous_jump_idx + 1:
            gaze.loc[mask, "line"] = current_jump_idx
            gaze.loc[mask, "line_uncertain"] = False
        else:
            # Qui significa che uno o più salti intermedi sono mancanti.
            # Meglio NON assegnare queste righe a un paragrafo sbagliato.
            gaze.loc[mask, "line"] = np.nan
            gaze.loc[mask, "line_uncertain"] = True

        previous_time = inizio
        previous_jump_idx = current_jump_idx

    # Dopo l'ultimo salto validato
    mask = gaze["timestamp"] >= previous_time
    gaze.loc[mask, "line"] = previous_jump_idx + 1
    gaze.loc[mask, "line_uncertain"] = False

    return gaze

def assign_paragraph_from_lines(gaze, lines_per_paragraph):
    gaze = gaze.copy()
    gaze["paragraph"] = np.nan

    mask = gaze["line"].notna()
    gaze.loc[mask, "paragraph"] = (
        ((gaze.loc[mask, "line"] - 1) // lines_per_paragraph) + 1
    )

    return gaze

def compute_gap_quality_report( gaze_paragraphs, paragraph_col="paragraph", t_col="timestamp", gap_multiplier=3, warning_max_gap=0.8, bad_gap_factor=2.0,bad_dt_factor=1.5):
    """
    Calcola la qualità dei dati gaze per paragrafo.

    Restituisce:
        gaze_out   = gaze_paragraphs con colonne dt, gap_flag, quality_label
        gap_report = report qualità per paragrafo
    """
    gaze = gaze_paragraphs.copy()
    gaze = gaze.sort_values(t_col).reset_index(drop=True)

    # dt tra campioni consecutivi
    gaze["dt"] = gaze[t_col].diff()

    # frequenza tipica del partecipante
    sampling_dt = gaze["dt"].median()

    # gap = intervallo temporale molto più grande del normale
    gaze["gap_flag"] = gaze["dt"] > gap_multiplier * sampling_dt

    gap_report = (gaze.groupby(paragraph_col).agg(n_points=(t_col, "count"), n_gaps=("gap_flag", "sum"), max_gap=("dt", "max"), median_dt=("dt", "median")).reset_index())

    # warning: almeno un buco temporale molto grande
    gap_report["quality_warning"] = (gap_report["max_gap"] > warning_max_gap)

    # bad: tanti gap rispetto agli altri paragrafi
    # oppure campionamento effettivo peggiorato
    gap_report["quality_bad"] = ((gap_report["n_gaps"] > gap_report["n_gaps"].median() * bad_gap_factor) | (gap_report["median_dt"] > gap_report["median_dt"].median() * bad_dt_factor))

    # etichetta finale
    gap_report["quality_label"] = "ok"

    gap_report.loc[gap_report["quality_warning"], "quality_label"] = "warning"

    gap_report.loc[gap_report["quality_bad"], "quality_label"] = "bad"

    # porta il flag dentro gaze_paragraphs
    gaze = gaze.merge(gap_report[[paragraph_col, "quality_label"]], on=paragraph_col, how="left")
    return gaze, gap_report

def plot_paragraphs_time(gaze_paragraphs):
    gaze = gaze_paragraphs.copy()

    max_paragraph = int(gaze["paragraph"].max())
    print("numero paragrafi:", max_paragraph)

    for paragraph_id in range(1, max_paragraph + 1):
        paragraph_data = gaze[gaze["paragraph"] == paragraph_id].copy()

        if len(paragraph_data) == 0:
            print(f"Paragrafo {paragraph_id} vuoto")
            continue

        plt.figure(figsize=(12, 4))
        sns.scatterplot(data=paragraph_data, x="timestamp", y="gaze2d_x", hue="line", palette="tab10")

        plt.ylim(0, 1)
        plt.title(f"Paragrafo {paragraph_id} - gaze2d_x nel tempo")
        plt.xlabel("timestamp")
        plt.ylabel("gaze2d_x")
        plt.legend(title="line", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        percorso_completo = os.path.join(CARTELLA_OUT, f"paragrafo_{paragraph_id}.png")
        plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
        plt.show()

def detracting_eye_stabilization_first_paragraph(gaze_paragraphs, first_row_timestamp):
    gaze_cpy = gaze_paragraphs.copy()
    gaze_return = gaze_cpy
    mask = gaze_cpy["paragraph"] == 1
    first_paragraph = gaze_cpy[mask] 
    mask = first_paragraph["timestamp"] <= first_row_timestamp
    first_row = first_paragraph[mask]
    id_x, _ = find_peaks(-first_row["gaze2d_x"], prominence = 0.04, distance = 5)
    jump = []
    jump = jumps_(id_x, first_row, 5)
    if len(jump) != 0:
        #plot_detected_jumps(first_row, jump)
        timestamp_inizio = jump[-1]["timestamp_inizio"]
        mask = gaze_cpy["timestamp"] >= timestamp_inizio
        gaze_return = gaze_cpy[mask]
    return gaze_return

def analysis_line(gaze):
    gaze = gaze.copy()

    # considera solo righe/paragrafi assegnati con certezza
    gaze = gaze.dropna(subset=["paragraph", "line"])

    gaze["paragraph"] = gaze["paragraph"].astype(int)
    gaze["line"] = gaze["line"].astype(int)

    max_paragraph = int(gaze["paragraph"].max())

    for paragraph_id in range(1, max_paragraph + 1):
        paragraph_data = gaze[gaze["paragraph"] == paragraph_id].copy()
        unique_lines = sorted(paragraph_data["line"].unique())

        for i, line_id in enumerate(unique_lines):
            line_data = paragraph_data[paragraph_data["line"] == line_id].copy()
            line_data = line_data.sort_values("timestamp").reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.set_title(f"Paragrafo {paragraph_id} - Riga {i+1} (line_id={line_id})")
            ax.set_xlabel("x_smooth")
            ax.set_ylabel("timestamp")

            xs = line_data["x_smooth"].values
            ys = line_data["timestamp"].values

            ax.scatter(xs, ys, s=15, zorder=3)

            step = 1
            for j in range(0, len(xs) - step, step):
                ax.annotate(
                    "",
                    xy=(xs[j + step], ys[j + step]),
                    xytext=(xs[j], ys[j]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="steelblue",
                        lw=1.0
                    )
                )

            plt.tight_layout()

            percorso_completo = os.path.join(
                CARTELLA_OUT,
                f"paragrafo_{paragraph_id}_riga{i+1}_lineid_{line_id}.png"
            )
            plt.savefig(percorso_completo, dpi=300, bbox_inches="tight")
            plt.close()

def compute_line_stats(gaze):
    gaze = gaze.copy()

    # tiene solo righe assegnate con certezza
    gaze = gaze.dropna(subset=["line", "paragraph", "x_smooth", "timestamp"])

    gaze["line"] = gaze["line"].astype(int)
    gaze["paragraph"] = gaze["paragraph"].astype(int)

    results = []

    for line_id in sorted(gaze["line"].unique()):
        line_data = gaze[gaze["line"] == line_id].copy()
        line_data = line_data.sort_values("timestamp").reset_index(drop=True)

        # sicurezza: se una riga ha meno di 2 punti, non calcolo statistiche
        if len(line_data) < 2:
            print(f"Riga {line_id} saltata: punti insufficienti")
            continue

        xs = line_data["x_smooth"].values
        ts = line_data["timestamp"].values

        durata = ts[-1] - ts[0]

        if durata <= 0:
            print(f"Riga {line_id} saltata: durata non valida")
            continue

        dx = np.diff(xs)
        x_range = xs.max() - xs.min()
        x_progressione_netta = xs[-1] - xs[0]
        velocita_media = x_progressione_netta / durata
        x_percorso_totale = np.abs(dx).sum()
        velocita_totale = x_percorso_totale / durata

        soglia_regressione = 0.01
        n_regressioni = (dx < -soglia_regressione).sum()

        results.append({
            "line":                 line_id,
            "paragraph":            line_data["paragraph"].iloc[0],
            "durata_totale_sec":    round(durata, 3),
            "x_range":              round(x_range, 4),
            "x_progressione_netta": round(x_progressione_netta, 4),
            "velocita_media":       round(velocita_media, 4),
            "velocita_totale":      round(velocita_totale, 4),
            "n_regressioni":        int(n_regressioni),
            "n_punti":              len(line_data)
        })

    df = pd.DataFrame(results)
    print(df.to_string())
    return df

def fixations_analysis(gaze, fixations):
    # aggiungi paragrafo ad ogni fissazione in base al timestamp
    def get_paragraph(t):
        row = gaze[gaze["timestamp"] <= t]
        if len(row) == 0:
            return None
        return row.iloc[-1]["paragraph"]

    fixations["paragraph"] = fixations["Fixation_StartTime"].apply(get_paragraph)
    fixations["font_size"]  = fixations["paragraph"].map(FONT_MAP)
    fixations = fixations.dropna(subset=["paragraph", "font_size"])

    # statistiche per paragrafo
    stats = fixations.groupby("font_size").agg(
        n_fissazioni        = ("Fixation_StartTime", "count"),
        durata_totale_sec   = ("Fixation_Duration",  "sum"),
        durata_media_sec    = ("Fixation_Duration",  "mean"),
    ).reset_index().sort_values("font_size")

    print(stats.to_string())

    # --- ISTOGRAMMI ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # istogramma durata totale fissazioni
    axes[0].bar(stats["font_size"], stats["durata_totale_sec"], color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Font size")
    axes[0].set_ylabel("Durata totale fissazioni (s)")
    axes[0].set_title("Somma durata fissazioni per font")
    axes[0].invert_xaxis()  # font grande a sinistra

    # istogramma numero fissazioni
    axes[1].bar(stats["font_size"], stats["n_fissazioni"], color="darkorange", edgecolor="none")
    axes[1].set_xlabel("Font size")
    axes[1].set_ylabel("Numero fissazioni")
    axes[1].set_title("Numero fissazioni per font")
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT, "fissazioni_per_font.png"), dpi=300, bbox_inches='tight')
    plt.show()

    return stats

def regressions_analysis_by_font(fixations, threshold=0.03):
    fix = fixations.copy()
    fix = fix.dropna(subset=["font_size", "Fixation_CentroidX"])
    fix = fix.sort_values(["font_size", "Fixation_StartTime"])

    results = []

    for font_size, data in fix.groupby("font_size"):
        data = data.sort_values("Fixation_StartTime").reset_index(drop=True)

        dx = data["Fixation_CentroidX"].diff()

        n_fix = len(data)
        n_reg = int((dx < -threshold).sum())
        durata_tot = data["Fixation_Duration"].sum()

        results.append({
            "font_size": font_size,
            "n_fissazioni": n_fix,
            "durata_totale_fissazioni": durata_tot,
            "n_regressioni": n_reg,
            "regressioni_per_fissazione": n_reg / n_fix if n_fix > 0 else 0
        })

    return pd.DataFrame(results).sort_values("font_size", ascending=False)

def eyetracklib_fixations(gaze):
    saccades, fixations = calculate_saccades_and_fixations(
        df = gaze,
        gaze_x_column = "x_smooth",
        gaze_y_column = "y_smooth",
        time_column = "timestamp",
        window_start_time = 0.0,
        window_end_time = gaze["timestamp"].max(),
        algorithm = "I-DT",             # <— recommended (I-VT is beta)
        dispersion_threshold = 0.05,    # px
        min_fixation_duration = 0.15,   # ms
        max_time_gap = 0.075             # ms
        )
    #stats_reg = regressions_analysis_by_font(fixations)
    #print(stats_reg)
    #file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_40.txt")
    #fixations.to_csv(file_out, index=False, header=False, sep="\t")
    #file_out = os.path.join(CARTELLA_OUT_SAC, "saccadi_e_40.txt")
    #saccades.to_csv(file_out, index=False, header=False, sep="\t")
    return (fixations, saccades)

def peyemmv_fixations(gaze, fixations):
    gaze_prova = gaze.copy()
    gaze_mmv = gaze_prova[["x_smooth", "y_smooth", "timestamp"]].dropna().astype(float).copy()
    file_out = os.path.join(CARTELLA_OUT, "gaze_prova_mmv.txt")
    
    # forza sempre la ricreazione del file, così è sempre aggiornato
    gaze_mmv.to_csv(file_out, index=False, header=False, sep="\t")
    print("File scritto:", file_out)

    parametri_fine = [
        (0.02, 0.01, 0.06),
        (0.02, 0.01, 0.065),
        (0.02, 0.01, 0.07),
        (0.02, 0.012, 0.06),
        (0.02, 0.012, 0.065),
        (0.02, 0.012, 0.07),
        (0.02, 0.013, 0.06),
        (0.02, 0.014, 0.065),
        (0.02, 0.015, 0.06),
        (0.02, 0.015, 0.065),
        (0.02, 0.015, 0.07),
        (0.021, 0.01, 0.065),
        (0.022, 0.011, 0.065),
        (0.023, 0.012, 0.065),
    ]

    best_params = None
    best_diff = float('inf')
    best_fix = None

    for p in parametri_fine:
        try:
            result = mmv.extract_fixations(file_out, p[0], p[1], p[2], '0')
            diff = abs(len(result) - len(fixations))
            if diff < best_diff:
                best_diff = diff
                best_params = p
                best_fix = result
        except Exception as e:
            print(f"Parametri {p} saltati: {e}")
            continue

    print(f"Migliori parametri: t1={best_params[0]}, t2={best_params[1]}, min_dur={best_params[2]}")
    print(f"Fissazioni trovate con PeyeMMV: {len(best_fix)}")

    columns = ["Fixation_X", "Fixation_Y", "Fixation_Duration", "Fixation_StartTime", "Fixation_EndTime", "Fixation_No_gaze_points"]
    fix = pd.DataFrame(best_fix, columns=columns)
    fixations_analysis(gaze, fix)
    fix = fix.rename(columns={"Fixation_X": "Fixation_CentroidX", "Fixation_Y": "Fixation_CentroidY"})
    stats_reg = regressions_analysis_by_font(fix)
    print(stats_reg)
    return fix

def pygazeanalyzer_fixations(gaze_initial):
    gaze = gaze_initial.copy()
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)
    
    x = gaze["x_smooth"].to_numpy()
    y = gaze["y_smooth"].to_numpy()
    t = gaze["timestamp"].to_numpy()
    
    # calcola i gap tra campioni consecutivi
    dt = np.diff(t)
    gap_threshold = 0.1  # 100ms, più grande dell'intervallo normale ~20ms
    
    # inserisci valori -1 (missing) nei punti subito dopo un gap
    x_masked = x.copy().astype(float)
    y_masked = y.copy().astype(float)
    
    gap_indices = np.where(dt > gap_threshold)[0] + 1
    print(f"Gap trovati: {len(gap_indices)}")
    x_masked[gap_indices] = -1.0
    y_masked[gap_indices] = -1.0
    
    Sfix, Efix = fixation_detection(
        x_masked, y_masked, t,
        missing=-1.0,
        maxdist=0.05,
        mindur=0.15
    )
    
    print(f"Fissazioni PyGaze: {len(Efix)}")
    columns = ["Fixation_StartTime", "Fixation_EndTime", "Fixation_Duration", "Fixation_X", "Fixation_Y"]
    fix = pd.DataFrame(Efix, columns=columns)
    fixations_analysis(gaze_initial, fix)
    fix = fix.rename(columns={"Fixation_X": "Fixation_CentroidX", "Fixation_Y": "Fixation_CentroidY"})
    stats_reg = regressions_analysis_by_font(fix)
    print(stats_reg)
    return fix

def plot_fixations(gaze, fix_etl, fix_mmv, fix_pgz):
    
    def get_stats(gaze, fixations, label):
        fixations = fixations.copy()
        def get_paragraph(t):
            row = gaze[gaze["timestamp"] <= t]
            if len(row) == 0:
                return None
            return row.iloc[-1]["paragraph"]
        fixations["paragraph"] = fixations["Fixation_StartTime"].apply(get_paragraph)
        fixations["font_size"]  = fixations["paragraph"].map(FONT_MAP)
        fixations = fixations.dropna(subset=["paragraph", "font_size"])
        stats = fixations.groupby("font_size").agg(
            n_fissazioni      = ("Fixation_StartTime", "count"),
            durata_totale_sec = ("Fixation_Duration",  "sum"),
        ).reset_index()
        stats["libreria"] = label
        return stats

    stats_etl = get_stats(gaze, fix_etl, "EyeTrackLib")
    stats_mmv = get_stats(gaze, fix_mmv, "PeyeMMV")
    stats_pgz = get_stats(gaze, fix_pgz, "PyGazeAnalyser")
    all_stats = pd.concat([stats_etl, stats_mmv, stats_pgz], ignore_index=True)

    font_sizes = sorted(all_stats["font_size"].unique(), reverse=True)
    librerie   = ["EyeTrackLib", "PeyeMMV", "PyGazeAnalyser"]
    x          = np.arange(len(font_sizes))
    width      = 0.25
    colors     = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, lib in enumerate(librerie):
        data   = all_stats[all_stats["libreria"] == lib].set_index("font_size")
        offset = (i - 1) * width  # -1, 0, +1 per centrare le 3 barre

        y_dur = [data.loc[f, "durata_totale_sec"] if f in data.index else 0 for f in font_sizes]
        y_nfi = [data.loc[f, "n_fissazioni"]      if f in data.index else 0 for f in font_sizes]

        axes[0].bar(x + offset, y_dur, width, label=lib, color=colors[i], edgecolor="none")
        axes[1].bar(x + offset, y_nfi, width, label=lib, color=colors[i], edgecolor="none")

    for ax, title, ylabel in zip(
        axes,
        ["Durata totale fissazioni per font", "Numero fissazioni per font"],
        ["Durata totale (s)", "Numero fissazioni"]
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(font_sizes)
        ax.set_xlabel("Font size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT, "confronto_librerie.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_fixation_30_35_40(gaze):
    columns = ['Fixation_StartTime', 'Fixation_EndTime', 'Fixation_Duration',
       'Fixation_CentroidX', 'Fixation_CentroidY', 'Fixation_StdX',
       'Fixation_StdY', 'Fixation_Dispersion', 'Fixation_RangeX',
       'Fixation_RangeY', 'Fixation_MaxX', 'Fixation_MinX', 'Fixation_MaxY',
       'Fixation_MinY', 'Fixation_FirstX', 'Fixation_FirstY', 'Fixation_LastX',
       'Fixation_LastY', 'paragraph', 'font_size']
    file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_30.txt")
    fixations_30 = pd.read_csv(file_out, sep="\t", header=None, names=columns)
    file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_35.txt")
    fixations_35 = pd.read_csv(file_out, sep="\t", header=None, names=columns)
    file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_40.txt")
    fixations_40 = pd.read_csv(file_out, sep="\t", header=None, names=columns)
    plot_fixations_30_35_40(gaze, fixations_30, fixations_35, fixations_40 )

    def get_stats(gaze, fixations, label):
            fixations = fixations.copy()
            fixations = fixations.dropna(subset=["paragraph", "font_size"])
            stats = fixations.groupby("font_size").agg(
                n_fissazioni      = ("Fixation_StartTime", "count"),
                durata_totale_sec = ("Fixation_Duration",  "sum"),
            ).reset_index()
            stats["libreria"] = label
            return stats

    stats_30 = get_stats(gaze, fix_30, "30 cm")
    stats_35 = get_stats(gaze, fix_35, "35 cm")
    stats_40 = get_stats(gaze, fix_40, "40 cm")
    all_stats = pd.concat([stats_30, stats_35, stats_40], ignore_index=True)

    font_sizes = sorted(all_stats["font_size"].unique(), reverse=True)
    librerie   = ["30 cm", "35 cm", "40 cm"]
    x          = np.arange(len(font_sizes))
    width      = 0.25
    colors     = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, lib in enumerate(librerie):
        data   = all_stats[all_stats["libreria"] == lib].set_index("font_size")
        offset = (i - 1) * width
        y_dur = [data.loc[f, "durata_totale_sec"] if f in data.index else 0 for f in font_sizes]
        y_nfi = [data.loc[f, "n_fissazioni"]      if f in data.index else 0 for f in font_sizes]
        axes[0].bar(x + offset, y_dur, width, label=lib, color=colors[i], edgecolor="none")
        axes[1].bar(x + offset, y_nfi, width, label=lib, color=colors[i], edgecolor="none")

    for ax, title, ylabel in zip(
        axes,
        ["Durata totale fissazioni per font", "Numero fissazioni per font"],
        ["Durata totale (s)", "Numero fissazioni"]
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(font_sizes)
        ax.set_xlabel("Font size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT, "confronto_distanze.png"), dpi=300, bbox_inches='tight')
    plt.show()

def calcola_regressioni_da_fissazioni(fixations, soglia_x=0.02, soglia_y=0.05):
    fixations = fixations.copy()
    fixations = fixations.dropna(subset=["Fixation_StartTime", "Fixation_EndTime", "Fixation_Duration", "Fixation_CentroidX", "Fixation_CentroidY", "paragraph", "font_size"])
    fixations = fixations.sort_values(["paragraph", "Fixation_StartTime"]).reset_index(drop=True)

    fixations["prev_x"] = fixations.groupby("paragraph")["Fixation_CentroidX"].shift(1)
    fixations["prev_y"] = fixations.groupby("paragraph")["Fixation_CentroidY"].shift(1)
    fixations["prev_start"] = fixations.groupby("paragraph")["Fixation_StartTime"].shift(1)
    fixations["prev_end"] = fixations.groupby("paragraph")["Fixation_EndTime"].shift(1)
    fixations["delta_x"] = fixations["Fixation_CentroidX"] - fixations["prev_x"]
    fixations["delta_y"] = fixations["Fixation_CentroidY"] - fixations["prev_y"]

    regressions = fixations[(fixations["delta_x"] < -soglia_x) &(fixations["delta_y"].abs() < soglia_y)].copy()
    regressions["Regression_Duration"] = (regressions["Fixation_EndTime"] - regressions["Fixation_StartTime"])
    regressions["Regression_Amplitude_X"] = regressions["delta_x"].abs()

    return regressions

def plot_regression_30_35_40(gaze):
    columns = [
        'Fixation_StartTime', 'Fixation_EndTime', 'Fixation_Duration',
        'Fixation_CentroidX', 'Fixation_CentroidY', 'Fixation_StdX',
        'Fixation_StdY', 'Fixation_Dispersion', 'Fixation_RangeX',
        'Fixation_RangeY', 'Fixation_MaxX', 'Fixation_MinX',
        'Fixation_MaxY', 'Fixation_MinY', 'Fixation_FirstX',
        'Fixation_FirstY', 'Fixation_LastX', 'Fixation_LastY',
        'paragraph', 'font_size'
    ]

    file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_30.txt")
    fixations_30 = pd.read_csv(file_out, sep="\t", header=None, names=columns)

    file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_35.txt")
    fixations_35 = pd.read_csv(file_out, sep="\t", header=None, names=columns)

    file_out = os.path.join(CARTELLA_OUT_FIX, "fixation_e_40.txt")
    fixations_40 = pd.read_csv(file_out, sep="\t", header=None, names=columns)

    def get_stats(fixations, label):
        regressions = calcola_regressioni_da_fissazioni(fixations)
        stats = regressions.groupby("font_size").agg(n_regressioni=("Fixation_StartTime", "count"), durata_totale_sec=("Regression_Duration", "sum")).reset_index()
        stats["distanza"] = label
        return stats

    stats_30 = get_stats(fixations_30, "30 cm")
    stats_35 = get_stats(fixations_35, "35 cm")
    stats_40 = get_stats(fixations_40, "40 cm")

    all_stats = pd.concat([stats_30, stats_35, stats_40], ignore_index=True   )

    #print("STATISTICHE REGRESSIONI:")
    #print(all_stats)

    if all_stats.empty:
        print("Nessuna regressione trovata.")
        return

    font_sizes = sorted(all_stats["font_size"].unique(), reverse=True)
    distanze = ["30 cm", "35 cm", "40 cm"]

    x = np.arange(len(font_sizes))
    width = 0.25
    colors = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, distanza in enumerate(distanze):
        data = all_stats[all_stats["distanza"] == distanza].set_index("font_size")
        offset = (i - 1) * width
        y_durata = [data.loc[f, "durata_totale_sec"] if f in data.index else 0 for f in font_sizes]
        y_numero = [data.loc[f, "n_regressioni"] if f in data.index else 0 for f in font_sizes]
        axes[0].bar(x + offset, y_durata, width, label=distanza, color=colors[i], edgecolor="none")
        axes[1].bar(x + offset, y_numero, width, label=distanza, color=colors[i], edgecolor="none")

    axes[0].set_title("Durata totale regressioni per font")
    axes[0].set_ylabel("Durata totale regressioni (s)")
    axes[1].set_title("Numero regressioni per font")
    axes[1].set_ylabel("Numero regressioni")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(font_sizes)
        ax.set_xlabel("Font size")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT_REG, "confronto_distanze_regressioni.png"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_regressions(gaze, fix_etl, fix_mmv, fix_pgz):
   
    def get_stats(gaze, fixations, label):
        fixations = fixations.copy()

        def get_paragraph(t):
            row = gaze[gaze["timestamp"] <= t]
            if len(row) == 0:
                return None
            return row.iloc[-1]["paragraph"]

        fixations["paragraph"] = fixations["Fixation_StartTime"].apply(get_paragraph)
        fixations["font_size"] = fixations["paragraph"].map(FONT_MAP)

        fixations = fixations.dropna(subset=["paragraph", "font_size"])

        regressions = calcola_regressioni_da_fissazioni(fixations)
        stats = regressions.groupby("font_size").agg(n_regressioni=("Fixation_StartTime", "count"), durata_totale_sec=("Regression_Duration", "sum"),).reset_index()
        stats["libreria"] = label
        return stats

    stats_etl = get_stats(gaze, fix_etl, "EyeTrackLib")
    stats_mmv = get_stats(gaze, fix_mmv, "PeyeMMV")
    stats_pgz = get_stats(gaze, fix_pgz, "PyGazeAnalyser")

    all_stats = pd.concat([stats_etl, stats_mmv, stats_pgz], ignore_index=True)

    if all_stats.empty:
        print("Nessuna regressione trovata.")
        return

    font_sizes = sorted(all_stats["font_size"].unique(), reverse=True)
    librerie = ["EyeTrackLib", "PeyeMMV", "PyGazeAnalyser"]

    x = np.arange(len(font_sizes))
    width = 0.25
    colors = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, lib in enumerate(librerie):
        data = all_stats[all_stats["libreria"] == lib].set_index("font_size")
        offset = (i - 1) * width
        y_dur = [data.loc[f, "durata_totale_sec"] if f in data.index else 0  for f in font_sizes]
        y_nreg = [data.loc[f, "n_regressioni"] if f in data.index else 0 for f in font_sizes]
        axes[0].bar(x + offset, y_dur, width, label=lib, color=colors[i], edgecolor="none")
        axes[1].bar(x + offset, y_nreg, width, label=lib, color=colors[i], edgecolor="none")

    for ax, title, ylabel in zip(axes,
        ["Durata totale regressioni per font", "Numero regressioni per font"],
        ["Durata totale regressioni (s)", "Numero regressioni"]):
        ax.set_xticks(x)
        ax.set_xticklabels(font_sizes)
        ax.set_xlabel("Font size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(CARTELLA_OUT, "confronto_librerie_regressioni.png"), dpi=300, bbox_inches="tight")

    plt.show()

def saccades_analysis(gaze, saccades):
    # aggiungi paragrafo ad ogni saccade in base al timestamp
    def get_paragraph(t):
        row = gaze[gaze["timestamp"] <= t]
        if len(row) == 0:
            return None
        return row.iloc[-1]["paragraph"]

    saccades["paragraph"] = saccades["Saccade_StartTime"].apply(get_paragraph)
    saccades["font_size"]  = saccades["paragraph"].map(FONT_MAP)
    saccades["Saccade_Duration"] = saccades["Saccade_EndTime"] - saccades["Saccade_StartTime"]
    saccades = saccades.dropna(subset=["paragraph", "font_size"])

    # statistiche per paragrafo
    stats = saccades.groupby("font_size").agg(
        n_saccadi        = ("Saccade_StartTime", "count"),
        durata_totale_sec   = ("Saccade_Duration",  "sum"),
        durata_media_sec    = ("Saccade_Duration",  "mean"),
    ).reset_index().sort_values("font_size")

    print(stats.to_string())

    # --- ISTOGRAMMI ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # istogramma durata totale fissazioni
    axes[0].bar(stats["font_size"], stats["durata_totale_sec"], color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Font size")
    axes[0].set_ylabel("Durata totale saccadi (s)")
    axes[0].set_title("Somma durata saccadi per font")
    axes[0].invert_xaxis()  # font grande a sinistra

    # istogramma numero fissazioni
    axes[1].bar(stats["font_size"], stats["n_saccadi"], color="darkorange", edgecolor="none")
    axes[1].set_xlabel("Font size")
    axes[1].set_ylabel("Numero saccadi")
    axes[1].set_title("Numero saccadi per font")
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT, "saccadi_per_font.png"), dpi=300, bbox_inches='tight')
    plt.show()

    return stats

def saccades_pymovements(gaze):
    gaze = gaze[["timestamp", "x_smooth", "y_smooth"]].dropna().copy()
    gaze = gaze.sort_values("timestamp").reset_index(drop=True)

    x = gaze["x_smooth"].to_numpy(dtype=float)
    y = gaze["y_smooth"].to_numpy(dtype=float)

    # timestamp originale in secondi
    t_sec = gaze["timestamp"].to_numpy(dtype=float)

    # timestamp in millisecondi per pymovements
    t_ms = t_sec * 1000.0
    t_ms_int = t_ms.astype(int)

    dt = np.diff(t_ms)
    dx = np.diff(x)
    dy = np.diff(y)

    valid = dt > 0

    vx = np.full(len(gaze), np.nan, dtype=float)
    vy = np.full(len(gaze), np.nan, dtype=float)

    idx = np.where(valid)[0] + 1

    vx[idx] = dx[valid] / dt[valid]
    vy[idx] = dy[valid] / dt[valid]

    velocities = np.column_stack([vx, vy])
    velocities = np.nan_to_num(velocities, nan=0.0)

    events = pm.events.microsaccades(
        velocities=velocities,
        timesteps=t_ms_int,
        minimum_duration=40,
        threshold_factor=12.0,
        name="saccade"
    )

    try:
        saccades = events.frame.to_pandas()
    except AttributeError:
        try:
            saccades = events.data.to_pandas()
        except AttributeError:
            saccades = pd.DataFrame(events)

    saccades.columns = [str(c) for c in saccades.columns]

    if len(saccades) == 0:
        return pd.DataFrame(columns=[
            "Saccade_StartTime",
            "Saccade_EndTime",
            "Saccade_Duration",
            "x_onset",
            "y_onset",
            "x_offset",
            "y_offset",
            "amplitude"
        ])

    def ms_to_index(ms_val):
        idx = np.searchsorted(t_ms_int, int(ms_val))
        idx = np.clip(idx, 0, len(t_sec) - 1)
        return idx

    onset_idx = saccades["onset"].apply(ms_to_index).astype(int)
    offset_idx = saccades["offset"].apply(ms_to_index).astype(int)

    saccades["Saccade_StartTime"] = t_sec[onset_idx]
    saccades["Saccade_EndTime"] = t_sec[offset_idx]
    saccades["Saccade_Duration"] = (
        saccades["Saccade_EndTime"] - saccades["Saccade_StartTime"]
    )

    saccades["x_onset"] = x[onset_idx]
    saccades["y_onset"] = y[onset_idx]
    saccades["x_offset"] = x[offset_idx]
    saccades["y_offset"] = y[offset_idx]

    saccades["amplitude"] = np.sqrt(
        (saccades["x_offset"] - saccades["x_onset"]) ** 2 +
        (saccades["y_offset"] - saccades["y_onset"]) ** 2
    )

    print("Pymovements - saccadi:", len(saccades))

    return saccades

def saccades_pygaze(gaze, target_count):
    gaze_clean = gaze[["timestamp", "x_smooth", "y_smooth"]].dropna()
    gaze_clean = gaze_clean.sort_values("timestamp").reset_index(drop=True)

    t_ms = gaze_clean["timestamp"].to_numpy(dtype=float) * 1000
    x    = gaze_clean["x_smooth"].to_numpy(dtype=float)
    y    = gaze_clean["y_smooth"].to_numpy(dtype=float)

    best_diff   = float("inf")
    best_params = None
    best_Esac   = None

    for maxvel in np.arange(0.1, 1.0, 0.05):
        for minlen in np.arange(5, 30, 5):
            try:
                _, Esac = saccade_detection(
                    x, y, t_ms,
                    missing=0.0,
                    minlen=minlen,
                    maxvel=maxvel,
                    maxacc=maxvel * 15
                )
                diff = abs(len(Esac) - target_count)
                if diff < best_diff:
                    best_diff   = diff
                    best_params = (minlen, maxvel)
                    best_Esac   = Esac
            except Exception as e:
                continue

    print(f"Migliori parametri: minlen={best_params[0]}, maxvel={best_params[1]:.2f}")
    print(f"Saccadi trovate: {len(best_Esac)}  (target: {target_count})")

    df = pd.DataFrame(best_Esac, columns=[
        "onset_ms", "offset_ms", "duration_ms",
        "x_onset", "y_onset",
        "x_offset", "y_offset"
    ])
    df["Saccade_StartTime"] = df["onset_ms"] / 1000.0
    df["Saccade_EndTime"]   = df["offset_ms"] / 1000.0
    df["Saccade_Duration"] = df["Saccade_EndTime"] - df["Saccade_StartTime"]

    return df

def plot_saccades(gaze, sac_etl, sac_pymvn, sac_pgz):
    
    def get_stats(gaze, saccades, label):
        saccades = saccades.copy()
        def get_paragraph(t):
            row = gaze[gaze["timestamp"] <= t]
            if len(row) == 0:
                return None
            return row.iloc[-1]["paragraph"]
        saccades["paragraph"] = saccades["Saccade_StartTime"].apply(get_paragraph)
        saccades["font_size"]  = saccades["paragraph"].map(FONT_MAP)
        saccades = saccades.dropna(subset=["paragraph", "font_size"])
        stats = saccades.groupby("font_size").agg(
            n_saccadi      = ("Saccade_StartTime", "count"),
            durata_totale_sec = ("Saccade_Duration",  "sum"),
        ).reset_index()
        stats["libreria"] = label
        return stats

    stats_etl = get_stats(gaze, sac_etl, "EyeTrackLib")
    stats_mmv = get_stats(gaze, sac_pymvn, "Pymovements")
    stats_pgz = get_stats(gaze, sac_pgz, "PyGazeAnalyser")
    all_stats = pd.concat([stats_etl, stats_mmv, stats_pgz], ignore_index=True)

    font_sizes = sorted(all_stats["font_size"].unique(), reverse=True)
    librerie   = ["EyeTrackLib", "Pymovements", "PyGazeAnalyser"]
    x          = np.arange(len(font_sizes))
    width      = 0.25
    colors     = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, lib in enumerate(librerie):
        data   = all_stats[all_stats["libreria"] == lib].set_index("font_size")
        offset = (i - 1) * width  # -1, 0, +1 per centrare le 3 barre

        y_dur = [data.loc[f, "durata_totale_sec"] if f in data.index else 0 for f in font_sizes]
        y_nfi = [data.loc[f, "n_saccadi"]      if f in data.index else 0 for f in font_sizes]

        axes[0].bar(x + offset, y_dur, width, label=lib, color=colors[i], edgecolor="none")
        axes[1].bar(x + offset, y_nfi, width, label=lib, color=colors[i], edgecolor="none")

    for ax, title, ylabel in zip(
        axes,
        ["Durata totale saccadi per font", "Numero saccadi per font"],
        ["Durata totale (s)", "Numero saccadi"]
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(font_sizes)
        ax.set_xlabel("Font size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT, "confronto_librerie_saccadi.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_saccades_30_35_40(gaze):
    columns = ['Saccade_StartTime', 'Saccade_EndTime', 'gaze_points', 'paragraph', 'font_size', 'Saccade_Duration']
    file_out = os.path.join(CARTELLA_OUT_SAC, "saccadi_e_30.txt")
    sac_30 = pd.read_csv(file_out, sep="\t", header=None, names=columns)
    file_out = os.path.join(CARTELLA_OUT_SAC, "saccadi_e_35.txt")
    sac_35 = pd.read_csv(file_out, sep="\t", header=None, names=columns)
    file_out = os.path.join(CARTELLA_OUT_SAC, "saccadi_e_40.txt")
    sac_40 = pd.read_csv(file_out, sep="\t", header=None, names=columns)

    def get_stats(gaze, saccades, label):
            saccades = saccades.copy()
            saccades = saccades.dropna(subset=["paragraph", "font_size"])
            stats = saccades.groupby("font_size").agg(
                n_saccadi         = ("Saccade_StartTime", "count"),
                durata_totale_sec = ("Saccade_Duration",  "sum"),
            ).reset_index()
            stats["libreria"] = label
            return stats

    stats_30 = get_stats(gaze, sac_30, "30 cm")
    stats_35 = get_stats(gaze, sac_35, "35 cm")
    stats_40 = get_stats(gaze, sac_40, "40 cm")
    all_stats = pd.concat([stats_30, stats_35, stats_40], ignore_index=True)

    font_sizes = sorted(all_stats["font_size"].unique(), reverse=True)
    librerie   = ["30 cm", "35 cm", "40 cm"]
    x          = np.arange(len(font_sizes))
    width      = 0.25
    colors     = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, lib in enumerate(librerie):
        data   = all_stats[all_stats["libreria"] == lib].set_index("font_size")
        offset = (i - 1) * width
        y_dur = [data.loc[f, "durata_totale_sec"] if f in data.index else 0 for f in font_sizes]
        y_nfi = [data.loc[f, "n_saccadi"]      if f in data.index else 0 for f in font_sizes]
        axes[0].bar(x + offset, y_dur, width, label=lib, color=colors[i], edgecolor="none")
        axes[1].bar(x + offset, y_nfi, width, label=lib, color=colors[i], edgecolor="none")

    for ax, title, ylabel in zip(
        axes,
        ["Durata totale saccadi per font", "Numero saccadi per font"],
        ["Durata totale (s)", "Numero saccadi"]
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(font_sizes)
        ax.set_xlabel("Font size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT_SAC, "confronto_distanze_saccadi.png"), dpi=300, bbox_inches='tight')
    plt.show()    

def plot_reading_sequence_one_font(gaze, fixations, font_size, font_col="font_size", line_col="line", time_col="timestamp", x_col="x_smooth", y_col="y_smooth", n_lines=11, y_jitter_strength=0.35, title=None, min_fix_duration=None, connect_only_same_line=True):
    gaze = gaze.copy()
    fix = fixations.copy()

    if font_col not in gaze.columns:
        raise ValueError(f"Manca la colonna '{font_col}' in gaze.")

    if line_col not in gaze.columns:
        raise ValueError(f"Manca la colonna '{line_col}' in gaze.")

    # colonne tempo fixation
    if "Fixation_StartTime" in fix.columns:
        fix_start_col = "Fixation_StartTime"
    elif "start_time" in fix.columns:
        fix_start_col = "start_time"
    elif "onset" in fix.columns:
        fix_start_col = "onset"
    else:
        raise ValueError("Non trovo la colonna di inizio fissazione.")

    if "Fixation_EndTime" in fix.columns:
        fix_end_col = "Fixation_EndTime"
    elif "end_time" in fix.columns:
        fix_end_col = "end_time"
    elif "offset" in fix.columns:
        fix_end_col = "offset"
    else:
        raise ValueError("Non trovo la colonna di fine fissazione.")

    rows = []

    for _, f in fix.iterrows():
        start = float(f[fix_start_col])
        end = float(f[fix_end_col])

        if min_fix_duration is not None and (end - start) < min_fix_duration:
            continue

        pts = gaze[(gaze[time_col] >= start) &(gaze[time_col] <= end)        ]

        if len(pts) == 0:
            continue

        current_font = pts[font_col].mode()
        current_line = pts[line_col].mode()

        if len(current_font) == 0 or len(current_line) == 0:
            continue

        current_font = current_font.iloc[0]
        current_line = int(current_line.iloc[0])

        if current_font != font_size:
            continue

        if current_line <= 0:
            continue

        rows.append({ "start": start, "end": end, "x": pts[x_col].mean(), "y_smooth": pts[y_col].mean(), "line_global": current_line, "font_size": current_font})

    fdf = pd.DataFrame(rows)

    if len(fdf) == 0:
        print(f"Nessuna fissazione trovata per font {font_size}")
        return fdf

    fdf = fdf.sort_values("start").reset_index(drop=True)

    # --------------------------------------------------
    # Converte righe globali in righe locali 1-10
    # Esempio: 11-20 diventa 1-10
    # --------------------------------------------------
    min_line = int(fdf["line_global"].min())
    fdf["line_local"] = fdf["line_global"] - min_line + 1

    # --------------------------------------------------
    # Usa y_smooth come piccolo scostamento dentro la riga
    # --------------------------------------------------
    fdf["y_plot"] = fdf["line_local"].astype(float)

    for line, g in fdf.groupby("line_local"):
        y_min = g["y_smooth"].min()
        y_max = g["y_smooth"].max()

        if y_max == y_min:
            offset = np.zeros(len(g))
        else:
            # normalizza y dentro la riga tra -0.5 e +0.5
            offset = (g["y_smooth"] - y_min) / (y_max - y_min)
            offset = offset - 0.5

        fdf.loc[g.index, "y_plot"] = (
            g["line_local"].astype(float) +
            offset * y_jitter_strength
        )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(14, 7))

    # linee tra fixation consecutive
    for i in range(len(fdf) - 1):
        same_line = fdf.loc[i, "line_local"] == fdf.loc[i + 1, "line_local"]

        if connect_only_same_line and not same_line:
            continue

        plt.plot([fdf.loc[i, "x"], fdf.loc[i + 1, "x"]], [fdf.loc[i, "y_plot"],  fdf.loc[i + 1, "y_plot"]], color="red", linewidth=1.5,alpha=0.7)

    # fixation
    plt.scatter(fdf["x"], fdf["y_plot"], s=70, color="orange", edgecolor="black",  zorder=5, label="Fixation"    )

    plt.yticks(range(1, n_lines + 1))
    plt.gca().invert_yaxis()

    plt.xlabel("X gaze normalizzata")
    plt.ylabel("Riga del testo")

    if title is None:
        title = f"Pattern di lettura - font {font_size}"

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fdf

def add_visual_proxy(stats, distance_cm):
    """
    Aggiunge una variabile proporzionale alla dimensione angolare apparente:
    visual_proxy = font_size / distanza.

    È una proxy di tan(theta), a meno di un fattore moltiplicativo.
    """
    stats = stats.copy()
    stats["distance_cm"] = distance_cm
    stats["visual_proxy"] = stats["font_size"] / distance_cm
    return stats

def plot_visual_proxy_and_eye_metrics():
    # -----------------------------
    # COLONNE FISSAZIONI
    # -----------------------------
    fixation_columns = ['Fixation_StartTime', 'Fixation_EndTime', 'Fixation_Duration', 'Fixation_CentroidX', 'Fixation_CentroidY', 'Fixation_StdX', 'Fixation_StdY', 'Fixation_Dispersion', 'Fixation_RangeX', 'Fixation_RangeY', 'Fixation_MaxX', 'Fixation_MinX', 'Fixation_MaxY', 'Fixation_MinY', 'Fixation_FirstX', 'Fixation_FirstY', 'Fixation_LastX', 'Fixation_LastY', 'paragraph', 'font_size']
    fix_30 = pd.read_csv(os.path.join(CARTELLA_OUT_FIX, "fixation_e_30.txt"), sep="\t", header=None, names=fixation_columns)
    fix_35 = pd.read_csv(os.path.join(CARTELLA_OUT_FIX, "fixation_e_35.txt"), sep="\t", header=None, names=fixation_columns)
    fix_40 = pd.read_csv(os.path.join(CARTELLA_OUT_FIX, "fixation_e_40.txt"), sep="\t", header=None, names=fixation_columns)
 
    # -----------------------------
    # COLONNE SACCADI
    # -----------------------------
    saccade_columns = ['Saccade_StartTime', 'Saccade_EndTime', 'gaze_points', 'paragraph', 'font_size', 'Saccade_Duration']
    sac_30 = pd.read_csv(os.path.join(CARTELLA_OUT_SAC, "saccadi_e_30.txt"), sep="\t", header=None, names=saccade_columns)
    sac_35 = pd.read_csv(os.path.join(CARTELLA_OUT_SAC, "saccadi_e_35.txt"), sep="\t", header=None, names=saccade_columns)
    sac_40 = pd.read_csv(os.path.join(CARTELLA_OUT_SAC, "saccadi_e_40.txt"), sep="\t", header=None, names=saccade_columns)
 
    # -----------------------------
    # STATISTICHE FISSAZIONI
    # -----------------------------
    def get_fixation_stats(fixations, distance_cm):
        fixations = fixations.copy()
        fixations = fixations.dropna(subset=["font_size"])
        stats = fixations.groupby("font_size").agg(n_fissazioni=("Fixation_StartTime", "count"), durata_fissazioni=("Fixation_Duration", "sum"), durata_media_fissazioni=("Fixation_Duration", "mean")).reset_index()
        stats["distance_cm"] = distance_cm
        stats["visual_proxy"] = stats["font_size"] / distance_cm
        return stats
 
    stats_fix_30 = get_fixation_stats(fix_30, 30)
    stats_fix_35 = get_fixation_stats(fix_35, 35)
    stats_fix_40 = get_fixation_stats(fix_40, 40)
    all_fix = pd.concat([stats_fix_30, stats_fix_35, stats_fix_40], ignore_index=True)
 
    # -----------------------------
    # STATISTICHE SACCADI
    # -----------------------------
    def get_saccade_stats(saccades, distance_cm):
        saccades = saccades.copy()
        saccades = saccades.dropna(subset=["font_size"])
        stats = saccades.groupby("font_size").agg(n_saccadi=("Saccade_StartTime", "count"), durata_saccadi=("Saccade_Duration", "sum"), durata_media_saccadi=("Saccade_Duration", "mean")).reset_index()
        stats["distance_cm"] = distance_cm
        stats["visual_proxy"] = stats["font_size"] / distance_cm
        return stats
 
    stats_sac_30 = get_saccade_stats(sac_30, 30)
    stats_sac_35 = get_saccade_stats(sac_35, 35)
    stats_sac_40 = get_saccade_stats(sac_40, 40)
    all_sac = pd.concat([stats_sac_30, stats_sac_35, stats_sac_40], ignore_index=True)
 
    # -----------------------------
    # LABEL VISUAL PROXY
    # -----------------------------
    all_fix["visual_proxy_label"] = all_fix["visual_proxy"].round(3).astype(str)
    all_sac["visual_proxy_label"] = all_sac["visual_proxy"].round(3).astype(str)
    proxy_order_fix = all_fix[["visual_proxy", "visual_proxy_label"]].drop_duplicates().sort_values("visual_proxy", ascending=False)["visual_proxy_label"].tolist()
    proxy_order_sac = all_sac[["visual_proxy", "visual_proxy_label"]].drop_duplicates().sort_values("visual_proxy", ascending=False)["visual_proxy_label"].tolist()
 
    # -----------------------------
    # FONT SIZE vs VISUAL PROXY
    # -----------------------------
    df_proxy = []
    for distance_cm in [30, 35, 40]:
        for font_size in sorted(all_fix["font_size"].unique(), reverse=True):
            df_proxy.append({"font_size": font_size, "distance_cm": distance_cm, "visual_proxy": font_size / distance_cm})
    df_proxy = pd.DataFrame(df_proxy)
    font_order = sorted(df_proxy["font_size"].unique(), reverse=True)
    hue_order = [30, 35, 40]
 
    # -----------------------------
    # TROVA COPPIE CON PROXY SIMILE
    # -----------------------------
    def find_proxy_matches(df_proxy, tol=0.005):
        matches = []
        already_seen = set()
        df = df_proxy.copy()
        df["combo"] = df["font_size"].astype(int).astype(str) + "." + df["distance_cm"].astype(int).astype(str)
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if j <= i:
                    continue
                if row1["distance_cm"] == row2["distance_cm"]:
                    continue
                diff = abs(row1["visual_proxy"] - row2["visual_proxy"])
                if diff <= tol:
                    combo1 = row1["combo"]
                    combo2 = row2["combo"]
                    key = tuple(sorted([combo1, combo2]))
                    if key in already_seen:
                        continue
                    already_seen.add(key)
                    matches.append({"combo1": combo1, "combo2": combo2, "font_size_1": row1["font_size"], "distance_cm_1": row1["distance_cm"], "visual_proxy_1": row1["visual_proxy"], "font_size_2": row2["font_size"], "distance_cm_2": row2["distance_cm"], "visual_proxy_2": row2["visual_proxy"], "visual_proxy_mean": (row1["visual_proxy"] + row2["visual_proxy"]) / 2})
        return pd.DataFrame(matches)
 
    proxy_matches = find_proxy_matches(df_proxy, tol=0.005)
 
    print("\nCoppie con visual proxy simile:")
    if not proxy_matches.empty:
        print(proxy_matches[["combo1", "combo2", "visual_proxy_1", "visual_proxy_2", "visual_proxy_mean"]])
    else:
        print("Nessuna coppia trovata con la tolleranza scelta.")
 
    PAIR_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#a65628", "#f781bf", "#999999", "#377eb8"]
    combo_to_pair = {}
    if not proxy_matches.empty:
        for idx, match in proxy_matches.iterrows():
            color = PAIR_COLORS[idx % len(PAIR_COLORS)]
            combo_to_pair[match["combo1"]] = (color, idx)
            combo_to_pair[match["combo2"]] = (color, idx)
 
    # -----------------------------
    # GRAFICO FONT SIZE vs VISUAL PROXY
    # -----------------------------
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df_proxy, x="font_size", y="visual_proxy", hue="distance_cm", order=font_order, hue_order=hue_order)
 
    for container, distance_cm in zip(ax.containers, hue_order):
        for patch, font_size in zip(container.patches, font_order):
            combo = f"{int(font_size)}.{int(distance_cm)}"
            if combo in combo_to_pair:
                color, _ = combo_to_pair[combo]
                patch.set_edgecolor(color)
                patch.set_linewidth(2.5)
            else:
                patch.set_edgecolor("none")
 
    def draw_proxy_match_lines(ax, df_proxy, proxy_matches, font_order, hue_order):
        bars = []
        for container, distance_cm in zip(ax.containers, hue_order):
            for patch, font_size in zip(container.patches, font_order):
                height = patch.get_height()
                if np.isnan(height):
                    continue
                bars.append({"font_size": font_size, "distance_cm": distance_cm, "visual_proxy": height, "x_center": patch.get_x() + patch.get_width() / 2})
        bars_df = pd.DataFrame(bars)
        y_offset_step = 0.006
        level_counter = {}
        for _, match in proxy_matches.iterrows():
            row1 = bars_df[(bars_df["font_size"] == match["font_size_1"]) & (bars_df["distance_cm"] == match["distance_cm_1"])]
            row2 = bars_df[(bars_df["font_size"] == match["font_size_2"]) & (bars_df["distance_cm"] == match["distance_cm_2"])]
            if row1.empty or row2.empty:
                continue
            row1 = row1.iloc[0]
            row2 = row2.iloc[0]
            y_base = match["visual_proxy_mean"]
            y_key = round(y_base, 3)
            level_counter[y_key] = level_counter.get(y_key, 0) + 1
            y = y_base + y_offset_step * (level_counter[y_key] - 1)
            x_start = min(row1["x_center"], row2["x_center"])
            x_end = max(row1["x_center"], row2["x_center"])
            label = f'{int(match["font_size_1"])}.{int(match["distance_cm_1"])} ≈ {int(match["font_size_2"])}.{int(match["distance_cm_2"])}'
            ax.hlines(y=y, xmin=x_start, xmax=x_end, colors="black", linestyles="--", linewidth=1)
            ax.text((x_start + x_end) / 2, y + 0.003, label, ha="center", va="bottom", fontsize=7, color="black")
 
    draw_proxy_match_lines(ax, df_proxy, proxy_matches, font_order, hue_order)
    plt.xlabel("Font size")
    plt.ylabel("Visual proxy = font size / distanza")
    plt.title("Visual proxy per font size e distanza")
    plt.legend(title="Distanza (cm)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(CARTELLA_OUT_PROX, "font_size_visual_proxy_con_linee.png"), dpi=300, bbox_inches="tight")
    plt.show()
 
    # -----------------------------
    # GRAFICI SU PROXY CON BORDI COLORATI PER COPPIA
    # -----------------------------
    def plot_metric_by_proxy(data, y_col, ylabel, title, file_path, order, proxy_matches, combo_to_pair):
        plt.figure(figsize=(14, 6))
        ax = plt.gca()
        hue_order_local = [30, 35, 40]
        width = 0.25
        x_positions = np.arange(len(order))
        offset_map = {30: -width, 35: 0, 40: width}
        color_map = {30: "#ead6d1", 35: "#a66f91", 40: "#2f243f"}
        legend_done = set()
        bar_positions = []
        for distance_cm in hue_order_local:
            subset = data[data["distance_cm"] == distance_cm].copy()
            for _, row in subset.iterrows():
                proxy_label = row["visual_proxy_label"]
                if proxy_label not in order:
                    continue
                x_base = order.index(proxy_label)
                x = x_base + offset_map[distance_cm]
                y = row[y_col]
                font_size = int(row["font_size"])
                distance = int(row["distance_cm"])
                combo = f"{font_size}.{distance}"
                label_legend = f"{distance_cm} cm" if distance_cm not in legend_done else None
                legend_done.add(distance_cm)
                edge_color, lw = (combo_to_pair[combo][0], 2.5) if combo in combo_to_pair else ("none", 0)
                ax.bar(x, y, width=width, label=label_legend, color=color_map[distance_cm], edgecolor=edge_color, linewidth=lw)
                ax.text(x, y, combo, ha="center", va="bottom", fontsize=8, rotation=90)
                bar_positions.append({"x": x, "y": y, "font_size": font_size, "distance_cm": distance, "combo": combo})
        bar_positions = pd.DataFrame(bar_positions)
        if not proxy_matches.empty and not bar_positions.empty:
            ymax = data[y_col].max()
            y_step = ymax * 0.055
            y_start = ymax * 1.05
            for k, match in proxy_matches.iterrows():
                combo1 = f'{int(match["font_size_1"])}.{int(match["distance_cm_1"])}'
                combo2 = f'{int(match["font_size_2"])}.{int(match["distance_cm_2"])}'
                row1 = bar_positions[bar_positions["combo"] == combo1]
                row2 = bar_positions[bar_positions["combo"] == combo2]
                if row1.empty or row2.empty:
                    continue
                row1 = row1.iloc[0]
                row2 = row2.iloc[0]
                pair_color = combo_to_pair.get(combo1, ("#000000", 0))[0]
                x_start = min(row1["x"], row2["x"])
                x_end = max(row1["x"], row2["x"])
                y_line = y_start + k * y_step
                ax.hlines(y=y_line, xmin=x_start, xmax=x_end, colors=pair_color, linestyles="--", linewidth=1.5)
                ax.text((x_start + x_end) / 2, y_line, f"{combo1} ≈ {combo2}", ha="center", va="bottom", fontsize=7, color=pair_color)
            ax.set_ylim(0, y_start + len(proxy_matches) * y_step * 1.2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(order, rotation=45)
        ax.set_xlabel("Visual proxy = font size / distanza")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Distanza (cm)")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.show()
 
    plot_metric_by_proxy(data=all_fix, y_col="durata_fissazioni", ylabel="Durata totale fissazioni (s)", title="Durata delle fissazioni in funzione della visual proxy", file_path=os.path.join(CARTELLA_OUT_PROX, "proxy_durata_fissazioni.png"), order=proxy_order_fix, proxy_matches=proxy_matches, combo_to_pair=combo_to_pair)
    plot_metric_by_proxy(data=all_fix, y_col="n_fissazioni", ylabel="Numero fissazioni", title="Numero di fissazioni in funzione della visual proxy", file_path=os.path.join(CARTELLA_OUT_PROX, "proxy_numero_fissazioni.png"), order=proxy_order_fix, proxy_matches=proxy_matches, combo_to_pair=combo_to_pair)
    plot_metric_by_proxy(data=all_sac, y_col="durata_saccadi", ylabel="Durata totale saccadi (s)", title="Durata delle saccadi in funzione della visual proxy", file_path=os.path.join(CARTELLA_OUT_PROX, "proxy_durata_saccadi.png"), order=proxy_order_sac, proxy_matches=proxy_matches, combo_to_pair=combo_to_pair)
    plot_metric_by_proxy(data=all_sac, y_col="n_saccadi", ylabel="Numero saccadi", title="Numero di saccadi in funzione della visual proxy", file_path=os.path.join(CARTELLA_OUT_PROX, "proxy_numero_saccadi.png"), order=proxy_order_sac, proxy_matches=proxy_matches, combo_to_pair=combo_to_pair)
 
    return all_fix, all_sac, df_proxy, proxy_matches
 
df = read_json()
gaze = gaze_info(df) #gestisco outlier in modo blando
#analyse_noise(gaze)
gaze = filter_by_speed(gaze, percentile=99)
gaze = noise_removal(gaze)
#plot_distribution(gaze, "Distribuzione dei dati dopo refining")

#JUMPS
gaze_new, jumps= find_jumps(gaze,N_JUMPS_EXPECTED)

jumps_valid = calibrate_and_validate_jumps(jumps, gaze_new, N_JUMPS_EXPECTED)
stringa_jump = "salti trovati"
#plot_detected_jumps(gaze_new, jumps_valid, stringa_jump)


#PARAGRAPHS
gaze_paragraphs = delete_inconsistencies_between_lines(gaze_new, jumps_valid)
gaze_paragraphs = find_lines(gaze_paragraphs, jumps_valid)
gaze_paragraphs = assign_paragraph_from_lines(gaze_paragraphs, LINES_PER_PARAGRAPH)
_, gap_report = compute_gap_quality_report(gaze_paragraphs, paragraph_col="paragraph", t_col="timestamp")
print(gap_report)

first_row_timestamp = jumps[0]["timestamp_inizio"]
gaze_paragraphs = detracting_eye_stabilization_first_paragraph(gaze_paragraphs, first_row_timestamp)
#plot_paragraphs_time(gaze_paragraphs)



#ANALISYS
#analysis_line(gaze_paragraphs)
#info_rows = compute_line_stats(gaze_paragraphs)


fixations_eyetracklib, saccades_eyetracklib = eyetracklib_fixations(gaze_paragraphs)
fixations_analysis(gaze_paragraphs, fixations_eyetracklib)
saccades_analysis(gaze_paragraphs,saccades_eyetracklib)
#fixations_peyemmv = peyemmv_fixations(gaze_paragraphs, fixations_eyetracklib)
#fixations_pygazeanalyzer = pygazeanalyzer_fixations(gaze_paragraphs)

#plot_fixations(gaze_paragraphs, fixations_eyetracklib, fixations_peyemmv, fixations_pygazeanalyzer)
#plot_fixation_30_35_40(gaze_paragraphs)
#plot_regression_30_35_40(gaze_paragraphs)
#plot_regressions(gaze_paragraphs, fixations_eyetracklib, fixations_peyemmv, fixations_pygazeanalyzer)


#saccades_pymovements = saccades_pymovements(gaze_paragraphs)
#saccades_pygaze = saccades_pygaze(gaze_paragraphs, len(saccades_eyetracklib))
#plot_saccades(gaze_paragraphs, saccades_eyetracklib, saccades_pymovements, saccades_pygaze)
#plot_saccades_30_35_40(gaze)
#gaze_paragraphs["font_size"]  = gaze_paragraphs["paragraph"].map(FONT_MAP)
#plot_reading_sequence_one_font(gaze_paragraphs, fixations_eyetracklib, 14)

#plot_visual_proxy_and_eye_metrics()