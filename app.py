"""Streamlit app: Feature Selection & Feature Extraction – Dataset Book-Crossing.
Dikonversi dari notebook Tugas DMV (Ahmad Qadavi, 5003241126)."""
import re, time, itertools, warnings
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, SelectFromModel, RFE, RFECV,
                                       SequentialFeatureSelector, f_classif, f_regression, chi2,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV, ElasticNetCV, lasso_path
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.datasets import make_circles, load_wine, load_digits

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
BIRU, ORANYE, TOSCA, MERAH = "#0a2a66", "#f5a623", "#1aa5c8", "#d62728"

st.set_page_config(page_title="Book-Crossing: Feature Selection & Extraction", page_icon="📚", layout="wide")

# ------------------------------------------------------------------ DATA
@st.cache_data(show_spinner="Membaca & membersihkan data ...")
def muat_data(sumber):
    """sumber: dict nama -> bytes/path. Mengembalikan df_buku, df_user, df_rating (sudah dibersihkan)."""
    kw_buku = dict(sep=";", quotechar='"', escapechar="\\", encoding="latin-1", low_memory=False, on_bad_lines="skip")
    df_buku = pd.read_csv(sumber["books"], **kw_buku)
    df_user = pd.read_csv(sumber["users"], sep=";", encoding="latin-1", na_values=["NULL"])
    df_rating = pd.read_csv(sumber["ratings"], sep=";", encoding="latin-1")
    df_buku.columns = ["isbn", "title", "author", "year", "publisher", "img_s", "img_m", "img_l"]
    df_user.columns = ["user_id", "location", "age"]
    df_rating.columns = ["user_id", "isbn", "rating"]
    df_user.loc[~df_user["age"].between(5, 100), "age"] = np.nan
    df_user["country"] = (df_user["location"].fillna("").str.split(",").str[-1].str.strip().replace("", "unknown"))
    df_buku["year"] = pd.to_numeric(df_buku["year"], errors="coerce")
    df_buku.loc[~df_buku["year"].between(1900, 2005), "year"] = np.nan
    for col in ["author", "publisher", "title"]:
        df_buku[col] = df_buku[col].fillna("unknown").astype(str).str.lower().str.strip()
    df_buku["isbn"] = df_buku["isbn"].astype(str).str.upper().str.strip()
    df_rating["isbn"] = df_rating["isbn"].astype(str).str.upper().str.strip()
    df_buku = df_buku.drop(columns=["img_s", "img_m", "img_l"])
    return df_buku, df_user, df_rating


def clean(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")[:18]


@st.cache_data(show_spinner="Rekayasa fitur (satu baris = satu pengguna) ...")
def rekayasa_fitur(df_buku, df_user, df_rating, min_rating, seed):
    gen = np.random.default_rng(seed)
    df_inter = df_rating.merge(df_buku, on="isbn", how="inner")
    eksplisit = df_inter[df_inter.rating > 0]
    stat_user = eksplisit.groupby("user_id").rating.agg(["mean", "count"])
    user_target = stat_user.index[stat_user["count"] >= min_rating]
    d = df_inter[df_inter.user_id.isin(user_target)].copy()
    d["log_book_pop"] = np.log1p(d.isbn.map(df_inter.groupby("isbn").size()))
    d["log_author_pop"] = np.log1p(d.author.map(df_inter.groupby("author").size()))
    d["title_len"] = d.title.str.len()

    g = d.groupby("user_id")
    f = pd.DataFrame(index=sorted(user_target))
    f["n_interactions"] = g.size()
    f["log_n_interactions"] = np.log1p(f.n_interactions)
    f["pct_implicit"] = (d.rating == 0).groupby(d.user_id).mean()
    f["year_mean"] = g.year.mean()
    f["year_std"] = g.year.std()
    f["log_book_pop_mean"] = g.log_book_pop.mean()
    f["log_author_pop_mean"] = g.log_author_pop.mean()
    f["title_len_mean"] = g.title_len.mean()
    f["author_diversity"] = g.author.nunique() / f.n_interactions
    f["publisher_diversity"] = g.publisher.nunique() / f.n_interactions

    def share(col, top_n, prefix):
        top = d[col].value_counts().head(top_n).index
        ct = (d[d[col].isin(top)].groupby(["user_id", col]).size().unstack(fill_value=0)
              .reindex(index=f.index, columns=top, fill_value=0))
        ct = ct.div(f.n_interactions, axis=0)
        ct.columns = [f"{prefix}_{i:02d}_{clean(c)}" for i, c in enumerate(top)]
        return ct

    dek = pd.cut(d.year, [1900, 1960, 1980, 1990, 1995, 2000, 2006], right=False,
                 labels=["1900_59", "1960_79", "1980_89", "1990_94", "1995_99", "2000_05"])
    ct_dek = (d.assign(dec=dek).groupby(["user_id", "dec"]).size().unstack(fill_value=0)
              .reindex(f.index, fill_value=0))
    ct_dek = ct_dek.div(f.n_interactions, axis=0)
    ct_dek.columns = [f"yr_share_{c}" for c in ct_dek.columns]

    du = df_user.set_index("user_id").loc[f.index]
    usia_hilang = du.age.isna().astype(float).rename("age_missing")
    usia = du.age.fillna(df_user.age.median()).rename("age")
    negara_top = du.country.value_counts().head(15).index
    dum = pd.get_dummies(du.country.where(du.country.isin(negara_top))).astype(float).add_prefix("ctry_")
    dum.columns = [clean(c) for c in dum.columns]

    X = pd.concat([f, ct_dek, share("publisher", 60, "pub"), share("author", 60, "auth"),
                   usia, usia_hilang, dum], axis=1)
    X = X.fillna(X.median())
    # fitur buatan untuk pembelajaran
    X["const_feature"] = 1.0
    X["near_const_flag"] = (gen.random(len(X)) < 0.0005).astype(float)
    X["age_months"] = X.age * 12 + gen.normal(0, 2, len(X))
    for i in range(1, 21):
        X[f"noise_{i:02d}"] = gen.normal(size=len(X))
    kol_noise = ["const_feature", "near_const_flag"] + [f"noise_{i:02d}" for i in range(1, 21)]

    y_rating = stat_user.loc[X.index, "mean"]
    y_bin = (y_rating > y_rating.median()).astype(int)
    y_tiga = pd.Series(pd.qcut(y_rating.rank(method="first"), 3, labels=False), index=X.index)
    info = dict(n_inter=len(df_inter), n_rating=len(df_rating), n_user=len(user_target),
                n_pakai=len(d), median_y=float(y_rating.median()))
    return X, y_rating, y_bin, y_tiga, kol_noise, info


def data_demo(n=6000, seed=0):
    """Data sintetis format Book-Crossing agar app bisa dicoba tanpa CSV asli."""
    r = np.random.default_rng(seed)
    n_buku, n_user = 1500, n
    pubs = [f"Publisher {i}" for i in range(40)]
    auth = [f"Author {i}" for i in range(120)]
    buku = pd.DataFrame({
        "isbn": [f"{i:010d}" for i in range(n_buku)], "title": [f"Judul buku {i}" * r.integers(1, 3) for i in range(n_buku)],
        "author": r.choice(auth, n_buku), "year": r.integers(1950, 2005, n_buku),
        "publisher": r.choice(pubs, n_buku), "img_s": "", "img_m": "", "img_l": ""})
    ctry = ["usa", "canada", "united kingdom", "germany", "australia", "spain", "italy", "france"]
    user = pd.DataFrame({"user_id": np.arange(1, n_user + 1),
                         "location": [f"kota, bagian, {c}" for c in r.choice(ctry, n_user)],
                         "age": np.where(r.random(n_user) < .4, np.nan, r.integers(10, 80, n_user))})
    bobot = r.dirichlet(np.ones(n_buku) * .3)
    baris = []
    for u in user.user_id:
        k = int(r.integers(5, 60))
        isb = r.choice(n_buku, k, p=bobot)
        bias = r.normal(7.5, 1)
        rt = np.where(r.random(k) < .5, 0, np.clip(np.round(r.normal(bias, 1.5, k)), 1, 10))
        baris += [(u, buku.isbn[i], int(x)) for i, x in zip(isb, rt)]
    rating = pd.DataFrame(baris, columns=["User-ID", "ISBN", "Book-Rating"])
    return buku, user, rating


# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title("📚 Book-Crossing")
st.sidebar.caption("Feature Selection & Extraction")
mode = st.sidebar.radio("Sumber data", ["Upload CSV", "Folder lokal", "Data demo (sintetis)"])
sumber = None
if mode == "Upload CSV":
    fb = st.sidebar.file_uploader("BX-Books.csv", type="csv")
    fu = st.sidebar.file_uploader("BX-Users.csv", type="csv")
    fr = st.sidebar.file_uploader("BX-Book-Ratings.csv", type="csv")
    if fb and fu and fr:
        sumber = {"books": fb, "users": fu, "ratings": fr}
elif mode == "Folder lokal":
    folder = Path(st.sidebar.text_input("Path folder berisi 3 CSV", "data"))
    if (folder / "BX-Books.csv").exists():
        sumber = {"books": folder / "BX-Books.csv", "users": folder / "BX-Users.csv",
                  "ratings": folder / "BX-Book-Ratings.csv"}
    else:
        st.sidebar.warning("BX-Books.csv tidak ditemukan di folder tersebut.")
min_rating = st.sidebar.slider("Min. rating eksplisit per pengguna", 1, 20, 5)
SEED = st.sidebar.number_input("Random seed", 0, 9999, 42)
TOP_N = st.sidebar.slider("Top-N fitur", 5, 30, 15)

if mode == "Data demo (sintetis)":
    @st.cache_data
    def _demo():
        b, u, r = data_demo()
        return b, u.rename(columns={}), r
    df_buku, df_user, df_rating = _demo()
    # samakan dengan hasil pembersihan
    df_buku.columns = ["isbn", "title", "author", "year", "publisher", "img_s", "img_m", "img_l"]
    df_user.columns = ["user_id", "location", "age"]
    df_rating.columns = ["user_id", "isbn", "rating"]
    df_user["country"] = df_user.location.str.split(",").str[-1].str.strip()
    for c in ["author", "publisher", "title"]:
        df_buku[c] = df_buku[c].str.lower()
    df_buku = df_buku.drop(columns=["img_s", "img_m", "img_l"])
    st.sidebar.info("Memakai data sintetis – hasil hanya untuk mencoba aplikasi.")
elif sumber is not None:
    df_buku, df_user, df_rating = muat_data(sumber)
else:
    st.title("Feature Selection & Feature Extraction – Book-Crossing")
    st.info("Unggah **BX-Books.csv**, **BX-Users.csv**, dan **BX-Book-Ratings.csv** di sidebar "
            "(atau pilih folder lokal / data demo) untuk memulai.")
    st.stop()

X_user, y_rating, y_biner, y_tiga, KOL_NOISE, info = rekayasa_fitur(df_buku, df_user, df_rating, min_rating, int(SEED))
KOL = list(X_user.columns)

# split & scaling (di-cache per konfigurasi)
@st.cache_data(show_spinner="Menyiapkan split & standardisasi ...")
def siapkan(_X, _yb, _yr, _y3, seed):
    idx_tr, idx_te = train_test_split(_X.index, test_size=0.2, stratify=_yb, random_state=seed)
    Xtr, Xte = _X.loc[idx_tr], _X.loc[idx_te]
    sc = StandardScaler().fit(Xtr); mm = MinMaxScaler().fit(Xtr)
    Xtr_std = pd.DataFrame(sc.transform(Xtr), columns=_X.columns, index=Xtr.index)
    Xtr_mm = pd.DataFrame(mm.transform(Xtr), columns=_X.columns, index=Xtr.index)
    n_sub = min(3000, len(Xtr)); n_cmp = min(5000, len(Xtr))
    idx_sub = Xtr.sample(n_sub, random_state=seed).index
    idx_cmp = Xtr.sample(n_cmp, random_state=seed).index
    return dict(Xtr=Xtr, Xte=Xte, ybtr=_yb.loc[idx_tr], ybte=_yb.loc[idx_te], yrtr=_yr.loc[idx_tr],
                yrte=_yr.loc[idx_te], y3tr=_y3.loc[idx_tr], y3te=_y3.loc[idx_te], Xtr_std=Xtr_std,
                Xtr_mm=Xtr_mm, Xte_std=sc.transform(Xte), idx_sub=idx_sub, idx_cmp=idx_cmp)

S = siapkan(X_user, y_biner, y_rating, y_tiga, int(SEED))
X_tr, X_te, yb_tr, yb_te, yr_tr, yr_te = S["Xtr"], S["Xte"], S["ybtr"], S["ybte"], S["yrtr"], S["yrte"]
Xtr_std, Xtr_mm, idx_sub, idx_cmp = S["Xtr_std"], S["Xtr_mm"], S["idx_sub"], S["idx_cmp"]
X_kecil, yb_kecil = X_tr.loc[idx_sub], yb_tr.loc[idx_sub]
X_banding, yb_banding = X_tr.loc[idx_cmp], yb_tr.loc[idx_cmp]
cv5 = StratifiedKFold(5, shuffle=True, random_state=int(SEED))
model_logit = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
gen_acak = np.random.default_rng(int(SEED))

if "hasil_pilih" not in st.session_state:
    st.session_state.hasil_pilih = {}
hasil_pilih = st.session_state.hasil_pilih


def ambil_top(s, k=None):
    return list(s.sort_values(ascending=False).head(k or TOP_N).index)


def plot_peringkat(s, judul, k=None):
    k = k or TOP_N
    s = s.sort_values(ascending=False).head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(s.index, s.values, color=[MERAH if i in KOL_NOISE else BIRU for i in s.index])
    ax.set_title(judul + "  (merah = noise/konstan buatan)", fontsize=10)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def logreg_l1(C):
    try:
        return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=500)
    except TypeError:
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=500)


# ------------------------------------------------------------------ HEADER
st.title("📚 Feature Selection & Feature Extraction")
st.caption("Dataset Book-Crossing · 1 baris = 1 pengguna · target = rata-rata rating eksplisit")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pengguna dipakai", f"{info['n_user']:,}")
c2.metric("Jumlah fitur (p)", X_user.shape[1])
c3.metric("Interaksi dipakai", f"{info['n_pakai']:,}")
c4.metric("Median rata-rata rating", f"{info['median_y']:.2f}")

tabs = st.tabs(["1 Data", "2 Curse of Dim.", "3 Filter", "4 Wrapper", "5 Embedded",
                "6 Perbandingan", "7 Extraction", "8 Konsensus"])

# ------------------------------------------------------------------ TAB 1: DATA
with tabs[0]:
    st.subheader("Data setelah pembersihan")
    a, b, c = st.columns(3)
    a.write("**Books**", ); a.dataframe(df_buku.head(8))
    b.write("**Users**"); b.dataframe(df_user.head(8))
    c.write("**Ratings**"); c.dataframe(df_rating.head(8))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    df_rating.rating.value_counts().sort_index().plot.bar(ax=ax[0], color=BIRU); ax[0].set_title("Distribusi rating (0 = implisit)")
    df_user.age.plot.hist(bins=50, ax=ax[1], color=ORANYE); ax[1].set_title("Distribusi usia")
    df_buku.year.plot.hist(bins=50, ax=ax[2], color=TOSCA); ax[2].set_title("Tahun terbit buku")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write(f"Rating implisit (0): **{(df_rating.rating == 0).mean():.1%}** dari semua interaksi")
    st.subheader("Tabel fitur per pengguna")
    st.write(f"X: {X_user.shape[0]:,} pengguna × {X_user.shape[1]} fitur. "
             f"Proporsi kelas y_bin: {y_biner.value_counts(normalize=True).round(3).to_dict()}")
    st.dataframe(X_user.head(20))
    st.download_button("Unduh tabel fitur (CSV)", X_user.to_csv().encode(), "fitur_user.csv", "text/csv")

# ------------------------------------------------------------------ TAB 2: CURSE
with tabs[1]:
    st.subheader("The Curse of Dimensionality")
    if st.button("Jalankan demo kontras jarak"):
        def kontras(p, n=200, reps=40):
            v = []
            for _ in range(reps):
                P, q = gen_acak.random((n, p)), gen_acak.random(p)
                d = np.linalg.norm(P - q, axis=1); v.append((d.max() - d.min()) / d.min())
            return np.median(v)
        dims = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        hk = [kontras(p) for p in dims]
        fig, ax = plt.subplots(figsize=(6.5, 3.6)); ax.plot(range(len(dims)), hk, "o-", color=BIRU)
        ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims); ax.set_yscale("log")
        ax.set_xlabel("dimensi p"); ax.set_ylabel("relative contrast (log)"); ax.set_title("Jarak kehilangan kontras")
        st.pyplot(fig); plt.close(fig)
    st.markdown("---")
    st.subheader("Fitur noise merusak k-NN")
    if st.button("Jalankan demo k-NN + noise"):
        skor_f, _ = f_classif(Xtr_std.fillna(0), yb_tr)
        skor_f = pd.Series(np.nan_to_num(skor_f), index=KOL)
        asli30 = [c for c in skor_f.sort_values(ascending=False).index if c not in KOL_NOISE][:30]
        Xr = X_tr.loc[idx_sub, asli30].values
        knn = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))])
        jml = [0, 10, 30, 100, 300, 1000]; acc = []
        with st.spinner("Menghitung ..."):
            for m in jml:
                Xn = np.hstack([Xr, gen_acak.normal(size=(len(Xr), m))]) if m else Xr
                acc.append(cross_val_score(knn, Xn, yb_kecil.values, cv=cv5).mean())
        fig, ax = plt.subplots(figsize=(6.5, 3.6)); ax.plot(range(len(jml)), acc, "o-", color=ORANYE)
        for i, a in enumerate(acc): ax.annotate(f"{a:.3f}", (i, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(range(len(jml))); ax.set_xticklabels(jml)
        ax.set_xlabel("fitur noise ditambahkan"); ax.set_ylabel("akurasi CV 5-fold"); ax.set_title("Noise menurunkan akurasi k-NN")
        st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ TAB 3: FILTER
with tabs[2]:
    st.subheader("Metode Filter")
    st.markdown("**Variance Threshold** (data MinMax)")
    thr = st.number_input("Threshold variansi", 0.0, 1.0, 1e-3, format="%.4f")
    sel_var = VarianceThreshold(threshold=thr).fit(Xtr_mm)
    buang = [c for c, k in zip(KOL, sel_var.get_support()) if not k]
    st.write(f"Fitur dibuang: **{len(buang)}** dari {len(KOL)}")
    st.write(buang)

    st.markdown("**Korelasi dengan target**")
    kor_y = X_tr.corrwith(yr_tr).abs().fillna(0)
    plot_peringkat(kor_y, "|Korelasi| dengan rata-rata rating")
    hasil_pilih["Korelasi |r|"] = ambil_top(kor_y)
    C = Xtr_std.drop(columns=["const_feature"]).corr().abs()
    pas = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    st.write("Pasangan fitur redundan (|r| > 0.9):"); st.dataframe(pas[pas > 0.9].round(4).to_frame("|r|"))

    st.markdown("**ANOVA F, Chi-square, Mutual Information**")
    Xf = Xtr_std.fillna(0)
    Fk, pk = f_classif(Xf, yb_tr); Fr, pr = f_regression(Xf, yr_tr)
    tabel_anova = pd.DataFrame({"F_classif": np.nan_to_num(Fk), "p_classif": np.nan_to_num(pk, nan=1),
                                "F_regression": np.nan_to_num(Fr), "p_regression": np.nan_to_num(pr, nan=1)}, index=KOL)
    st.dataframe(tabel_anova.sort_values("F_classif", ascending=False).head(10).round(4))
    st.write(f"Signifikan (p<0.05): classif={int((tabel_anova.p_classif < .05).sum())}, "
             f"regression={int((tabel_anova.p_regression < .05).sum())}")
    hasil_pilih["ANOVA F (classif)"] = ambil_top(tabel_anova.F_classif)
    chi_ser = pd.Series(np.nan_to_num(chi2(Xtr_mm, yb_tr)[0]), index=KOL)
    hasil_pilih["Chi-square"] = ambil_top(chi_ser)
    plot_peringkat(chi_ser, "Skor χ² (MinMax)")

    if st.button("Hitung Mutual Information (agak lama)"):
        t0 = time.time()
        mi_k = pd.Series(mutual_info_classif(Xtr_std, yb_tr, random_state=int(SEED)), index=KOL)
        mi_r = pd.Series(mutual_info_regression(Xtr_std, yr_tr, random_state=int(SEED)), index=KOL)
        st.session_state.mi = (mi_k, mi_r, time.time() - t0)
    if "mi" in st.session_state:
        mi_k, mi_r, dt = st.session_state.mi
        st.caption(f"Waktu hitung MI: {dt:.1f} dtk")
        plot_peringkat(mi_k, "Mutual information (classif)")
        hasil_pilih["Mutual Info (classif)"] = ambil_top(mi_k)
        hasil_pilih["Mutual Info (regression)"] = ambil_top(mi_r)
        tabel_skor = pd.DataFrame({"|corr|": kor_y, "F_clf": tabel_anova.F_classif, "F_reg": tabel_anova.F_regression,
                                   "chi2": chi_ser, "MI_clf": mi_k, "MI_reg": mi_r})
        kr = tabel_skor.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(5.2, 4.4)); im = ax.imshow(kr, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(kr))); ax.set_xticklabels(kr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(kr))); ax.set_yticklabels(kr.columns)
        for i in range(len(kr)):
            for j in range(len(kr)): ax.text(j, i, f"{kr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.grid(False); fig.colorbar(im); ax.set_title("Korelasi peringkat antar skor filter")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Keterbatasan filter: demo XOR**")
    if st.button("Jalankan demo XOR"):
        xx = gen_acak.normal(size=(3000, 2)); xy = (xx[:, 0] * xx[:, 1] > 0).astype(int)
        st.write("F per fitur:", np.round(f_classif(xx, xy)[0], 3).tolist(),
                 "| MI per fitur:", np.round(mutual_info_classif(xx, xy, random_state=0), 4).tolist())
        pohon = DecisionTreeClassifier(max_depth=4, random_state=0)
        st.table(pd.Series({n: cross_val_score(pohon, xx[:, c], xy, cv=5).mean()
                            for n, c in {"hanya x1": [0], "hanya x2": [1], "x1 + x2": [0, 1]}.items()},
                           name="akurasi pohon").round(3))

# ------------------------------------------------------------------ TAB 4: WRAPPER
with tabs[3]:
    st.subheader("Metode Wrapper")
    Fk = pd.Series(np.nan_to_num(f_classif(Xtr_std.fillna(0), yb_tr)[0]), index=KOL)
    kolam8, kolam30 = ambil_top(Fk, 8), ambil_top(Fk, 30)

    st.markdown("**Exhaustive Feature Selection** – semua subset dari 8 kandidat (255 subset)")
    if st.button("Jalankan EFS"):
        with st.spinner("Mengevaluasi subset ..."):
            t0 = time.time(); hs = []
            for r in range(1, 9):
                for comb in itertools.combinations(kolam8, r):
                    hs.append((comb, r, cross_val_score(model_logit, X_kecil[list(comb)], yb_kecil, cv=cv5).mean()))
            t = pd.DataFrame(hs, columns=["subset", "size", "cv_acc"]).sort_values("cv_acc", ascending=False)
        hasil_pilih["EFS (8 kandidat)"] = list(t.iloc[0].subset)
        st.write(f"{len(t)} subset dalam {time.time()-t0:.1f} dtk. Terbaik: {list(t.iloc[0].subset)} → CV acc {t.iloc[0].cv_acc:.4f}")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(t["size"] + gen_acak.normal(0, .05, len(t)), t.cv_acc, s=12, alpha=.5, color=BIRU)
        bs = t.groupby("size").cv_acc.max(); ax.plot(bs.index, bs.values, "o-", color=ORANYE, label="terbaik per ukuran")
        ax.set_xlabel("ukuran subset"); ax.set_ylabel("akurasi CV"); ax.legend(); st.pyplot(fig); plt.close(fig)

    st.markdown("**SFS (forward) dan SBS (backward)** – dari 30 kandidat, pilih 8")
    if st.button("Jalankan SFS & SBS"):
        with st.spinner("Memilih fitur ..."):
            for nama, arah in [("SFS (forward)", "forward"), ("SBS (backward)", "backward")]:
                t0 = time.time()
                sfs = SequentialFeatureSelector(model_logit, n_features_to_select=8, direction=arah,
                                                scoring="accuracy", cv=cv5, n_jobs=-1).fit(X_kecil[kolam30], yb_kecil)
                hasil_pilih[nama] = [str(f) for f in np.array(kolam30)[sfs.get_support()]]
                st.write(f"{nama}: {hasil_pilih[nama]} ({time.time()-t0:.1f} dtk)")
        a, b = set(hasil_pilih["SFS (forward)"]), set(hasil_pilih["SBS (backward)"])
        st.write(f"Jaccard SFS vs SBS: **{len(a & b)/len(a | b):.2f}**")

    st.markdown("**Stepwise (p-value, OLS)**")
    if st.button("Jalankan Stepwise"):
        def stepwise(Xd, y, p_in=0.05, p_out=0.10, max_iter=100):
            pakai = []
            for _ in range(max_iter):
                ch = False; sisa = [c for c in Xd.columns if c not in pakai]
                pb = pd.Series({c: sm.OLS(y, sm.add_constant(Xd[pakai + [c]])).fit().pvalues[c] for c in sisa})
                if len(pb) and pb.min() < p_in: pakai.append(pb.idxmin()); ch = True
                if pakai:
                    pv = sm.OLS(y, sm.add_constant(Xd[pakai])).fit().pvalues.drop("const")
                    if pv.max() > p_out: pakai.remove(pv.idxmax()); ch = True
                if not ch: break
            return pakai
        with st.spinner("Stepwise ..."):
            fs = stepwise(Xtr_std[kolam30], yr_tr)
        hasil_pilih["Stepwise (p-value)"] = fs
        if fs:
            st.write(f"{len(fs)} fitur: {fs} | R² = {sm.OLS(yr_tr, sm.add_constant(Xtr_std[fs])).fit().rsquared:.4f}")
        else:
            st.write("Tidak ada fitur yang lolos.")

    st.markdown("**RFE & RFECV**")
    if st.button("Jalankan RFE & RFECV"):
        with st.spinner("RFE ..."):
            t0 = time.time()
            rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=TOP_N, step=10).fit(Xtr_std.fillna(0), yb_tr)
            hasil_pilih[f"RFE (k={TOP_N})"] = list(np.array(KOL)[rfe.support_])
            st.write(f"RFE ({time.time()-t0:.1f} dtk): {hasil_pilih[f'RFE (k={TOP_N})']}")
            rcv = RFECV(LogisticRegression(max_iter=2000), step=5, cv=3, scoring="accuracy", n_jobs=-1)
            rcv.fit(StandardScaler().fit_transform(X_kecil), yb_kecil)
            sk = rcv.cv_results_["mean_test_score"]; nf = rcv.cv_results_.get("n_features", np.arange(1, len(sk) + 1))
        fig, ax = plt.subplots(figsize=(7, 3.4)); ax.plot(nf, sk, "o-", color=BIRU); ax.axvline(rcv.n_features_, color=MERAH, ls="--")
        ax.set_xlabel("jumlah fitur"); ax.set_ylabel("akurasi CV"); ax.set_title(f"RFECV: optimal = {rcv.n_features_}")
        st.pyplot(fig); plt.close(fig)

    st.markdown("**Overfitting wrapper** (CV saat seleksi vs data uji)")
    if all(k in hasil_pilih for k in ["EFS (8 kandidat)", "SFS (forward)", "SBS (backward)"]) and st.button("Hitung"):
        rows = []
        for n, k in [("EFS", "EFS (8 kandidat)"), ("SFS", "SFS (forward)"), ("SBS", "SBS (backward)")]:
            f_ = hasil_pilih[k]
            cvd = cross_val_score(model_logit, X_kecil[f_], yb_kecil, cv=cv5).mean()
            te = accuracy_score(yb_te, model_logit.fit(X_kecil[f_], yb_kecil).predict(X_te[f_]))
            rows.append((n, len(f_), cvd, te, cvd - te))
        st.dataframe(pd.DataFrame(rows, columns=["metode", "n_fitur", "CV(seleksi)", "akurasi uji", "selisih"]).round(4))
    elif not all(k in hasil_pilih for k in ["EFS (8 kandidat)", "SFS (forward)", "SBS (backward)"]):
        st.caption("Jalankan EFS dan SFS & SBS dahulu untuk membuka perbandingan ini.")

# ------------------------------------------------------------------ TAB 5: EMBEDDED
with tabs[4]:
    st.subheader("Metode Embedded")
    st.markdown("**Lasso, Ridge, Elastic Net** (target regresi)")
    if st.button("Jalankan Lasso / Ridge / Elastic Net"):
        with st.spinner("Fitting ..."):
            al, kp, _ = lasso_path(Xtr_std.values, yr_tr.values, alphas=np.logspace(-3.5, 0, 60))
            lcv = LassoCV(cv=5, random_state=int(SEED)).fit(Xtr_std, yr_tr)
            rcv_ = RidgeCV(alphas=np.logspace(-2, 4, 40)).fit(Xtr_std, yr_tr)
            ecv = ElasticNetCV(l1_ratio=[.2, .5, .8, .95], cv=5, random_state=int(SEED)).fit(Xtr_std, yr_tr)
        fig, ax = plt.subplots(figsize=(7.5, 3.8)); ax.plot(al, kp.T, lw=.8); ax.set_xscale("log")
        ax.axvline(lcv.alpha_, color="k", ls="--"); ax.set_xlabel("alpha"); ax.set_ylabel("koefisien"); ax.set_title("Lasso path")
        st.pyplot(fig); plt.close(fig)
        tk = pd.DataFrame({"Lasso": lcv.coef_, "Ridge": rcv_.coef_, "ElasticNet": ecv.coef_}, index=KOL)
        ring = pd.DataFrame({"alpha": [lcv.alpha_, rcv_.alpha_, ecv.alpha_], "koef. nol": (tk == 0).sum().values,
                             "koef. non-nol": (tk != 0).sum().values,
                             "R² uji": [m.score(S["Xte_std"], yr_te) for m in (lcv, rcv_, ecv)]},
                            index=["Lasso", "Ridge", "ElasticNet"])
        st.dataframe(ring.round(4)); st.write(f"l1_ratio terbaik: {ecv.l1_ratio_}")
        hasil_pilih["Lasso (|coef| top-N)"] = ambil_top(tk.Lasso.abs()[tk.Lasso != 0])
        hasil_pilih["Elastic Net (|coef| top-N)"] = ambil_top(tk.ElasticNet.abs()[tk.ElasticNet != 0])
        st.write("Pasangan berkorelasi (age vs age_months):"); st.dataframe(tk.loc[["age", "age_months"]].round(4))

    st.markdown("**L1-logistic**")
    if st.button("Jalankan L1-logistic"):
        rows = []
        with st.spinner("Mencari C ..."):
            for Cv in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
                pipe = Pipeline([("sc", StandardScaler()), ("clf", logreg_l1(Cv))])
                acc = cross_val_score(pipe, X_tr, yb_tr, cv=cv5, n_jobs=-1).mean()
                nz = int((pipe.fit(X_tr, yb_tr).named_steps["clf"].coef_ != 0).sum())
                rows.append((Cv, nz, acc))
        tl = pd.DataFrame(rows, columns=["C", "fitur non-nol", "CV acc"]); st.dataframe(tl.round(4))
        Cp = float(tl[tl["CV acc"] >= tl["CV acc"].max() - 0.002].C.min())
        m = Pipeline([("sc", StandardScaler()), ("clf", logreg_l1(Cp))]).fit(X_tr, yb_tr)
        w = pd.Series(m.named_steps["clf"].coef_.ravel(), index=KOL)
        st.write(f"C terpilih = {Cp} → {int((w != 0).sum())} fitur non-nol; noise ikut terpilih: {[c for c in KOL_NOISE if w[c] != 0]}")
        hasil_pilih["L1-logistic (top-N)"] = ambil_top(w.abs()[w != 0])
        plot_peringkat(w.abs(), f"|Koefisien| L1-logistic (C={Cp})")
        try:
            sv = Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(penalty="l1", dual=False, C=0.01, max_iter=5000))]).fit(X_tr, yb_tr)
            bs_ = pd.Series(sv.named_steps["clf"].coef_.ravel(), index=KOL)
            hasil_pilih["L1-SVM (top-N)"] = ambil_top(bs_.abs()[bs_ != 0])
            st.write(f"L1-SVM: {int((bs_ != 0).sum())} fitur non-nol, akurasi uji {sv.score(X_te, yb_te):.4f}")
        except Exception as e:
            st.warning(f"L1-SVM dilewati: {e}")

    st.markdown("**Random Forest importance**")
    if st.button("Jalankan Random Forest"):
        with st.spinner("Melatih RF ..."):
            t0 = time.time()
            rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=int(SEED)).fit(X_tr, yb_tr)
        imp = pd.Series(rf.feature_importances_, index=KOL)
        nz = [c for c in KOL_NOISE if c.startswith("noise")]
        st.write(f"RF dilatih {time.time()-t0:.1f} dtk | akurasi uji = {rf.score(X_te, yb_te):.4f} | "
                 f"importance noise maks = {imp[nz].max():.5f}, rata-rata = {imp[nz].mean():.5f} | "
                 f"fitur di atas lantai noise: {int((imp > imp[nz].max()).sum())} dari {len(imp)}")
        plot_peringkat(imp, "Importance Random Forest")
        hasil_pilih["Random Forest importance"] = ambil_top(imp)
        st.write("SelectFromModel(threshold='mean') mempertahankan",
                 int(SelectFromModel(rf, threshold="mean", prefit=True).get_support().sum()), "fitur")
        dt = DecisionTreeClassifier(max_depth=5, random_state=int(SEED)).fit(X_tr, yb_tr)
        st.write("Decision tree (depth 5) – 5 fitur terpenting:")
        st.dataframe(pd.Series(dt.feature_importances_, index=KOL).sort_values(ascending=False).head(5).round(4).to_frame("importance"))

# ------------------------------------------------------------------ TAB 6: PERBANDINGAN
with tabs[5]:
    st.subheader("Membandingkan semua metode (pipeline logistic regression)")
    st.caption("Dievaluasi pada subset (CV 5-fold) lalu diuji pada data uji.")
    pilihan = st.multiselect("Metode", ["Baseline", "Variance Threshold", "ANOVA F", "Mutual Info", "RFE",
                                        "L1-logistic", "RF importance", "PCA", "LDA"],
                             default=["Baseline", "ANOVA F", "L1-logistic", "PCA"])
    if st.button("Bandingkan"):
        fmi = partial(mutual_info_classif, random_state=int(SEED))
        bp = lambda s: Pipeline([("sc", StandardScaler()), ("sel", s), ("clf", LogisticRegression(max_iter=2000))])
        semua = {
            "Baseline": model_logit,
            "Variance Threshold": bp(VarianceThreshold(0.5)),
            "ANOVA F": bp(SelectKBest(f_classif, k=TOP_N)),
            "Mutual Info": bp(SelectKBest(fmi, k=TOP_N)),
            "RFE": bp(RFE(LogisticRegression(max_iter=1000), n_features_to_select=TOP_N, step=10)),
            "L1-logistic": bp(SelectFromModel(logreg_l1(0.05), max_features=TOP_N, threshold=-np.inf)),
            "RF importance": bp(SelectFromModel(RandomForestClassifier(100, min_samples_leaf=5, n_jobs=-1, random_state=int(SEED)),
                                                max_features=TOP_N, threshold=-np.inf)),
            "PCA": Pipeline([("sc", StandardScaler()), ("pca", PCA(TOP_N)), ("clf", LogisticRegression(max_iter=2000))]),
            "LDA": Pipeline([("sc", StandardScaler()), ("lda", LDA(n_components=1)), ("clf", LogisticRegression(max_iter=2000))]),
        }
        rows = []
        prog = st.progress(0.0)
        for i, n in enumerate(pilihan):
            p = semua[n]
            cv_acc = cross_val_score(p, X_banding, yb_banding, cv=cv5).mean()
            t0 = time.time(); p.fit(X_tr, yb_tr); wf = time.time() - t0
            rows.append((n, cv_acc, p.score(X_te, yb_te), wf)); prog.progress((i + 1) / len(pilihan))
        tb = pd.DataFrame(rows, columns=["metode", "CV acc", "akurasi uji", "waktu fit (dtk)"]).set_index("metode")
        st.dataframe(tb.round(4))
        u = tb.index[::-1]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        ax[0].barh(u, tb.loc[u, "akurasi uji"], color=BIRU); ax[0].set_title("Akurasi data uji")
        ax[0].set_xlim(tb["akurasi uji"].min() - .02, tb["akurasi uji"].max() + .02)
        ax[1].barh(u, tb.loc[u, "waktu fit (dtk)"], color=ORANYE); ax[1].set_title("Waktu fit (dtk)"); ax[1].set_xscale("log")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ TAB 7: EXTRACTION
with tabs[6]:
    st.subheader("Feature Extraction")
    st.markdown("**PCA: scree plot**")
    pca_full = PCA().fit(Xtr_std)
    ev, rv = pca_full.explained_variance_, pca_full.explained_variance_ratio_
    kum = np.cumsum(rv)
    k80, k90, k95 = [int(np.searchsorted(kum, q) + 1) for q in (.80, .90, .95)]
    st.write(f"PC1: {rv[0]:.1%} | komponen untuk 80/90/95%: {k80}/{k90}/{k95} | Kaiser (eig>1): {int((ev > 1).sum())} dari p={len(KOL)}")
    K = min(40, len(rv)); xs = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, rv[:K], "o-", color=BIRU, label="per komponen"); ax.plot(xs, kum[:K], "o-", color=ORANYE, label="kumulatif")
    for q in (.9, .95): ax.axhline(q, color="gray", ls=":", lw=.8)
    ax.set_xlabel("komponen k"); ax.set_ylabel("share of variance"); ax.legend(); st.pyplot(fig); plt.close(fig)
    bb = pd.DataFrame(pca_full.components_[:2].T, index=KOL, columns=["PC1", "PC2"])
    st.write("Bobot terbesar PC1 & PC2:")
    st.dataframe(pd.DataFrame({"PC1": bb.PC1.abs().sort_values(ascending=False).head(5).index,
                               "PC2": bb.PC2.abs().sort_values(ascending=False).head(5).index}))

    if st.button("Pilih jumlah komponen PCA via CV"):
        ks = [k for k in [2, 5, 10, 15, 20, 30, 50, 80, 120] if k < len(KOL)]
        with st.spinner("Cross-validation ..."):
            ac = [cross_val_score(Pipeline([("sc", StandardScaler()), ("pca", PCA(k)), ("clf", LogisticRegression(max_iter=2000))]),
                                  X_banding, yb_banding, cv=cv5).mean() for k in ks]
            dasar = cross_val_score(model_logit, X_banding, yb_banding, cv=cv5).mean()
        fig, ax = plt.subplots(figsize=(7, 3.6)); ax.plot(range(len(ks)), ac, "o-", color=BIRU, label="PCA(k)+logistic")
        ax.axhline(dasar, color=MERAH, ls="--", label=f"semua fitur ({dasar:.3f})")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_xlabel("k"); ax.set_ylabel("akurasi CV"); ax.legend()
        st.pyplot(fig); plt.close(fig); st.write("k terbaik:", ks[int(np.argmax(ac))])

    st.markdown("**PCA sebagai kompresi (galat rekonstruksi)**")
    Xc = Xtr_std.values; mu = Xc.mean(0); W = pca_full.components_
    ks = [k for k in [1, 2, 5, 10, 20, 40, 80, 120, 160] if k <= W.shape[0]]
    mse = [np.mean((Xc - (mu + (Xc - mu) @ W[:k].T @ W[:k])) ** 2) for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.plot(ks, mse, "o-", color=BIRU); ax.set_xscale("log")
    ax.set_xlabel("k komponen"); ax.set_ylabel("MSE rekonstruksi"); st.pyplot(fig); plt.close(fig)

    with st.expander("Replikasi: PCA kompresi pada dataset Digits"):
        Xd = load_digits().data; mud = Xd.mean(0); pd_ = PCA().fit(Xd); cd = np.cumsum(pd_.explained_variance_ratio_)
        ksd = [2, 5, 10, 20, 40]
        fig, ax = plt.subplots(4, 6, figsize=(9, 6.2))
        for r in range(4):
            for c, k in enumerate(ksd):
                rec = mud + (Xd[r] - mud) @ pd_.components_[:k].T @ pd_.components_[:k]
                ax[r, c].imshow(rec.reshape(8, 8), cmap="gray_r"); ax[r, c].axis("off")
                if r == 0: ax[r, c].set_title(f"k={k}\n{cd[k-1]:.0%}", fontsize=8)
            ax[r, 5].imshow(Xd[r].reshape(8, 8), cmap="gray_r"); ax[r, 5].axis("off")
            if r == 0: ax[r, 5].set_title("asli", fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**LDA vs PCA (target 3 kelas)**")
    peta = {0: BIRU, 1: ORANYE, 2: TOSCA}
    if st.button("Jalankan LDA vs PCA"):
        y3tr, y3te = S["y3tr"], S["y3te"]
        Zp = PCA(2).fit(Xtr_std).transform(S["Xte_std"])
        Zl = LDA(n_components=2).fit(Xtr_std, y3tr).transform(S["Xte_std"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, Z, t in [(ax[0], Zp, "PCA (unsupervised)"), (ax[1], Zl, "LDA (supervised)")]:
            for k_, col in peta.items(): a.scatter(Z[y3te.values == k_, 0], Z[y3te.values == k_, 1], s=5, alpha=.5, color=col, label=f"kelas {k_}")
            a.set_title(t)
        ax[1].legend(markerscale=3); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        y3c = y_tiga.loc[idx_cmp]
        sc_ = lambda s: cross_val_score(s, X_banding, y3c, cv=cv5).mean()
        st.write(f"Silhouette uji: PCA = {silhouette_score(Zp, y3te):.3f} | LDA = {silhouette_score(Zl, y3te):.3f}")
        st.write(f"Akurasi CV 3-kelas (acak = 0.333): semua fitur = {sc_(model_logit):.3f} | "
                 f"PCA(2) = {sc_(Pipeline([('sc', StandardScaler()), ('pca', PCA(2)), ('clf', LogisticRegression(max_iter=2000))])):.3f} | "
                 f"LDA(2) = {sc_(Pipeline([('sc', StandardScaler()), ('lda', LDA(n_components=2)), ('clf', LogisticRegression(max_iter=2000))])):.3f}")
        st.caption("Pada data pengguna, rating sulit diprediksi dari profil bacaan sehingga pemisahan kelas lemah.")
    with st.expander("Replikasi: PCA vs LDA pada dataset Wine"):
        dw = load_wine(); Xw = StandardScaler().fit_transform(dw.data); yw = dw.target
        Zp, Zl = PCA(2).fit_transform(Xw), LDA(n_components=2).fit_transform(Xw, yw)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for a, Z, n in [(ax[0], Zp, "PCA"), (ax[1], Zl, "LDA")]:
            for k_, col in peta.items(): a.scatter(Z[yw == k_, 0], Z[yw == k_, 1], s=14, color=col, label=f"cultivar {k_+1}")
            a.set_title(f"{n}: silhouette {silhouette_score(Z, yw):.2f}")
        ax[1].legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Kernel PCA (nonlinear)**")
    if st.button("Jalankan Kernel PCA"):
        Xc_, yc_ = make_circles(500, factor=0.3, noise=0.05, random_state=int(SEED))
        hl = PCA(2).fit_transform(Xc_); hk = KernelPCA(2, kernel="rbf", gamma=4).fit_transform(Xc_)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
        for a, Z, t in [(ax[0], Xc_, "Data asli"), (ax[1], hl, "PCA linear"), (ax[2], hk, "Kernel PCA RBF")]:
            a.scatter(Z[:, 0], Z[:, 1], c=np.where(yc_ == 0, BIRU, ORANYE), s=8); a.set_title(t)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        for n, Z in [("PCA linear (1 komp.)", hl[:, :1]), ("Kernel PCA (1 komp.)", hk[:, :1])]:
            st.write(f"{n}: akurasi logistic = {cross_val_score(LogisticRegression(), Z, yc_, cv=5).mean():.3f}")
        p = len(KOL)
        for n, pipe in {"PCA linear (10)": Pipeline([("sc", StandardScaler()), ("pca", PCA(10)), ("clf", LogisticRegression(max_iter=2000))]),
                        "Kernel PCA RBF (10)": Pipeline([("sc", StandardScaler()), ("kp", KernelPCA(10, kernel="rbf", gamma=1.0 / p)),
                                                         ("clf", LogisticRegression(max_iter=2000))])}.items():
            st.write(f"Data pengguna – {n}: CV = {cross_val_score(pipe, X_kecil, yb_kecil, cv=cv5).mean():.4f}")

    st.markdown("**t-SNE (visualisasi saja)**")
    if st.button("Jalankan t-SNE"):
        n = min(1500, len(idx_sub)); ids = idx_sub[:n]
        with st.spinner("t-SNE ..."):
            Zs = PCA(30).fit_transform(StandardScaler().fit(X_tr).transform(X_tr.loc[ids]))
            emb = TSNE(2, perplexity=30, init="pca", random_state=int(SEED)).fit_transform(Zs)
        lab = S["y3tr"].loc[ids].values
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        for k_, col in peta.items(): ax.scatter(emb[lab == k_, 0], emb[lab == k_, 1], s=5, color=col, alpha=.6, label=f"kelas {k_}")
        ax.legend(markerscale=3); ax.set_title(f"t-SNE ({n:,} pengguna)"); st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ TAB 8: KONSENSUS
with tabs[7]:
    st.subheader("Sintesis: fitur yang paling konsisten dipilih")
    if not hasil_pilih:
        st.info("Belum ada metode yang dijalankan.")
    else:
        df_s = pd.DataFrame({m: pd.Series(1, index=f) for m, f in hasil_pilih.items()}).reindex(KOL).fillna(0).astype(int)
        suara = df_s.sum(axis=1).sort_values(ascending=False)
        st.write(f"{len(hasil_pilih)} metode diikutkan: {', '.join(hasil_pilih)}")
        t20 = suara.head(20)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(t20.index, t20.values, color=[MERAH if i in KOL_NOISE else BIRU for i in t20.index])
        ax.set_xlabel("jumlah metode yang memilih"); ax.set_title("Konsensus seleksi fitur")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.write("False positive (noise/konstan terpilih) per metode:")
        st.dataframe(pd.DataFrame({"n_terpilih": df_s.sum(), "noise terpilih": df_s.loc[KOL_NOISE].sum()}))
    st.markdown("---")
    st.markdown("""**Kesimpulan.** Pemilihan metode seleksi fitur perlu mempertimbangkan ukuran dataset, jenis model,
interpretabilitas, dan sumber daya komputasi. Filter paling cepat; wrapper menangkap interaksi tetapi mahal dan rawan overfitting;
embedded memberi keseimbangan; extraction (PCA/LDA/Kernel PCA) menyusun fitur baru dan menurunkan interpretabilitas.""")
