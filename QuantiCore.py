import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm

st.set_page_config(page_title="QuantiCore", layout="wide")

# ---------------- HEADER ----------------
st.title("QuantiCore")
st.subheader("A Smart System for Quantitative Data Analysis")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Controls")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    if "df" not in st.session_state:
        st.session_state.df = df.copy()
    df = st.session_state.df

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    option = st.sidebar.selectbox("Select Analysis", [
        "Dashboard",
        "Preprocessing",
        "Visualization",
        "Sampling",
        "Correlation & SLR",
        "Partial & Multiple Correlation",
        "MLR",
        "MLE",
        "Hypothesis Testing"
    ])

# ---------------- DASHBOARD ----------------
    if option == "Dashboard":
        st.subheader("Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

        st.dataframe(df.head())

# ---------------- PREPROCESSING ----------------
    elif option == "Preprocessing":
        st.subheader("Preprocessing")

        if st.button("Handle Missing"):
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.success("✅ Missing values handled successfully!")

        if st.button("Encode"):
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            st.success("✅ Categorical data encoded!")

        if st.button("Normalize"):
            df[num_cols] = StandardScaler().fit_transform(df[num_cols])
            st.success("✅ Data normalized successfully!")

        st.session_state.df = df
        st.dataframe(df.head())

# ---------------- VISUALIZATION ----------------
    elif option == "Visualization":
        col = st.selectbox("Column", num_cols)
        chart = st.selectbox("Type", ["Histogram", "Boxplot", "Scatter"])

        fig, ax = plt.subplots()

        if chart == "Histogram":
            sns.histplot(df[col], kde=True, stat="density", ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(f"Histogram of {col}")

        elif chart == "Boxplot":
            sns.boxplot(y=df[col], ax=ax)
            ax.set_ylabel(col)
            ax.set_title(f"Boxplot of {col}")

        else:
            col2 = st.selectbox("Y", num_cols)
            sns.regplot(
                x=df[col],
                y=df[col2],
                ax=ax,
                scatter_kws={"s": 60},
                line_kws={"color": "red", "linewidth": 2}
            )
            ax.set_xlabel(col)
            ax.set_ylabel(col2)
            ax.set_title(f"{col} vs {col2} (Trend Line)")

        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close()

# ---------------- SAMPLING ----------------
    elif option == "Sampling":
        method = st.selectbox("Method", ["Random", "Stratified"])
        size = st.slider("Size", 1, len(df), 10)

        if method == "Random":
            sample = df.sample(size)
        else:
            col = st.selectbox("Column", df.columns)
            sample = df.groupby(col).apply(lambda x: x.sample(1))

        st.dataframe(sample)

# ---------------- CORRELATION & SLR ----------------
    elif option == "Correlation & SLR":
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", num_cols)

        corr = df[x].corr(df[y])
        st.metric("Correlation", round(corr, 4))

        st.subheader("Heatmap")

        drop_cols = ["Date", "Person_ID"]
        heatmap_cols = [c for c in num_cols if c not in drop_cols]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[heatmap_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        sns.regplot(x=df[x], y=df[y], ax=ax)
        st.pyplot(fig)
        plt.close()

# ---------------- PARTIAL + MULTIPLE ----------------
    elif option == "Partial & Multiple Correlation":
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", num_cols)
        z = st.selectbox("Z", num_cols)

        if len(set([x, y, z])) == 3:
            rxy = df[x].corr(df[y])
            rxz = df[x].corr(df[z])
            ryz = df[y].corr(df[z])

            pc = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
            st.metric("Partial Correlation", round(pc, 4))

        target = st.selectbox("Target", num_cols)
        features = st.multiselect("Features", num_cols)

        if features:
            model = LinearRegression().fit(df[features], df[target])
            st.metric("Multiple Correlation (R)", round(np.sqrt(model.score(df[features], df[target])), 4))

# ---------------- MLR ----------------
    elif option == "MLR":
        target = st.selectbox("Target", num_cols)
        features = st.multiselect("Features", num_cols)

        if features:
            X = sm.add_constant(df[features])
            y = df[target]
            model = sm.OLS(y, X).fit()

            st.write("Intercept:", round(model.params.iloc[0], 4))
            for i, col in enumerate(features):
                st.write(col, ":", round(model.params.iloc[i+1], 4))

            y_pred = model.predict(X)

            fig, ax = plt.subplots()
            sns.scatterplot(x=y, y=y_pred, ax=ax)

            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("MLR: Actual vs Predicted")

            st.pyplot(fig)
            plt.close()

# ---------------- MLE (IMPROVED ONLY) ----------------
    elif option == "MLE":
        col = st.selectbox("Column", num_cols)
        data = df[col].dropna()

        dist = st.selectbox("Distribution", ["Normal", "Poisson", "Exponential"])

        fig, ax = plt.subplots()

        sns.histplot(data, bins=20, kde=True, stat="density", ax=ax)
        x = np.linspace(data.min(), data.max(), 300)

        if dist == "Normal":
            mu, sigma = stats.norm.fit(data)
            st.metric("Mean (μ)", round(mu, 4))
            st.metric("Std Dev (σ)", round(sigma, 4))

            y = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, y, linewidth=2, label="Normal Fit")

            st.info("Fits continuous data using mean and standard deviation.")

        elif dist == "Poisson":
            lam = np.mean(data)
            st.metric("Lambda (λ)", round(lam, 4))

            k = np.arange(int(data.min()), int(data.max()) + 1)
            pmf = stats.poisson.pmf(k, lam)

            ax.bar(k, pmf, alpha=0.6, label="Poisson PMF")

            st.info("Models count-based events.")

        else:
            lam = 1 / np.mean(data)
            st.metric("Lambda (λ)", round(lam, 4))

            y = lam * np.exp(-lam * x)
            ax.plot(x, y, linewidth=2, label="Exponential Fit")

            st.info("Models time between events.")

        ax.set_xlabel(col)
        ax.set_ylabel("Density / Probability")
        ax.set_title(f"MLE Fit - {dist}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)
        plt.close()

# ---------------- HYPOTHESIS ----------------
    elif option == "Hypothesis Testing":
        col = st.selectbox("Column", num_cols)
        data = df[col].dropna()

        test = st.selectbox("Test", ["Z-test", "T-test"])

        sample_mean = np.mean(data)
        st.write("Sample Mean:", round(sample_mean, 4))

        mu = st.number_input("Hypothesized Mean", value=float(sample_mean))

        if test == "Z-test":
            z = (sample_mean - mu) / (np.std(data) / np.sqrt(len(data)))
            st.metric("Z", round(z, 4))

            p = 2 * (1 - stats.norm.cdf(abs(z)))

            if p < 0.05:
                st.error("Reject H₀")
            else:
                st.success("Accept H₀")

        elif test == "T-test":
            t, p = stats.ttest_1samp(data, mu)
            st.metric("T", round(t, 4))

            if p < 0.05:
                st.error("Reject H₀")
            else:
                st.success("Accept H₀")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;">
A-01 Yash Arjugade | A-13 Aaryan Ghori<br>
KJ Somaiya Institute of Technology
</div>
""", unsafe_allow_html=True)