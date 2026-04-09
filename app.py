import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



def show_popup(message, color="#ff4b4b"):
    st.markdown(f"""
    <style>
    .popup {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: {color};
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
        animation: fadein 0.5s;
    }}
    @keyframes fadein {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>

    <div class="popup">
        🚨 {message}
    </div>

    <script>
    setTimeout(function() {{
        const el = window.parent.document.querySelector('.popup');
        if(el) el.remove();
    }}, 3000);
    </script>
    """, unsafe_allow_html=True)

    




# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="SentinelNet IDS", layout="wide")
st.title("🚀 SentinelNet - AI Intrusion Detection")

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    try:
        scaler    = joblib.load("models/scaler.pkl")
        pca       = joblib.load("models/pca.pkl")
        iso       = joblib.load("models/iso_model.pkl")
        ocsvm     = joblib.load("models/svm_model.pkl")
        feat_means= joblib.load("models/feature_means.pkl")
        return scaler, pca, iso, ocsvm, feat_means
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()

scaler, pca, iso, ocsvm, feature_means = load_models()
feature_names = [f"F{i}" for i in range(scaler.n_features_in_)]

# -------------------------------
# SIDEBAR
# -------------------------------
mode      = st.sidebar.radio("Select Mode", ["Upload CSV", "Manual Input", "Real-time Simulation"])
threshold = st.sidebar.slider("Anomaly Score Threshold", 0.1, 0.9, 0.6)

# -------------------------------
# SCORE NORMALISATION  (BUG FIX #1)
# MinMaxScaler on a single row always returns 1.0, so we use
# a sigmoid-based normalisation anchored to the training-time
# score distribution instead.
# -------------------------------
def sigmoid_norm(scores: np.ndarray) -> np.ndarray:
    """Map raw decision-function scores to [0, 1] via sigmoid.
    Lower decision-function values (more anomalous) → higher output score."""
    return 1.0 / (1.0 + np.exp(scores))          # invert: anomaly → high score

# -------------------------------
# PREDICT FUNCTION  (BUG FIX #1, #2)
# -------------------------------
def predict(df: pd.DataFrame):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Align columns: keep only the features the scaler knows
    n_expected = scaler.n_features_in_
    if df.shape[1] > n_expected:
        df = df.iloc[:, :n_expected]
    elif df.shape[1] < n_expected:
        # Pad missing columns with zeros
        for i in range(df.shape[1], n_expected):
            df[f"F{i}"] = 0.0

    X        = df.values.astype(float)
    X_scaled = scaler.transform(X)
    X_pca    = pca.transform(X_scaled)

    iso_scores  = iso.decision_function(X_pca)       # negative = anomaly
    svm_scores  = ocsvm.decision_function(X_pca)

    # BUG FIX #1 – use sigmoid norm, not per-batch MinMaxScaler
    iso_norm = sigmoid_norm(iso_scores)
    svm_norm = sigmoid_norm(svm_scores)

    ensemble_score = 0.6 * iso_norm + 0.4 * svm_norm  # still in [0, 1]
    y_pred         = (ensemble_score > threshold).astype(int)

    return y_pred, ensemble_score, X_scaled

# -------------------------------
# SEVERITY
# -------------------------------
# def get_severity(score: float) -> str:
#     if score > 0.85:   return "🔴 CRITICAL"
#     elif score > 0.65: return "🟠 HIGH"
#     elif score > 0.40: return "🟡 MEDIUM"
#     else:              return "🟢 LOW"
def get_severity(score):
    if score > 0.8:
        return "🔴CRITICAL"
    elif score > 0.6:
        return "🟠HIGH"
    elif score > 0.45:
        return "🟡MEDIUM"
    else:
        return "🟢LOW"

# -------------------------------
# ATTACK TYPE  (BUG FIX #3)
# Uses the original (un-scaled) feature array safely.
# -------------------------------
def detect_attack_type(row: np.ndarray) -> str:
    """Rule-based heuristic on the first few raw features."""
    try:
        dst_port   = float(row[0]) if len(row) > 0 else 0
        fwd_pkts   = float(row[2]) if len(row) > 2 else 0
        bwd_pkts   = float(row[3]) if len(row) > 3 else 0
        fwd_length = float(row[4]) if len(row) > 4 else 0

        if dst_port == 80 and fwd_pkts > 500:
            return "DDoS"
        elif dst_port == 22 and fwd_pkts > 100:
            return "Brute Force"
        elif fwd_length > 1000:
            return "Web Attack"
        elif bwd_pkts == 0:
            return "Port Scan"
        else:
            return "Suspicious Activity"
    except Exception:
        return "Unknown Attack"

# -------------------------------
# SHAP EXPLAINER  (BUG FIX #4, #5)
# Lazy-load inside a function so it only runs when needed,
# and we cache it correctly.
# -------------------------------
@st.cache_resource
def get_explainer():
    base       = np.array(feature_means, dtype=float).reshape(1, -1)
    background = np.repeat(base, 50, axis=0)
    background_scaled = scaler.transform(background)
    background_pca    = pca.transform(background_scaled)

    def model_fn(x_pca):
        # sigmoid-norm of IsolationForest scores so SHAP values are in [0,1]
        return sigmoid_norm(iso.decision_function(x_pca))

    return shap.KernelExplainer(model_fn, background_pca)

# -------------------------------
# SHAP PLOT  (BUG FIX #5 – was defined but never displayed)
# -------------------------------
def show_shap(X_scaled: np.ndarray, row_idx: int = 0):
    with st.spinner("Computing SHAP explanation…"):
        explainer = get_explainer()
        X_pca     = pca.transform(X_scaled)
        shap_vals = explainer.shap_values(X_pca[[row_idx]])  # shape (1, n_pca)

    n_pca = X_pca.shape[1]
    pca_feat_names = [f"PC{i+1}" for i in range(n_pca)]

    fig, ax = plt.subplots(figsize=(8, 3))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_vals[0],
            base_values   = explainer.expected_value,
            feature_names = pca_feat_names,
        ),
        show=False,
    )
    st.pyplot(fig)
    plt.close(fig)



# =========================================================
# 🔹 MODE 1: CSV UPLOAD
# =========================================================
if mode == "Upload CSV":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview:")
        st.dataframe(df.head())

        df.columns = df.columns.str.strip()
        df_numeric = df.select_dtypes(include=[np.number]).copy()

        if df_numeric.empty:
            st.error("❌ No numeric columns found")
            st.stop()

        # ✅ Row selection (mentor requirement)
        st.info(f"Total rows: {len(df_numeric)}")

        num_rows = st.slider(
            "Select number of rows to analyze",
            100,
            len(df_numeric),
            min(1000, len(df_numeric)),
            step=100
        )

        df_subset = df_numeric.iloc[:num_rows]

        st.success(f"Processing first {num_rows} rows...")

        try:
            y_pred, scores, X_scaled = predict(df_subset)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

        results = df.iloc[:num_rows].copy()
        results["Score"] = np.round(scores, 4)
        results["Label"] = np.where(y_pred == 1, "Attack", "Normal")
        results["Severity"] = [get_severity(s) for s in scores]

        raw_vals = df_subset.values
        results["Attack Type"] = [
            detect_attack_type(raw_vals[i]) if y_pred[i] == 1 else "—"
            for i in range(len(y_pred))
        ]

        st.subheader("📊 Results")
        st.dataframe(results.head(50))

        # Metrics
        n_attacks = int(y_pred.sum())
        st.metric("Attack Rate", f"{n_attacks/len(y_pred)*100:.2f}%")

        # Metrics
        n_attacks = int(y_pred.sum())
        attack_rate = n_attacks / len(y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(y_pred))
        c2.metric("Attacks", n_attacks)
        c3.metric("Attack Rate", f"{attack_rate*100:.2f}%")

        # -------------------------------
        # 🚨 FINAL SYSTEM ALERT (NEW)
        # -------------------------------
        st.subheader("🚨 System Status")

        # 🚨 POPUP ALERT (ADD HERE)
        if attack_rate > 0.3:
            st.toast("🚨 CRITICAL ATTACK DETECTED")

        elif attack_rate > 0.15:
            st.toast("⚠️ HIGH RISK TRAFFIC")

        elif attack_rate > 0.05:
            st.toast("SUSPICIOUS ACTIVITY DETECTED")

        else:
            st.toast("✅ NORMAL TRAFFIC")

        

        # 🔊 SOUND ALERT (ADD HERE)
        if attack_rate > 0.05:
                st.markdown("""
            <audio autoplay>
            <source src="https://www.soundjay.com/buttons/sounds/beep-07.mp3" type="audio/mpeg">
            </audio>
            """, unsafe_allow_html=True)
            

        # if attack_rate > 0.3:
        #     st.error(f"🚨 CRITICAL ATTACK DETECTED | Attack Rate: {attack_rate*100:.2f}%")

        # elif attack_rate > 0.15:
        #     st.warning(f"⚠️ HIGH RISK TRAFFIC | Attack Rate: {attack_rate*100:.2f}%")

        # elif attack_rate > 0.05:
        #     st.info(f"🔍 SUSPICIOUS ACTIVITY | Attack Rate: {attack_rate*100:.2f}%")

        # else:
        #     st.success(f"✅ NORMAL TRAFFIC | Attack Rate: {attack_rate*100:.2f}%")

        # Graph
        fig, ax = plt.subplots()
        ax.hist(scores, bins=30)
        ax.axvline(threshold, color="red")
        st.pyplot(fig)

        # Download
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv)

        # SHAP only for small data
        if num_rows < 5000:
            with st.expander("SHAP Explanation"):
                show_shap(X_scaled, 0)
        else:
            st.warning("SHAP disabled for large data")



# =========================================================
# 🔹 MODE 2: MANUAL INPUT
# =========================================================
elif mode == "Manual Input":
    st.subheader("✍️ Manual Input")

    col1, col2 = st.columns(2)
    with col1:
        dst_port      = st.slider("Destination Port",  0,  65535, 80)
        flow_duration = st.slider("Flow Duration",      0, 10_000_000, 500_000)
        fwd_packets   = st.slider("Fwd Packets",        0, 1000, 10)
    with col2:
        bwd_packets   = st.slider("Bwd Packets",        0, 1000, 10)
        fwd_length    = st.slider("Fwd Length",         0, 10000, 100)
        bwd_length    = st.slider("Bwd Length",         0, 10000, 100)

    if st.button("🔍 Predict"):
        # Build input row from feature means, then override the 6 known fields
        input_data    = np.array(feature_means, dtype=float).reshape(1, -1)
        input_data[0, 0] = dst_port
        input_data[0, 1] = flow_duration
        input_data[0, 2] = fwd_packets
        input_data[0, 3] = bwd_packets
        input_data[0, 4] = fwd_length
        input_data[0, 5] = bwd_length

        df_input               = pd.DataFrame(input_data, columns=feature_names)
        y_pred, scores, X_scaled = predict(df_input)

        score = float(scores[0])
        severity = get_severity(score)

        # ✅ ADD THIS LINE HERE
        attack_type = detect_attack_type(input_data[0])

        st.metric("Anomaly Score", f"{score:.4f}", delta=None)

        margin = 0.05

        if score > threshold + margin:
            decision = "ATTACK"
        elif score > threshold:
            decision = "SUSPICIOUS"
        else:
            decision = "NORMAL"

        # if score > threshold:
        #     attack_type = detect_attack_type(input_data[0])
        #     st.error(f"🚨 ATTACK DETECTED  |  {attack_type}  |  {severity}  |  Score: {score:.4f}")
        # else:
        #     st.success(f"✅ NORMAL TRAFFIC  |  {severity}  |  Score: {score:.4f}")
        margin = 0.05

        if score > threshold + margin:
            decision = "ATTACK"
        elif score > threshold:
            decision = "SUSPICIOUS"
        else:
            decision = "NORMAL"

        if decision == "ATTACK":
            st.error(f"🚨 ATTACK | {attack_type} | {severity} | Score: {score:.4f}")
        elif decision == "SUSPICIOUS":
            st.warning(f"⚠️ SUSPICIOUS TRAFFIC | Score: {score:.4f}")
        else:
            st.success(f"✅ NORMAL | Score: {score:.4f}")

        # SHAP explanation (BUG FIX #5 – now actually displayed)
        with st.expander("🔎 SHAP Explanation", expanded=False):
            show_shap(X_scaled, row_idx=0)



# =========================================================
# 🔹 MODE 3: REAL-TIME SIMULATION  (BUG FIX #7 – added history chart)
# =========================================================
elif mode == "Real-time Simulation":
    file = st.file_uploader("Upload CSV for Simulation", type=["csv"])

    if file:
        df       = pd.read_csv(file)
        df_num   = df.select_dtypes(include=[np.number]).copy()
        speed    = st.slider("Speed (rows / sec)", 1, 20, 5)
        max_rows = st.slider("Max rows to simulate", 10, min(len(df), 500), 100)

        if st.button("▶ Start Simulation"):
            status_box   = st.empty()
            chart_box    = st.empty()
            summary_box  = st.empty()

            history_scores = []
            history_labels = []

            for i in range(min(len(df_num), max_rows)):
                row_df = df_num.iloc[[i]]

                try:
                    y_pred, scores, _ = predict(row_df)
                except Exception:
                    continue

                score    = float(scores[0])
                label    = "Attack" if y_pred[0] == 1 else "Normal"
                severity = get_severity(score)

                history_scores.append(score)
                history_labels.append(label)

                # Status indicator
                if y_pred[0] == 1:
                    attack_type = detect_attack_type(df_num.values[i])
                    status_box.error(
                        f"Row {i+1} | 🚨 {attack_type} | {severity} | Score: {score:.4f}"
                    )
                else:
                    status_box.success(
                        f"Row {i+1} | ✅ Normal | {severity} | Score: {score:.4f}"
                    )

                # Live chart
                fig, ax = plt.subplots(figsize=(9, 2.5))
                colors  = ["red" if l == "Attack" else "green" for l in history_labels]
                ax.bar(range(len(history_scores)), history_scores, color=colors, width=1.0)
                ax.axhline(threshold, color="orange", linestyle="--", label="Threshold")
                ax.set_ylim(0, 1)
                ax.set_xlabel("Row")
                ax.set_ylabel("Score")
                ax.set_title("Live Anomaly Scores  (🔴 Attack  |  🟢 Normal)")
                ax.legend(fontsize=8)
                chart_box.pyplot(fig)
                plt.close(fig)

                # Running summary
                n_att = history_labels.count("Attack")
                summary_box.info(
                    f"Processed: {i+1}  |  Attacks: {n_att}  |  "
                    f"Rate: {n_att/(i+1)*100:.1f}%"
                )

                time.sleep(1 / speed)

            status_box.success("✅ Simulation complete.")


            
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import time
# import shap
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # -------------------------------
# # CONFIG
# # -------------------------------
# st.set_page_config(page_title="SentinelNet IDS", layout="wide")
# st.title("🚀 SentinelNet - AI Intrusion Detection")

# # -------------------------------
# # LOAD MODELS
# # -------------------------------
# @st.cache_resource
# def load_models():
#     return (
#         joblib.load("models/scaler.pkl"),
#         joblib.load("models/pca.pkl"),
#         joblib.load("models/iso_model.pkl"),
#         joblib.load("models/svm_model.pkl"),
#         joblib.load("models/feature_means.pkl"),
#     )

# scaler, pca, iso, ocsvm, feature_means = load_models()
# feature_names = [f"F{i}" for i in range(scaler.n_features_in_)]

# # -------------------------------
# # SIDEBAR
# # -------------------------------
# mode = st.sidebar.radio("Select Mode", ["Manual Input", "Upload CSV", "Real-time Simulation"])
# threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.6)

# # -------------------------------
# # PREDICT FUNCTION
# # -------------------------------
# def predict(df):
#     df = df.copy()
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df = df.fillna(0)

#     X = df.values

#     X_scaled = scaler.transform(X)
#     X_pca = pca.transform(X_scaled)

#     iso_scores = iso.decision_function(X_pca)
#     svm_scores = ocsvm.decision_function(X_pca)

#     iso_norm = MinMaxScaler().fit_transform(iso_scores.reshape(-1,1)).flatten()
#     svm_norm = MinMaxScaler().fit_transform(svm_scores.reshape(-1,1)).flatten()

#     ensemble_score = 0.7 * iso_norm + 0.3 * svm_norm

#     y_pred = (ensemble_score > threshold).astype(int)

#     return y_pred, ensemble_score, X_scaled

# # -------------------------------
# # SEVERITY
# # -------------------------------
# def get_severity(score):
#     if score > 0.85:
#         return "CRITICAL"
#     elif score > 0.65:
#         return "HIGH"
#     elif score > 0.4:
#         return "MEDIUM"
#     else:
#         return "LOW"

# # -------------------------------
# # ATTACK TYPE (RULE-BASED)
# # -------------------------------
# def detect_attack_type(row):
#     try:
#         if row[0] == 80 and row[2] > 500:
#             return "DDoS"
#         elif row[0] == 22 and row[2] > 100:
#             return "Brute Force"
#         elif row[4] > 1000:
#             return "Web Attack"
#         elif row[3] == 0:
#             return "Port Scan"
#         else:
#             return "Suspicious Activity"
#     except:
#         return "Unknown Attack"

# # -------------------------------
# # SHAP
# # -------------------------------
# @st.cache_resource
# def get_explainer():
#     base = np.array(feature_means).reshape(1, -1)
#     background = np.repeat(base, 50, axis=0)

#     def model_fn(x):
#         return iso.decision_function(pca.transform(x))

#     return shap.KernelExplainer(model_fn, background)

# explainer = get_explainer()

# def shap_explain(X):
#     return explainer.shap_values(X)

# # =========================================================
# # 🔹 MODE 1: MANUAL INPUT
# # =========================================================
# if mode == "Manual Input":

#     st.subheader("✍️ Manual Input")

#     col1, col2 = st.columns(2)

#     with col1:
#         dst_port = st.slider("Destination Port", 0, 65535, 80)
#         flow_duration = st.slider("Flow Duration", 0, 10000000, 500000)
#         fwd_packets = st.slider("Fwd Packets", 0, 1000, 10)

#     with col2:
#         bwd_packets = st.slider("Bwd Packets", 0, 1000, 10)
#         fwd_length = st.slider("Fwd Length", 0, 10000, 100)
#         bwd_length = st.slider("Bwd Length", 0, 10000, 100)

#     if st.button("🔍 Predict"):

#         input_data = np.array(feature_means).reshape(1, -1)

#         input_data[0][0] = dst_port
#         input_data[0][1] = flow_duration
#         input_data[0][2] = fwd_packets
#         input_data[0][3] = bwd_packets
#         input_data[0][4] = fwd_length
#         input_data[0][5] = bwd_length

#         df_input = pd.DataFrame(input_data, columns=feature_names)

#         y_pred, scores, X_scaled = predict(df_input)

#         score = float(scores[0])
#         severity = get_severity(score)

#         if score > threshold:
#             attack_type = detect_attack_type(input_data[0])
#             st.error(f"🚨 ATTACK | {attack_type} | {severity} | Score: {score:.4f}")
#         else:
#             st.success(f"✅ NORMAL | Score: {score:.4f}")

# # =========================================================
# # 🔹 MODE 2: CSV UPLOAD
# # =========================================================
# elif mode == "Upload CSV":

#     file = st.file_uploader("Upload CSV", type=["csv"])

#     if file:
#         df = pd.read_csv(file)

#         st.write("Preview:")
#         st.dataframe(df.head())

#         try:
#             y_pred, scores, X_scaled = predict(df)
#         except:
#             st.error("❌ Feature mismatch!")
#             st.stop()

#         df["Prediction"] = y_pred
#         df["Score"] = scores

#         df["Prediction"] = df["Prediction"].map({0: "Normal", 1: "Attack"})

#         st.subheader("Results")
#         st.dataframe(df.head(20))

#         st.write(f"Attack Rate: {(y_pred.mean()*100):.2f}%")

#         csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button("Download Results", csv, "results.csv")

# # =========================================================
# # 🔹 MODE 3: REAL-TIME SIMULATION
# # =========================================================
# elif mode == "Real-time Simulation":

#     file = st.file_uploader("Upload CSV for Simulation", type=["csv"])

#     if file:
#         df = pd.read_csv(file)

#         speed = st.slider("Speed (rows/sec)", 1, 20, 5)

#         if st.button("▶ Start Simulation"):

#             placeholder = st.empty()

#             for i in range(min(len(df), 100)):

#                 row = df.iloc[[i]]

#                 try:
#                     y_pred, scores, _ = predict(row)
#                 except:
#                     continue

#                 score = float(scores[0])
#                 severity = get_severity(score)

#                 if score > threshold:
#                     placeholder.error(f"🚨 ATTACK | {severity} | Score: {score:.4f}")
#                 else:
#                     placeholder.success(f"✅ NORMAL | Score: {score:.4f}")

#                 time.sleep(1/speed)