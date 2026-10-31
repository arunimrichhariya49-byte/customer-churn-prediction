import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Show popup notifications if stored from a previous button click
if "popup_msg" in st.session_state and st.session_state.popup_msg:
    st.toast(st.session_state.popup_msg)
    st.success(st.session_state.popup_msg)
    st.session_state.popup_msg = None

st.title("Multi-Company Customer Churn Prediction System")
st.write("Objective: Manage company datasets, predict churn for newly added customers using ML, and generate retention reports.")

# ==========================================
# 1. HELPER FUNCTION TO GENERATE CUSTOMERS
# ==========================================
def generate_customers(service_category, n, start_idx=1000):
    ids = [f"CUST-{start_idx + i}" for i in range(n)]
    tenure = np.random.randint(1, 48, n)
    
    if service_category == "OTT Streaming":
        monthly_fee = np.random.choice([199.0, 499.0, 649.0, 799.0], n)
        usage_metric = np.random.randint(1, 60, n)
        contract = np.random.choice(["Monthly Plan", "Annual Plan"], n, p=[0.7, 0.3])
        support_calls = np.random.randint(0, 5, n)
        logit = -0.5 - 0.04 * tenure - 0.05 * usage_metric + (contract == "Monthly Plan") * 1.2 + 0.4 * support_calls
        usage_col_name = "Monthly_Watch_Hours"
        
    elif service_category == "Telecom / ISP":
        monthly_fee = np.random.uniform(299, 1499, n).round(0)
        usage_metric = np.random.randint(5, 150, n)
        contract = np.random.choice(["Month-to-Month", "1-Year Contract", "2-Year Contract"], n, p=[0.55, 0.25, 0.20])
        support_calls = np.random.randint(0, 8, n)
        logit = -1.0 - 0.05 * tenure + 0.001 * monthly_fee + (contract == "Month-to-Month") * 1.3 + 0.35 * support_calls
        usage_col_name = "Monthly_Data_GB"
        
    else:  # E-Commerce
        monthly_fee = np.random.uniform(500, 8000, n).round(0)
        usage_metric = np.random.randint(1, 15, n)
        contract = np.random.choice(["Regular User", "Plus/Prime Member"], n, p=[0.6, 0.4])
        support_calls = np.random.randint(0, 6, n)
        logit = -0.3 - 0.03 * tenure - 0.15 * usage_metric + (contract == "Regular User") * 1.1 + 0.45 * support_calls
        usage_col_name = "Orders_Per_Month"

    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "CustomerID": ids,
        "Tenure_Months": tenure,
        "Plan_Type": contract,
        "Monthly_Spend_Rs": monthly_fee,
        usage_col_name: usage_metric,
        "Support_Tickets": support_calls,
        "Churn": churn
    })
    return df, usage_col_name

# ==========================================
# 2. INITIALIZE SESSION STATE (DATABASE)
# ==========================================
if "companies" not in st.session_state:
    np.random.seed(42)
    df1, col1 = generate_customers("OTT Streaming", 500, 1000)
    df2, col2 = generate_customers("Telecom / ISP", 500, 1000)
    df3, col3 = generate_customers("E-Commerce", 500, 1000)
    
    st.session_state.companies = {
        "Netflix (OTT)": {"category": "OTT Streaming", "df": df1, "usage_col": col1},
        "Jio Fiber (Telecom)": {"category": "Telecom / ISP", "df": df2, "usage_col": col2},
        "Zomato (E-Commerce)": {"category": "E-Commerce", "df": df3, "usage_col": col3}
    }

if "selected_company" not in st.session_state:
    st.session_state.selected_company = "Netflix (OTT)"

# Ensure selected_company is valid
company_list = list(st.session_state.companies.keys())
if st.session_state.selected_company not in company_list:
    st.session_state.selected_company = company_list[0]

# ==========================================
# 3. SIDEBAR: COMPANY SECTION & MODEL CONFIG
# ==========================================
st.sidebar.header("1. Company Section")

selected_company = st.sidebar.selectbox(
    "Select Active Company:",
    company_list,
    index=company_list.index(st.session_state.selected_company)
)
st.session_state.selected_company = selected_company

# Add Company Expander
with st.sidebar.expander("Add a New Company"):
    new_comp_name = st.text_input("Company Name (e.g., Hotstar, Airtel)")
    new_comp_cat = st.selectbox("Industry Type", ["OTT Streaming", "Telecom / ISP", "E-Commerce"])
    init_cust_count = st.number_input("Initial Historical Customers", min_value=50, max_value=2000, value=200, step=50)
    
    if st.button("Create Company"):
        if new_comp_name.strip() == "":
            st.sidebar.warning("Please enter a company name.")
        elif new_comp_name in st.session_state.companies:
            st.sidebar.warning("This company already exists!")
        else:
            new_df, new_col = generate_customers(new_comp_cat, int(init_cust_count), 1000)
            st.session_state.companies[new_comp_name] = {
                "category": new_comp_cat,
                "df": new_df,
                "usage_col": new_col
            }
            st.session_state.selected_company = new_comp_name
            st.session_state.popup_msg = f"Company '{new_comp_name}' has been created and selected!"
            st.rerun()

# Delete Company Expander
with st.sidebar.expander("Delete a Company"):
    comp_to_delete = st.selectbox("Select Company to Delete", company_list, key="del_comp_box")
    if st.button("Delete Selected Company"):
        if len(st.session_state.companies) <= 1:
            st.sidebar.error("Cannot delete the last remaining company!")
        else:
            del st.session_state.companies[comp_to_delete]
            st.session_state.selected_company = list(st.session_state.companies.keys())[0]
            st.session_state.popup_msg = f"Company '{comp_to_delete}' was deleted."
            st.rerun()

st.sidebar.header("2. Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Algorithm:",
    ["Random Forest Classifier", "Logistic Regression"]
)

# Load current company data
comp_info = st.session_state.companies[selected_company]
df = comp_info["df"]
usage_col = comp_info["usage_col"]
service_cat = comp_info["category"]

# ==========================================
# 4. PREPROCESS & TRAIN ML MODEL ON DATA
# ==========================================
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

num_cols = ["Tenure_Months", "Monthly_Spend_Rs", usage_col, "Support_Tickets"]
cat_cols = ["Plan_Type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
])

if model_choice == "Random Forest Classifier":
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
else:
    clf = LogisticRegression(random_state=42)

pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df["CustomerID"], test_size=0.25, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Top Metrics Bar
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Company", selected_company)
col2.metric("Total Customers in DB", len(df))
col3.metric("Model Accuracy", f"{acc:.1%}")
col4.metric("ROC-AUC Score", f"{auc:.3f}")

st.divider()

# ==========================================
# 5. APPLICATION TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Customer Data Manager (Add / Delete)",
    "2. Live What-If Simulator", 
    "3. Churn Risk Report (All Customers)", 
    "4. Model Evaluation"
])

# --- TAB 1: ADD RANDOM, ADD MANUAL (MODEL PREDICTS), OR DELETE CUSTOMERS ---
with tab1:
    st.subheader(f"Manage Customer Database for: {selected_company}")
    col_left, col_mid, col_right = st.columns(3)
    
    # Option A: Generate Random Customers
    with col_left:
        st.markdown("#### A. Bulk Generate Customers")
        st.write("Randomly generate customer records and add them to this company.")
        num_to_gen = st.number_input("Number of customers to generate:", min_value=1, max_value=2000, value=50, step=10)
        
        if st.button("Generate & Add Customers"):
            next_id = 1000 + len(df)
            extra_df, _ = generate_customers(service_cat, int(num_to_gen), start_idx=next_id)
            st.session_state.companies[selected_company]["df"] = pd.concat([df, extra_df], ignore_index=True)
            st.session_state.popup_msg = f"Added {num_to_gen} random customers to {selected_company}!"
            st.rerun()

    # Option B: Add a Single Customer Manually (Model Predicts Churn!)
    with col_mid:
        st.markdown("#### B. Add Customer Manually")
        st.write("Enter details below. The AI model will predict their Churn automatically!")
        with st.form("manual_customer_form"):
            default_id = f"CUST-{1000 + len(df)}"
            m_id = st.text_input("Customer ID / Name", value=default_id)
            m_tenure = st.number_input("Tenure (Months)", min_value=1, max_value=100, value=2)
            m_plan = st.selectbox("Plan Type", df["Plan_Type"].unique())
            m_spend = st.number_input("Monthly Spend (Rs)", min_value=50.0, max_value=20000.0, value=649.0)
            m_usage = st.number_input(f"{usage_col.replace('_', ' ')}", min_value=0, max_value=500, value=8)
            m_tickets = st.number_input("Support Tickets / Complaints", min_value=0, max_value=20, value=4)
            
            submitted = st.form_submit_button("Predict Churn & Save Customer")
            if submitted:
                # Prepare input features for the trained model
                new_features = pd.DataFrame([{
                    "Tenure_Months": int(m_tenure),
                    "Plan_Type": m_plan,
                    "Monthly_Spend_Rs": float(m_spend),
                    usage_col: int(m_usage),
                    "Support_Tickets": int(m_tickets)
                }])
                
                # Model predicts Churn (0 or 1) and Probability from previous data
                predicted_churn = int(pipeline.predict(new_features)[0])
                predicted_prob = float(pipeline.predict_proba(new_features)[0][1])
                status_str = "CHURN (1)" if predicted_churn == 1 else "STAY (0)"
                
                new_row = pd.DataFrame([{
                    "CustomerID": m_id,
                    "Tenure_Months": int(m_tenure),
                    "Plan_Type": m_plan,
                    "Monthly_Spend_Rs": float(m_spend),
                    usage_col: int(m_usage),
                    "Support_Tickets": int(m_tickets),
                    "Churn": predicted_churn
                }])
                
                st.session_state.companies[selected_company]["df"] = pd.concat([df, new_row], ignore_index=True)
                st.session_state.popup_msg = f"Customer '{m_id}' added! AI Predicted: {status_str} (Risk: {predicted_prob:.1%})"
                st.rerun()

    # Option C: Delete Customers
    with col_right:
        st.markdown("#### C. Delete Customers")
        st.write("Select one or more customers from the database to remove them.")
        cust_to_delete = st.multiselect(
            "Select Customer ID(s) to Delete:",
            options=df["CustomerID"].tolist()
        )
        if st.button("Delete Selected Customer(s)"):
            if not cust_to_delete:
                st.warning("Please select at least one customer ID to delete.")
            elif len(df) - len(cust_to_delete) < 20:
                st.error("Cannot delete! At least 20 customers are required to train the model.")
            else:
                updated_df = df[~df["CustomerID"].isin(cust_to_delete)].reset_index(drop=True)
                st.session_state.companies[selected_company]["df"] = updated_df
                st.session_state.popup_msg = f"Deleted {len(cust_to_delete)} customer(s) from {selected_company}."
                st.rerun()

    st.divider()
    st.markdown(f"#### Complete Company Database ({len(df)} Total Customers — Newest Added at Bottom)")
    st.dataframe(df, use_container_width=True)

# --- TAB 2: LIVE CUSTOMER PREDICTOR ---
with tab2:
    st.subheader(f"Live What-If Customer Simulator ({selected_company})")
    c1, c2 = st.columns(2)
    
    with c1:
        in_tenure = st.slider("Tenure (Months)", 1, 60, 3)
        in_plan = st.selectbox("Subscription Plan", df["Plan_Type"].unique())
        in_spend = st.number_input("Monthly Spend (Rs)", float(df["Monthly_Spend_Rs"].min()), float(df["Monthly_Spend_Rs"].max()), float(df["Monthly_Spend_Rs"].median()))
        in_usage = st.slider(f"{usage_col.replace('_', ' ')}", int(df[usage_col].min()), int(df[usage_col].max()), int(df[usage_col].median()))
        in_tickets = st.slider("Support Tickets / Complaints", 0, 15, 3)

    input_df = pd.DataFrame([{
        "Tenure_Months": in_tenure,
        "Plan_Type": in_plan,
        "Monthly_Spend_Rs": in_spend,
        usage_col: in_usage,
        "Support_Tickets": in_tickets
    }])

    pred_prob = pipeline.predict_proba(input_df)[0][1]

    with c2:
        st.markdown("### Prediction Output")
        if pred_prob >= 0.65:
            st.error(f"HIGH CHURN RISK ({pred_prob:.1%})")
            st.write("Recommended Action: Offer 20% discount on Annual Plan and assign priority customer support.")
        elif pred_prob >= 0.35:
            st.warning(f"MEDIUM CHURN RISK ({pred_prob:.1%})")
            st.write("Recommended Action: Send personalized engagement notification or cashback voucher.")
        else:
            st.success(f"LOW CHURN RISK ({pred_prob:.1%})")
            st.write("Recommended Action: Loyal customer. Continue standard engagement.")

# --- TAB 3: FULL CHURN RISK REPORT ---
with tab3:
    st.subheader(f"Churn Risk Report for All {len(df)} Customers in {selected_company}")
    st.write("The trained model evaluates every customer in the dataset (including newly added ones) and ranks their risk.")
    
    all_probs = pipeline.predict_proba(X)[:, 1]
    report_df = df.copy()
    report_df["Churn_Probability_%"] = (all_probs * 100).round(1)
    report_df["Risk_Level"] = pd.cut(
        all_probs, bins=[-0.01, 0.35, 0.65, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    
    def get_action(risk):
        if risk == "High Risk":
            return "Offer 20% Annual Plan Discount + Priority Support Call"
        elif risk == "Medium Risk":
            return "Send Rs.100 Cashback Voucher & App Notification"
        return "Standard engagement"

    report_df["Recommended_Action"] = report_df["Risk_Level"].apply(get_action)
    report_df_sorted = report_df.sort_values(by="Churn_Probability_%", ascending=False)
    
    st.dataframe(report_df_sorted, use_container_width=True)
    
    csv_data = report_df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Risk Report (CSV)", csv_data, f"{selected_company}_churn_report.csv", "text/csv")

# --- TAB 4: MODEL EVALUATION ---
with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Confusion Matrix (Test Set)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Stayed (0)", "Churned (1)"],
                    yticklabels=["Stayed (0)", "Churned (1)"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
    with col_b:
        st.markdown("#### Risk Tier Distribution (All Customers)")
        risk_counts = report_df["Risk_Level"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        risk_counts.plot(kind="bar", color=["#ef4444", "#f59e0b", "#10b981"], ax=ax2)
        ax2.set_ylabel("Customers")
        plt.xticks(rotation=0)
        st.pyplot(fig2)

# --- END OF FILE ---