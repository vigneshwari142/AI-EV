
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Titanic EDA Dashboard",
    page_icon="🚢",
    layout="wide"
)


# ============================================================
# 2. APPLICATION TITLE
# ============================================================

st.title("🚢 Titanic Exploratory Data Analysis Dashboard")

st.write(
    "Interactive dashboard for exploring the Kaggle Titanic dataset."
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")


df = load_data()


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 EDA Navigation")

option = st.sidebar.selectbox(
    "Choose Analysis",
    [
        "Overview",
        "Data Preview",
        "Missing Values",
        "Statistics",
        "Univariate Analysis",
        "Categorical Analysis",
        "Survival Analysis",
        "Correlation",
        "Family Analysis"
    ]
)


# ============================================================
# 5. KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    df.shape[0]
)

col2.metric(
    "Columns",
    df.shape[1]
)

col3.metric(
    "Survivors",
    int(df["Survived"].sum())
)

col4.metric(
    "Survival Rate",
    f"{df['Survived'].mean() * 100:.2f}%"
)


st.divider()


# ============================================================
# 6. OVERVIEW
# ============================================================

if option == "Overview":

    st.header("📋 Dataset Overview")

    st.write("### Dataset Shape")
    st.write(f"Rows: **{df.shape[0]}**")
    st.write(f"Columns: **{df.shape[1]}**")

    st.write("### Column Names")

    st.write(df.columns.tolist())

    st.write("### Data Types")

    dtype_df = df.dtypes.astype(str).to_frame("Data Type")

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.write("### Dataset Information")

    st.write(
        "The Titanic dataset contains information about passengers "
        "such as age, gender, passenger class, fare, family members "
        "and survival status."
    )


# ============================================================
# 7. DATA PREVIEW
# ============================================================

elif option == "Data Preview":

    st.header("🔍 Dataset Preview")

    rows = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=min(100, len(df)),
        value=10
    )

    st.dataframe(
        df.head(rows),
        use_container_width=True
    )

    st.write("### Random Sample")

    sample_size = st.slider(
        "Number of random records",
        min_value=1,
        max_value=min(20, len(df)),
        value=5
    )

    st.dataframe(
        df.sample(sample_size),
        use_container_width=True
    )


# ============================================================
# 8. MISSING VALUES
# ============================================================

elif option == "Missing Values":

    st.header("❌ Missing Value Analysis")

    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": df.isnull().mean() * 100
    })

    missing = missing[
        missing["Missing Count"] > 0
    ]

    if missing.empty:

        st.success("No missing values found!")

    else:

        st.dataframe(
            missing,
            use_container_width=True
        )

        # Bar Chart

        fig, ax = plt.subplots()

        missing["Missing Count"].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Number of Missing Values")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        # Missing percentage chart

        fig, ax = plt.subplots()

        missing["Missing %"].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Missing Values Percentage")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Percentage (%)")

        plt.xticks(rotation=45)

        st.pyplot(fig)


# ============================================================
# 9. DESCRIPTIVE STATISTICS
# ============================================================

elif option == "Statistics":

    st.header("📈 Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.write("### Statistical Summary")

    st.write(
        "The table above provides statistical measures such as "
        "count, mean, standard deviation, minimum, maximum and "
        "quartiles for numerical variables."
    )


# ============================================================
# 10. UNIVARIATE NUMERICAL ANALYSIS
# ============================================================

elif option == "Univariate Analysis":

    st.header("📊 Univariate Numerical Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    selected_column = st.selectbox(
        "Select a numerical column",
        numeric_columns
    )

    # Histogram

    fig, ax = plt.subplots()

    sns.histplot(
        df[selected_column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )

    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # Boxplot

    st.write("### Box Plot")

    fig, ax = plt.subplots()

    sns.boxplot(
        x=df[selected_column],
        ax=ax
    )

    ax.set_title(
        f"Box Plot of {selected_column}"
    )

    st.pyplot(fig)

    # Statistics

    st.write("### Selected Column Statistics")

    st.write(
        df[selected_column].describe()
    )


# ============================================================
# 11. CATEGORICAL ANALYSIS
# ============================================================

elif option == "Categorical Analysis":

    st.header("📊 Categorical Data Analysis")

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    if len(categorical_columns) == 0:

        st.warning(
            "No categorical columns found."
        )

    else:

        selected_column = st.selectbox(
            "Select categorical column",
            categorical_columns
        )

        # Count table

        counts = (
            df[selected_column]
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            selected_column,
            "Count"
        ]

        st.write("### Category Counts")

        st.dataframe(
            counts,
            use_container_width=True
        )

        # Count Plot

        fig, ax = plt.subplots()

        sns.countplot(
            data=df,
            x=selected_column,
            ax=ax
        )

        ax.set_title(
            f"Count of {selected_column}"
        )

        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")

        plt.xticks(rotation=45)

        st.pyplot(fig)


# ============================================================
# 12. SURVIVAL ANALYSIS
# ============================================================

elif option == "Survival Analysis":

    st.header("🚢 Survival Analysis")

    selected_column = st.selectbox(
        "Analyze survival by:",
        [
            "Sex",
            "Pclass",
            "Embarked"
        ]
    )

    survival = (
        df.groupby(selected_column)["Survived"]
        .mean()
        .reset_index()
    )

    survival["Survival Rate"] *= 100

    st.write("### Survival Rate Table")

    st.dataframe(
        survival,
        use_container_width=True
    )

    # Bar Chart

    fig, ax = plt.subplots()

    sns.barplot(
        data=survival,
        x=selected_column,
        y="Survival Rate",
        ax=ax
    )

    ax.set_title(
        f"Survival Rate by {selected_column}"
    )

    ax.set_ylabel(
        "Survival Rate (%)"
    )

    ax.set_xlabel(
        selected_column
    )

    st.pyplot(fig)


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

elif option == "Correlation":

    st.header("🔥 Correlation Analysis")

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    st.write("### Correlation Matrix")

    st.dataframe(
        correlation,
        use_container_width=True
    )

    # Heatmap

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    st.pyplot(fig)


# ============================================================
# 14. FAMILY SIZE ANALYSIS
# ============================================================

elif option == "Family Analysis":

    st.header("👨‍👩‍👧 Family Size Analysis")

    data = df.copy()

    # Create FamilySize

    data["FamilySize"] = (
        data["SibSp"]
        + data["Parch"]
        + 1
    )

    # Survival by family size

    survival = (
        data.groupby("FamilySize")["Survived"]
        .mean()
        .reset_index()
    )

    survival["Survival Rate"] *= 100

    st.write("### Survival Rate by Family Size")

    st.dataframe(
        survival,
        use_container_width=True
    )

    # Line plot

    fig, ax = plt.subplots()

    sns.lineplot(
        data=survival,
        x="FamilySize",
        y="Survival Rate",
        marker="o",
        ax=ax
    )

    ax.set_title(
        "Survival Rate vs Family Size"
    )

    ax.set_xlabel(
        "Family Size"
    )

    ax.set_ylabel(
        "Survival Rate (%)"
    )

    st.pyplot(fig)

    # Family size categories

    data["FamilyCategory"] = pd.cut(
        data["FamilySize"],
        bins=[0, 1, 4, 7, 20],
        labels=[
            "Alone",
            "Small Family",
            "Medium Family",
            "Large Family"
        ]
    )

    category_survival = (
        data.groupby(
            "FamilyCategory",
            observed=False
        )["Survived"]
        .mean()
        .reset_index()
    )

    category_survival["Survival Rate"] *= 100

    st.write(
        "### Survival Rate by Family Category"
    )

    st.dataframe(
        category_survival,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=category_survival,
        x="FamilyCategory",
        y="Survival Rate",
        ax=ax
    )

    ax.set_title(
        "Survival Rate by Family Category"
    )

    ax.set_xlabel(
        "Family Category"
    )

    ax.set_ylabel(
        "Survival Rate (%)"
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)


# ============================================================
# 15. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    "Titanic EDA Dashboard\n\n"
    "Built using Python, Pandas, Matplotlib, "
    "Seaborn and Streamlit."
)

