from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).with_name("data_set.csv")
MODEL_PATHS = {
    "Logistic Regression Basic": Path(__file__).with_name("logistic_regression_model.pkl"),
    "Random Forest": Path(__file__).with_name("random_forest_model.pkl"),
}
DISPLAY_FEATURES = [
    "Age",
    "Fare_log",
    "Sex_Encoded",
    "Pclass",
    "family",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S",
]


st.set_page_config(
    page_title="Titanic Survival Dashboard",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("Unknown")
    df["family"] = df.get("family", df["SibSp"] + df["Parch"])
    df["family_group"] = pd.cut(
        df["family"],
        bins=[-1, 0, 3, 6, np.inf],
        labels=["Alone", "Small", "Medium", "Large"],
    )
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 25, 45, 60, np.inf],
        labels=["Child", "Young Adult", "Adult", "Middle Aged", "Senior"],
        include_lowest=True,
    )
    df["Pclass_label"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    df["Survival"] = df["Survived"].map({0: "Did Not Survive", 1: "Survived"})
    return df


@st.cache_resource
def load_models() -> tuple[dict, dict]:
    models = {}
    errors = {}
    for model_name, model_path in MODEL_PATHS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with model_path.open("rb") as file:
                    models[model_name] = pickle.load(file)
        except Exception as exc:
            errors[model_name] = str(exc)
    return models, errors


def survival_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return df["Survived"].mean() * 100


def rate_by(df: pd.DataFrame, column: str, title: str, labels: dict | None = None):
    grouped = (
        df.groupby(column, observed=True)["Survived"]
        .mean()
        .mul(100)
        .reset_index(name="Survival Rate")
    )
    fig = px.bar(
        grouped,
        x=column,
        y="Survival Rate",
        text=grouped["Survival Rate"].map(lambda value: f"{value:.1f}%"),
        color="Survival Rate",
        color_continuous_scale=["#c94f4f", "#e5b65b", "#2f8f7f"],
        labels=labels or {column: column, "Survival Rate": "Survival Rate (%)"},
        title=title,
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_showscale=False,
        title_font_size=18,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def numeric_scaling_stats(training_df: pd.DataFrame) -> dict:
    return {
        "Age": {
            "mean": training_df["Age"].mean(),
            "std": training_df["Age"].std(),
        },
        "Fare_log": {
            "mean": training_df["Fare_log"].mean(),
            "std": training_df["Fare_log"].std(),
        },
    }


def scale_numeric(series: pd.Series, stats: dict, column: str) -> pd.Series:
    column_stats = stats[column]
    return (series - column_stats["mean"]) / column_stats["std"]


def build_model_features(
    source_df: pd.DataFrame,
    feature_names: list[str],
    scaling_stats: dict,
) -> pd.DataFrame:
    features = pd.DataFrame(index=source_df.index)
    family_size = source_df["family"]
    fare_log = source_df["Fare_log"] if "Fare_log" in source_df else np.log1p(source_df["Fare"])

    values = {
        "Age": scale_numeric(source_df["Age"], scaling_stats, "Age"),
        "Fare_log": scale_numeric(fare_log, scaling_stats, "Fare_log"),
        "Sex_Encoded": source_df["Sex"].map({"female": 0, "male": 1}).fillna(0),
        "Pclass": source_df["Pclass"],
        "family": family_size,
        "Embarked_C": (source_df["Embarked"] == "C").astype(int),
        "Embarked_Q": (source_df["Embarked"] == "Q").astype(int),
        "Embarked_S": (source_df["Embarked"] == "S").astype(int),
        "AgeGroup_0-9": source_df["Age"].between(0, 9, inclusive="both").astype(int),
        "AgeGroup_10-19": source_df["Age"].between(10, 19, inclusive="both").astype(int),
        "AgeGroup_20-29": source_df["Age"].between(20, 29, inclusive="both").astype(int),
        "AgeGroup_30-39": source_df["Age"].between(30, 39, inclusive="both").astype(int),
        "AgeGroup_40-49": source_df["Age"].between(40, 49, inclusive="both").astype(int),
        "AgeGroup_50-59": source_df["Age"].between(50, 59, inclusive="both").astype(int),
        "Pclass_1": (source_df["Pclass"] == 1).astype(int),
        "Pclass_2": (source_df["Pclass"] == 2).astype(int),
        "FamilyGroup_0-2": family_size.between(0, 2, inclusive="both").astype(int),
        "FamilyGroup_3-4": family_size.between(3, 4, inclusive="both").astype(int),
    }

    for feature_name in feature_names:
        features[feature_name] = values.get(feature_name, 0)
    return features


def predict_passenger(
    model,
    passenger_df: pd.DataFrame,
    scaling_stats: dict,
) -> tuple[int, float | None, pd.DataFrame]:
    feature_names = list(getattr(model, "feature_names_in_", []))
    x_values = build_model_features(passenger_df, feature_names, scaling_stats)
    prediction = int(model.predict(x_values)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x_values)[0, 1] * 100)
    return prediction, probability, x_values


df = load_data()
scaling_stats = numeric_scaling_stats(df)
models, model_errors = load_models()

st.title("Titanic Survival Dashboard")
st.caption("Interactive survival analysis across passenger demographics, class, fare, and family size.")

with st.sidebar:
    st.header("Filters")
    selected_sex = st.selectbox("Sex", ["All"] + sorted(df["Sex"].dropna().unique().tolist()))
    selected_pclass = st.selectbox("Passenger Class", ["All", 1, 2, 3])
    selected_embarked = st.selectbox(
        "Embarkation Port",
        ["All"] + sorted(df["Embarked"].dropna().unique().tolist()),
    )
    age_range = st.slider(
        "Age Range",
        float(df["Age"].min()),
        float(df["Age"].max()),
        (float(df["Age"].min()), float(df["Age"].max())),
    )
    selected_family_size = st.selectbox(
        "Family Size",
        ["All", "Alone", "Small", "Medium", "Large"],
    )

filtered_df = df.copy()
if selected_sex != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == selected_sex]
if selected_pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == selected_pclass]
if selected_embarked != "All":
    filtered_df = filtered_df[filtered_df["Embarked"] == selected_embarked]
filtered_df = filtered_df[
    (filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])
]
if selected_family_size != "All":
    filtered_df = filtered_df[filtered_df["family_group"].astype(str) == selected_family_size]

overall_survival_rate = survival_rate(filtered_df)
female_survival_rate = survival_rate(filtered_df[filtered_df["Sex"] == "female"])
male_survival_rate = survival_rate(filtered_df[filtered_df["Sex"] == "male"])
first_class_survival_rate = survival_rate(filtered_df[filtered_df["Pclass"] == 1])

kpi_cols = st.columns(4)
kpi_cols[0].metric("Overall Survival Rate", f"{overall_survival_rate:.2f}%")
kpi_cols[1].metric("Female Survival Rate", f"{female_survival_rate:.2f}%")
kpi_cols[2].metric("Male Survival Rate", f"{male_survival_rate:.2f}%")
kpi_cols[3].metric("1st Class Survival Rate", f"{first_class_survival_rate:.2f}%")

st.divider()

if filtered_df.empty:
    st.warning("No passengers match the selected filters.")
    st.stop()

analysis_tab, model_tab = st.tabs(["Survival Analysis", "Model Dashboard"])

with analysis_tab:
    st.subheader("Survival Patterns")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.plotly_chart(
            rate_by(filtered_df, "Sex", "Survival Rate by Gender", {"Sex": "Gender"}),
            use_container_width=True,
        )
        st.markdown(
            f"Female passengers survived at **{female_survival_rate:.1f}%**, compared with "
            f"**{male_survival_rate:.1f}%** for male passengers, showing a clear gender gap."
        )

    with chart_col_2:
        st.plotly_chart(
            rate_by(
                filtered_df,
                "Pclass_label",
                "Survival Rate by Passenger Class",
                {"Pclass_label": "Passenger Class"},
            ),
            use_container_width=True,
        )
        third_class_rate = survival_rate(filtered_df[filtered_df["Pclass"] == 3])
        st.markdown(
            f"1st class survival was **{first_class_survival_rate:.1f}%**, while 3rd class survival "
            f"was **{third_class_rate:.1f}%**, highlighting the role of passenger class."
        )

    chart_col_3, chart_col_4 = st.columns(2)
    with chart_col_3:
        st.plotly_chart(
            rate_by(
                filtered_df,
                "Embarked",
                "Survival Rate by Embarkation Port",
                {"Embarked": "Embarkation Port"},
            ),
            use_container_width=True,
        )
        st.markdown(
            "Survival varied by embarkation port, which can reflect differences in class mix, fares, "
            "and passenger demographics."
        )

    with chart_col_4:
        st.plotly_chart(
            rate_by(
                filtered_df,
                "age_group",
                "Survival Rate by Age Group",
                {"age_group": "Age Group"},
            ),
            use_container_width=True,
        )
        st.markdown(
            "Age groups reveal vulnerability patterns that are hidden in the overall average, especially "
            "for children and older passengers."
        )

    chart_col_5, chart_col_6 = st.columns(2)
    with chart_col_5:
        st.plotly_chart(
            rate_by(
                filtered_df,
                "family_group",
                "Survival Rate by Family Size",
                {"family_group": "Family Size"},
            ),
            use_container_width=True,
        )
        st.markdown(
            "Small family groups often show different survival outcomes than passengers traveling alone "
            "or in larger groups."
        )

    with chart_col_6:
        fare_column = "Fare_log" if "Fare_log" in filtered_df.columns else "Fare"
        fig_fare = px.box(
            filtered_df,
            x="Survival",
            y=fare_column,
            color="Survival",
            color_discrete_map={"Survived": "#2f8f7f", "Did Not Survive": "#c94f4f"},
            points="outliers",
            title="Fare Distribution for Survivors vs. Non-Survivors",
            labels={fare_column: "Log Fare" if fare_column == "Fare_log" else "Fare"},
        )
        fig_fare.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20), title_font_size=18)
        st.plotly_chart(fig_fare, use_container_width=True)
        st.markdown(
            "Fare differences reinforce the class insight: passengers who paid more generally had "
            "better access to survival opportunities."
        )

    st.subheader("Filtered Passenger Data")
    st.dataframe(
        filtered_df[
            [
                "PassengerId",
                "Name",
                "Survived",
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "family",
                "Fare",
                "Embarked",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with model_tab:
    st.subheader("Predict Survival")

    if model_errors:
        for model_name, error_message in model_errors.items():
            if "libomp" in error_message:
                st.warning(f"{model_name} could not load because OpenMP is missing. On macOS, install it with `brew install libomp`.")
            else:
                st.warning(f"{model_name} could not load: {error_message}")

    if not models:
        st.error("No saved models could be loaded.")
    else:
        prediction_model_name = st.selectbox("Prediction Model", list(models.keys()))
        input_col_1, input_col_2, input_col_3 = st.columns(3)

        with input_col_1:
            input_pclass = st.selectbox("Passenger Class", [1, 2, 3], key="predict_pclass")
            input_sex = st.selectbox("Sex", ["female", "male"], key="predict_sex")
            input_age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                key="predict_age",
            )

        with input_col_2:
            input_sibsp = st.number_input(
                "Siblings / Spouses",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="predict_sibsp",
            )
            input_parch = st.number_input(
                "Parents / Children",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="predict_parch",
            )
            input_fare = st.number_input(
                "Fare",
                min_value=0.0,
                value=float(df["Fare"].median()),
                step=1.0,
                key="predict_fare",
            )

        with input_col_3:
            input_embarked = st.selectbox("Embarkation Port", ["S", "C", "Q"], key="predict_embarked")

        passenger_data = pd.DataFrame(
            [
                {
                    "PassengerId": 0,
                    "Pclass": input_pclass,
                    "Sex": input_sex,
                    "Age": input_age,
                    "SibSp": input_sibsp,
                    "Parch": input_parch,
                    "family": input_sibsp + input_parch,
                    "Fare": input_fare,
                    "Fare_log": np.log1p(input_fare),
                    "Embarked": input_embarked,
                }
            ]
        )

        selected_model = models[prediction_model_name]
        prediction, probability, model_input = predict_passenger(
            selected_model,
            passenger_data,
            scaling_stats,
        )
        prediction_label = "Survived" if prediction == 1 else "Did Not Survive"

        prediction_cols = st.columns(3)
        prediction_cols[0].metric("Prediction", prediction_label)
        if probability is not None:
            prediction_cols[1].metric("Survival Probability", f"{probability:.2f}%")
            prediction_cols[2].metric("Risk Probability", f"{100 - probability:.2f}%")
        else:
            prediction_cols[1].metric("Survival Probability", "Unavailable")
            prediction_cols[2].metric("Risk Probability", "Unavailable")

        st.dataframe(
            passenger_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "family", "Fare", "Embarked"]],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Final model input"):
            display_input = build_model_features(passenger_data, DISPLAY_FEATURES, scaling_stats)
            st.dataframe(display_input, use_container_width=True, hide_index=True)
