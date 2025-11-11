import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# --- Load API Key ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Streamlit page config ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard + Live Analyzer")

st.markdown("Analyze and visualize sentiment results from files **or** try live text analysis below.")

# --- Section 1: Live Sentiment Analyzer ---
st.header("💬 Live Sentiment Analyzer")

user_text = st.text_area("Enter any text below for instant sentiment analysis:", height=120)

if st.button("🔍 Analyze Sentiment"):
    if user_text.strip():
        with st.spinner("Analyzing sentiment..."):
            prompt = f"""
            Analyze the sentiment of the following text and respond ONLY in JSON format.

            Text: "{user_text}"

            Return JSON with fields:
            - sentiment: "Positive", "Negative", or "Neutral"
            - score: number between -1 (negative) and +1 (positive)
            - explanation: short reason (max 25 words)
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )

                raw_output = response.choices[0].message.content.strip()

                # 🧩 Extract JSON safely (ignore extra text if model adds any)
                json_start = raw_output.find("{")
                json_end = raw_output.rfind("}") + 1
                json_str = raw_output[json_start:json_end] if json_start != -1 else "{}"

                result = json.loads(json_str)

                # ✅ Display results
                st.success("✅ Sentiment Analysis Complete")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment", result.get("sentiment", "N/A"))
                col2.metric("Score", result.get("score", "N/A"))
                col3.metric("Explanation", result.get("explanation", "No explanation provided."))

                # 🪄 Option to show raw model response
                with st.expander("🔍 Show raw model output (for debugging)"):
                    st.code(raw_output, language="json")

            except json.JSONDecodeError:
                st.error("⚠️ Could not parse model output as valid JSON.")
                with st.expander("Show returned content"):
                    st.code(raw_output)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some text first.")


st.divider()

# --- Section 2: File Upload for Bulk Visualization ---
st.header("📁 Upload Sentiment Results (CSV or Excel)")

uploaded_file = st.file_uploader("Upload your sentiment_results.csv or .xlsx file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if {"sentiment", "score"}.issubset(df.columns):
        # --- Summary metrics ---
        st.subheader("🔍 Summary Insights")

        total_rows = len(df)
        avg_score = df["score"].mean()
        sentiment_counts = df["sentiment"].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Texts", total_rows)
        col2.metric("Average Score", f"{avg_score:.2f}")
        col3.metric("Dominant Sentiment", sentiment_counts.idxmax())

        # --- Sentiment Distribution Chart ---
        st.subheader("📈 Sentiment Distribution")

        fig_pie = px.pie(
            df,
            names="sentiment",
            title="Overall Sentiment Distribution",
            color="sentiment",
            color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"},
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- Histogram ---
        st.subheader("📊 Sentiment Score Histogram")

        fig_hist = px.histogram(
            df,
            x="score",
            nbins=20,
            title="Distribution of Sentiment Scores",
            color="sentiment",
            color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- Explore Data Table ---
        st.subheader("🗒️ Explore Text Samples by Sentiment")

        selected_sentiment = st.radio("Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])
        if selected_sentiment != "All":
            filtered_df = df[df["sentiment"] == selected_sentiment]
        else:
            filtered_df = df

        st.dataframe(
            filtered_df[["text", "sentiment", "score", "explanation"]]
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=400
        )

    else:
        st.error("The uploaded file must contain 'sentiment' and 'score' columns.")
else:
    st.info("⬆️ Upload a file to see charts or use the live analyzer above.")
