
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Application heading
st.title("🤖 Agentic AI Assistant")
st.write(
    "Explore Agentic AI using a language model "
    "powered by Groq and LangChain."
)

st.divider()

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Select Groq Model",
        [
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant"
        ]
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    st.caption(
        "The selected model must be available "
        "to your Groq API key."
    )

# User input
question = st.text_area(
    "Enter your question",
    value="Explain Agentic AI in simple English "
          "with one real-world example.",
    height=120
)

# Generate response
if st.button("Generate Answer", type="primary"):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. "
            "Please configure your .env file."
        )

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Generating response..."):
                llm = ChatGroq(
                    model=model_name,
                    api_key=api_key,
                    temperature=temperature
                )

                response = llm.invoke(question)

            st.subheader("AI Response")
            st.markdown(response.content)

        except Exception as e:
            st.error("Unable to generate the response.")
            st.caption(
                "Check your API key, model availability, "
                "internet connection, and package versions."
            )
            st.code(str(e))

st.divider()
st.caption("Built with Streamlit, LangChain, and Groq")
