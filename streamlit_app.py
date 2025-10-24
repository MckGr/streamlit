import streamlit as st
from transformers import pipeline

# --- USTAWIENIA STRONY ---
st.set_page_config(
    page_title="Translator & Sentiment Analyzer",
    page_icon="",
    layout="centered"
)

# --- TYTUŁ I WSTĘP ---
st.title("Aplikacja NLP: Analiza wydzwieku i tłumaczenie tekstu")
st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=200)

st.write("""
Witaj w aplikacji **NLP z wykorzystaniem Hugging Face Transformers**!  
Możesz tutaj:
- przeanalizować **wydźwięk emocjonalny** tekstu w języku angielskim  
- przetłumaczyć tekst **z angielskiego na niemiecki**  

Aplikacja korzysta z modeli dostępnych na platformie [Hugging Face](https://huggingface.co/).
""")

option = st.selectbox(
    "Wybierz funkcję:",
    [
        "Wydźwięk emocjonalny tekstu (angielski)",
        "Tłumaczenie z angielskiego na niemiecki"
    ],
)

# --- OPCJA 1: Analiza sentymentu ---
if option == "Wydźwięk emocjonalny tekstu (angielski)":
    text = st.text_area("✍Wpisz tekst po angielsku:")
    if text:
        with st.spinner("Analizuję emocje..."):
            try:
                classifier = pipeline("sentiment-analysis")
                result = classifier(text)[0]
                label = result['label']
                score = result['score']
                st.success(f"Wydźwięk: **{label}** (pewność: {score:.2f})")
            except Exception as e:
                st.error(f"Błąd analizy: {e}")

# --- OPCJA 2: Tłumaczenie EN → DE ---
elif option == "Tłumaczenie z angielskiego na niemiecki":
    text = st.text_area("Wpisz tekst po angielsku do przetłumaczenia:")
    if text:
        with st.spinner("Tłumaczę tekst..."):
            try:
                translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
                translation = translator(text)[0]['translation_text']
                st.success("Tłumaczenie zakończone pomyślnie!")
                st.write("### Tłumaczenie:")
                st.info(translation)
            except Exception as e:
                st.error(f"Błąd tłumaczenia: {e}")

# --- STOPKA ---
st.divider()
st.caption("Autor: Maciek Grubek | Numer indeksu: s27758")
