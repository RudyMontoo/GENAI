import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate Research Tool", layout="wide")

st.title("Real Estate Research Tool")


with st.sidebar:
    url1 = st.text_input("Enter URL 1")
    url2 = st.text_input("Enter URL 2")
    url3 = st.text_input("Enter URL 3")

    process_url_button = st.button("Process URLs")

    status_placeholder = st.empty()

    if process_url_button:
        urls = [url for url in (url1, url2, url3) if url.strip()]
        if not urls:
            status_placeholder.error("You must enter at least one URL")
        else:
            status_placeholder.success("Processing URLs...")
            process_urls(urls)
            status_placeholder.success("URLs processed successfully")


st.write("")
st.write("")


left, center, right = st.columns([1, 2, 1])

with center:
    query = st.text_input(
        "Question",
        placeholder="Ask about mortgage rates, housing trends, Fed policy..."
    )

if query:
    try:
        with st.spinner("Generating answer..."):
            result = generate_answer(query)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.subheader("Sources")
            for src in result["sources"]:
                st.write(src)

    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        st.error("Something went wrong while generating the answer.")
        st.exception(e)
