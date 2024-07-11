import streamlit as st

from data_retreival import hybrid_retrieve, answer_question

# Streamlit User Interface
def main():
    st.title("Hybrid Retrieval and Question Answering System")

    st.write("Enter a query to retrieve relevant information and get an answer based on the context.")

    query = st.text_input("Query:", "")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Retrieving information..."):
                results = hybrid_retrieve(query, top_k=10)

                context = " ".join([result['text'] for result in results])
                answer = answer_question(query, context)

                st.subheader("Answer:")
                st.write(answer)

                st.subheader("Retrieved Context:")
                for i, result in enumerate(results):
                    st.write(f"Rank {i + 1}: {result['url']} - Similarity Score: {result['similarity']}")
                    st.write(result['text'])
                    st.write("---")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()