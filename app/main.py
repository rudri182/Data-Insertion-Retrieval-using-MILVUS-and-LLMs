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
                st.subheader("Similarity Scores:")

                for i, (embedding, similarity) in enumerate(results):
                    st.write(f"Rank {i+1}: Similarity Score: {similarity}")

                # Prepare context for question answering
                context = " ".join([str(result) for result in results])

                # Get the answer from the question answering model
                answer = answer_question(query, context)
                
                st.subheader("Answer:")
                st.write(answer)

        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()