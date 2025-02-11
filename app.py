import streamlit as st


def process_text(input_text):
    # This is a sample processing function
    # You can replace it with your actual processing logic
    return input_text.upper()


def main():
    st.title("Text Processor")

    # Create text input
    user_input = st.text_input("Enter your text:")

    # Create button
    if st.button("Process"):
        if user_input:
            # Process the input and display result
            result = process_text(user_input)
            st.write("Result:", result)
        else:
            st.warning("Please enter some text")


if __name__ == "__main__":
    main()
