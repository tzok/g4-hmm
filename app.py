import streamlit as st


def process_text(input_text):
    # This is a sample processing function
    # You can replace it with your actual processing logic
    return input_text.upper()


def main():
    st.title("G4 loop location prediction")

    # Create text input
    user_input = st.text_input("Enter your G4 sequence:")

    # Create button
    if st.button("Process"):
        if user_input:
            # Process the input and display result
            result = process_text(user_input)
            st.write("Result:")
            st.code(result, language=None)
        else:
            st.warning("Please enter some text")


if __name__ == "__main__":
    main()
