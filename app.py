import pickle

import streamlit as st


def validate_sequence(sequence):
    """Validate if the sequence contains only ACGUT nucleotides."""
    valid_chars = set('ACGUT')
    sequence = sequence.upper()
    return all(char in valid_chars for char in sequence)

def process_text(input_text):
    # This is a sample processing function
    # You can replace it with your actual processing logic
    input_text = input_text.upper()
    if not validate_sequence(input_text):
        raise ValueError("Sequence must contain only A, C, G, U, or T nucleotides")
    return input_text


def main():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_refined.pkl", "rb") as f:
        model_refined = pickle.load(f)

    st.title("G4 loop location prediction")

    # Create text input
    user_input = st.text_input("Enter your G4 sequence:")

    # Create button
    if st.button("Process"):
        if user_input:
            try:
                # Process the input and display result
                result = process_text(user_input)
                st.write("Result:")
                st.code(result, language=None)
            except ValueError as e:
                st.error(str(e))
        else:
            st.warning("Please enter some text")


if __name__ == "__main__":
    main()
