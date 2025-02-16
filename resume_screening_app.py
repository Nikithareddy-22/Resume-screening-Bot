import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re

# ✅ Load pre-trained model and vectorizer
try:
    svc_model = pickle.load(open('clf.pkl', 'rb'))  
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
    le = pickle.load(open('encoder.pkl', 'rb'))  
except FileNotFoundError:
    st.error("One or more model files are missing. Ensure 'clf.pkl', 'tfidf.pkl', and 'encoder.pkl' exist in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()


# 🔹 Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)  # Remove URLs
    cleanText = re.sub('RT|cc', ' ', cleanText)  # Remove RTs
    cleanText = re.sub('#\S+\s', ' ', cleanText)  # Remove hashtags
    cleanText = re.sub('@\S+', ' ', cleanText)  # Remove mentions
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Remove non-ASCII characters
    cleanText = re.sub('\s+', ' ', cleanText)  # Remove extra spaces
    return cleanText.strip()


# 🔹 Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text.strip()


# 🔹 Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()


# 🔹 Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')  # Fallback for different encodings
    return text.strip()


# 🔹 Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")


# 🔹 Function to predict the category of a resume
def predict_resume_category(input_resume):
    # Clean the text
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the pre-trained TF-IDF model
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense array
    vectorized_text = vectorized_text.toarray()

    # Make prediction
    predicted_category = svc_model.predict(vectorized_text)

    # Get the actual category name from the label encoder
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# 🔹 Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("📄 Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format to predict its job category.")

    # File upload section
    uploaded_file = st.file_uploader("📤 Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("🔍 Show extracted text", False):
                st.text_area("📄 Extracted Resume Text", resume_text, height=300)

            # Predict category
            st.subheader("📌 Predicted Job Category")
            category = predict_resume_category(resume_text)
            st.success(f"🏆 The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"⚠️ Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
