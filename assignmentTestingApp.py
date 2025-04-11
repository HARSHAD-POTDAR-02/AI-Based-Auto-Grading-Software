import os
import time
import re
import fitz
from flask import Flask, request, jsonify, render_template, redirect, url_for, session , send_file
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from docx import Document
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

app = Flask(__name__)
app.secret_key = "3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8"  # Replace with a secure key

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder to store the extracted text files
EXTRACTED_FOLDER = 'extracted_text'
if not os.path.exists(EXTRACTED_FOLDER):
    os.makedirs(EXTRACTED_FOLDER)

# Folder to store the ideal answers
IDEAL_ANSWER_FOLDER = 'ideal_answers'
if not os.path.exists(IDEAL_ANSWER_FOLDER):
    os.makedirs(IDEAL_ANSWER_FOLDER)

# Folder to store evaluation reports
EVALUATION_FOLDER = 'evaluation_reports'
if not os.path.exists(EVALUATION_FOLDER):
    os.makedirs(EVALUATION_FOLDER)

# Initialize Azure Form Recognizer client
# Replace with your Azure Form Recognizer endpoint and key
endpoint = "https://harshad.cognitiveservices.azure.com/"
api_key = "8KEwKkCvjI9TbWiZYHK5Qv9hRuqaSslnsuFCgIEw9rNjskTGPKpMJQQJ99BBACGhslBXJ3w3AAALACOGJjok"  # Replace with your actual API key
client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

# Constants
MAX_FILE_SIZE_MB = 2  # Maximum file size in MB
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

# LLaMA Model Configuration for text generation
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device=-1  # Set to -1 to use CPU
)


def generate_response(prompt, max_new_tokens=256):
    """Generate a response using the LLaMA model and return plain text."""
    messages = [{"role": "user", "content": prompt}]
    try:
        outputs = pipe(messages, max_new_tokens=max_new_tokens)
        print(f"Model Output Structure: {outputs}")  # Debugging

        response = ""
        if isinstance(outputs, list) and len(outputs) > 0:
            first_output = outputs[0]
            if 'generated_text' in first_output:
                gen_text = first_output['generated_text']
                # If gen_text is a list (nested format), extract assistant content
                if isinstance(gen_text, list):
                    assistant_texts = [entry.get("content", "") for entry in gen_text if entry.get("role", "") == "assistant"]
                    # If no entries are labeled as assistant, join all content entries
                    if not assistant_texts:
                        assistant_texts = [entry.get("content", "") for entry in gen_text]
                    response = " ".join(assistant_texts)
                else:
                    response = str(gen_text)
            else:
                # Fallback if no 'generated_text' key is present
                if first_output.get("role", "") == "assistant":
                    response = first_output.get("content", "")
                else:
                    response = str(first_output)
        else:
            response = "No response generated."
        return response.strip()
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "Error generating response."


def correct_spelling(text):
    # Adjusted to avoid over-correction
    return text  # Temporarily disable spell correction


def format_extracted_text(text):
    """
    Insert extra blank lines before each question marker (e.g. 'Q1.', 'Q2.', etc.)
    to improve readability.
    """
    # This regex will add a blank line before any occurrence of "Q" followed by a digit and a dot.
    formatted_text = re.sub(r'\s*(Q\d+\.)', r'\n\n\1', text)
    return formatted_text


# Function to process PDF documents with Azure Form Recognizer
def process_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document("prebuilt-layout", f)
            result = poller.result()

        extracted_text = ""
        for page in result.pages:
            extracted_text += f"Page {page.page_number}:\n"
            for line in page.lines:
                extracted_text += f"{line.content}\n"

        # Format the text to add extra spaces (blank lines) before question markers
        extracted_text = format_extracted_text(extracted_text)
        return extracted_text

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return str(e)


def process_docx(file_path):
    try:
        print(f"Processing DOCX file: {file_path}")
        doc = Document(file_path)
        extracted_text = "\n".join([para.text for para in doc.paragraphs])
        print("DOCX text extraction completed.")
        # Optionally apply spell correction if needed
        corrected_text = extracted_text
        # Format the text to add extra spaces before question markers
        corrected_text = format_extracted_text(corrected_text)
        return corrected_text
    except Exception as e:
        print(f"Error processing DOCX: {e}")
        return ""


def is_file_size_valid(file_path):
    return os.path.getsize(file_path) <= MAX_FILE_SIZE


# Function to compress PDF files
def compress_pdf(input_pdf_path, output_pdf_path):
    pdf_document = fitz.open(input_pdf_path)
    pdf_document.save(output_pdf_path, garbage=4, deflate=True)
    pdf_document.close()


def extract_text(file_path):
    """Extract text from a file based on its extension."""
    if file_path.lower().endswith(".pdf"):
        return process_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return process_docx(file_path)
    else:
        print("Unsupported file format.")
        return ""


def split_questions(text):
    """Split text into individual questions using regex pattern."""
    print("Splitting text into individual questions.")
    text = re.sub(r'(Question|Q|question|q)\s*(\d+)[\.\)]', r'Q\2.', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*(\d+)[\.\)]', r'\nQ\1.', text)
    questions = re.split(r'(?:^|\n)(Q\d+\..*?)(?=(?:\nQ\d+\.|$))', text, flags=re.DOTALL)
    questions = [q.strip() for q in questions if q.strip()]
    questions = [q for q in questions if q.startswith('Q')]
    print(f"Total questions found: {len(questions)}")
    return questions


def split_student_answers(text):
    """
    Split the student answers text into individual answers based on blank lines.
    Adjust the regex if your answer sheets use a different delimiter.
    """
    parts = re.split(r'\n\s*\n', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def generate_ideal_answer(question, custom_prompt=""):
    subject_prompt = f"Explain the answer according to the needs of the question:\n{question}"
    full_prompt = f"{custom_prompt}\n{subject_prompt}" if custom_prompt else subject_prompt
    try:
        response = generate_response(full_prompt)
        if isinstance(response, str):
            return response.strip()
        else:
            return "Error: Generated response is not a string."
    except Exception as e:
        print(f"Error generating ideal answer: {e}")
        return "Error generating ideal answer"


###############################################
# Semantic Similarity Checker & Evaluation Code
###############################################

# Initialize the sentence transformer components for semantic similarity
SIM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
sim_tokenizer = AutoTokenizer.from_pretrained(SIM_MODEL_NAME)
sim_model = AutoModel.from_pretrained(SIM_MODEL_NAME)


def compute_semantic_similarity(text1, text2):
    """
    Compute semantic similarity between two texts using mean-pooled embeddings and cosine similarity.
    Returns a similarity percentage.
    """
    inputs1 = sim_tokenizer(text1, return_tensors='pt', truncation=True, padding=True)
    inputs2 = sim_tokenizer(text2, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        emb1 = sim_model(**inputs1).last_hidden_state.mean(dim=1)
        emb2 = sim_model(**inputs2).last_hidden_state.mean(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    similarity_percentage = cos_sim.item() * 100
    return similarity_percentage


def generate_evaluation_report(questions, ideal_answers, student_answers, total_marks=10):
    """
    Generate a text evaluation report comparing ideal and student answers.
    For each question, include:
      - The question text
      - The ideal answer (AI)
      - The student answer
      - Similarity percentage
      - Marks awarded (based on total_marks per question)
    The report is saved as a txt file in the EVALUATION_FOLDER.
    Also, includes total marks awarded over the total possible marks.
    """
    report_lines = []
    report_lines.append(f"Evaluation Report generated at: {time.ctime()}\n")
    report_lines.append("=" * 80 + "\n")

    total_awarded = 0.0
    num_evaluated = min(len(questions), len(ideal_answers), len(student_answers))

    for i in range(num_evaluated):
        question = questions[i]
        ideal_ans = ideal_answers[i]
        student_ans = student_answers[i]
        similarity = compute_semantic_similarity(ideal_ans, student_ans)
        marks_awarded = round((similarity / 100.0) * total_marks, 2)
        total_awarded += marks_awarded

        report_lines.append(f"Question {i + 1}:\n{question}\n")
        report_lines.append(f"Ideal Answer:\n{ideal_ans}\n")
        report_lines.append(f"Student Answer:\n{student_ans}\n")
        report_lines.append(f"Similarity: {similarity:.2f}%\n")
        report_lines.append(f"Marks Awarded: {marks_awarded} / {total_marks}\n")
        report_lines.append("-" * 80 + "\n")

    total_possible = num_evaluated * total_marks
    report_lines.append(f"Total Marks Awarded: {total_awarded} / {total_possible}\n")

    report_filename = f"evaluation_report_{int(time.time())}.txt"
    report_filepath = os.path.join(EVALUATION_FOLDER, report_filename)
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    return report_filename, report_filepath


###############################################
# End Semantic Similarity & Evaluation Code
###############################################

@app.route('/')
def home():
    return render_template('UploadQA.html')  # Ensure this template exists

@app.route('/submit', methods=['POST'])
def submit():
    try:
        print("Request received")
        print("Form data:", request.form)
        print("Files received:", request.files)

        custom_prompt = request.form.get('custom_prompt', '').strip()

        # Process question paper upload
        if 'question-paper' not in request.files:
            return jsonify({"error": "Question paper is missing in request"}), 400

        question_paper = request.files['question-paper']
        if question_paper.filename == '':
            return jsonify({"error": "No file selected for question paper"}), 400

        question_paper_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                           'question_paper' + os.path.splitext(question_paper.filename)[1])
        question_paper.save(question_paper_path)
        print(f"Question paper saved: {question_paper_path}")

        if not is_file_size_valid(question_paper_path):
            print(f"File size exceeds limit. Compressing {question_paper_path}...")
            compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_question_paper.pdf')
            compress_pdf(question_paper_path, compressed_path)
            question_paper_path = compressed_path
            print(f"Compressed file saved to: {question_paper_path}")

        if question_paper.filename.lower().endswith('.pdf'):
            question_paper_text = process_pdf(question_paper_path)
        elif question_paper.filename.lower().endswith('.docx'):
            question_paper_text = process_docx(question_paper_path)
        else:
            return jsonify({"error": "Unsupported file type for question paper"}), 400

        question_paper_txt_path = os.path.join(EXTRACTED_FOLDER, 'extracted_question_paper.txt')
        with open(question_paper_txt_path, 'w', encoding='utf-8') as f:
            f.write(question_paper_text)
        print(f"Question paper text saved to: {question_paper_txt_path}")

        # Process answer sheets
        answer_sheets = request.files.getlist('answer-sheets')
        if not answer_sheets or all(sheet.filename == '' for sheet in answer_sheets):
            return jsonify({"error": "No answer sheets uploaded"}), 400

        extracted_answers = []
        for answer_sheet in answer_sheets:
            if answer_sheet.filename == '':
                continue
            answer_sheet_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_sheet.filename)
            answer_sheet.save(answer_sheet_path)
            print(f"Answer sheet saved: {answer_sheet_path}")

            if not is_file_size_valid(answer_sheet_path):
                print(f"File size exceeds limit. Compressing {answer_sheet_path}...")
                compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_' + answer_sheet.filename)
                compress_pdf(answer_sheet_path, compressed_path)
                answer_sheet_path = compressed_path
                print(f"Compressed file saved to: {answer_sheet_path}")

            if answer_sheet.filename.lower().endswith('.pdf'):
                answer_sheet_text = process_pdf(answer_sheet_path)
            elif answer_sheet.filename.lower().endswith('.docx'):
                answer_sheet_text = process_docx(answer_sheet_path)
            else:
                return jsonify({"error": f"Unsupported file type for {answer_sheet.filename}"}), 400

            extracted_answers.append(answer_sheet_text)
            print(f"Extracted text from {answer_sheet.filename}: {answer_sheet_text[:100]}...")

        answers_txt_path = os.path.join(EXTRACTED_FOLDER, 'extracted_answers.txt')
        with open(answers_txt_path, 'w', encoding='utf-8') as f:
            for answer in extracted_answers:
                f.write(answer + "\n")
        print(f"Extracted answers saved to: {answers_txt_path}")

        # Split student answers into individual responses
        combined_student_answers = "\n\n".join(extracted_answers)
        student_answers_list = split_student_answers(combined_student_answers)
        print(f"Student answers split into {len(student_answers_list)} parts.")

        # Extract individual questions from the question paper text
        questions = split_questions(question_paper_text)
        if not questions:
            return jsonify({"error": "No questions found in the question paper"}), 400

        # Generate ideal answers and store them in a list for evaluation
        ideal_answers_list = []
        timestamp = int(time.time())
        ideal_answers_filename = f"ideal_answers_{timestamp}.txt"
        ideal_answers_path = os.path.join(IDEAL_ANSWER_FOLDER, ideal_answers_filename)

        with open(ideal_answers_path, 'w', encoding='utf-8') as f:
            f.write(f"Generated at: {time.ctime()}\n\n")
            f.write("=" * 80 + "\n")
            for idx, question in enumerate(questions, start=1):
                print(f"Generating ideal answer for Question {idx}...")
                ideal_answer = generate_ideal_answer(question, custom_prompt=custom_prompt)
                ideal_answers_list.append(ideal_answer)
                print(f"Question {idx} answer generated.")
                f.write(f"Question {idx}:\n{question}\n\n")
                f.write(f"Ideal Answer:\n{ideal_answer}\n\n")
                f.write("-" * 80 + "\n\n")
            print(f"Ideal answers saved to {ideal_answers_filename}")

        # Generate evaluation report using the split student answers
        evaluation_report_filename, evaluation_report_filepath = generate_evaluation_report(
            questions, ideal_answers_list, student_answers_list, total_marks=10
        )
        print(f"Evaluation report saved to {evaluation_report_filename}")

        # Store the filename in the session
        session["evaluation_report_file"] = evaluation_report_filename
        session.modified = True  # Ensure the session is saved
        print(f"Session updated with filename: {evaluation_report_filename}")

        # Redirect to the evaluation report page
        print("Redirecting to /evaluation_report...")
        return redirect(url_for('evaluation_report'))

    except Exception as e:
        print(f"Error in /submit route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluation_report')
def evaluation_report():
    # Debug session contents
    print(f"Session contents: {dict(session)}")

    # Get the evaluation report filename from the session
    evaluation_report_filename = session.get("evaluation_report_file")
    if not evaluation_report_filename:
        return "No evaluation report file found in session.", 404

    # Construct the full file path
    filepath = os.path.join(EVALUATION_FOLDER, evaluation_report_filename)
    print(f"Looking for file: {evaluation_report_filename}")
    print(f"Full path: {filepath}")

    # Check if the file exists
    if not os.path.exists(filepath):
        return f"Error: Evaluation report file not found at {filepath}.", 404

    # Read the file content
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"Content to render: {content}")  # Debug content
    except Exception as e:
        content = f"Error reading the evaluation report: {e}"

    # Render the template with the content
    return render_template("evaluation_report.html", content=content)


@app.route('/download-report')
def download_report():
    try:
        # Get filename from session
        evaluation_report_filename = session.get("evaluation_report_file")

        if not evaluation_report_filename:
            print("Error: No filename in session")
            return "Report not found", 404

        # Construct full path
        file_path = os.path.join(EVALUATION_FOLDER, evaluation_report_filename)
        print(f"Attempting to download file: {os.path.abspath(file_path)}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return "File not found", 404

        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            print(f"Error: Path is not a file {os.path.abspath(file_path)}")
            return "Invalid file path", 400

        # Send the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=evaluation_report_filename,
            mimetype='text/plain'
        )

    except Exception as e:
        print(f"Error in download_report: {str(e)}")
        return f"Download failed: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)