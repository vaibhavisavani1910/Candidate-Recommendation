# gradio_app.py

import os
import json
import shutil
import tempfile
import re
import time
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient
from query_processor import ResumeMatcher
from resume_ingest import load_and_split_resumes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# ------------------ Load Environment Variables ------------------
load_dotenv(override=True)


openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")



client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["rag_db"]["embedded_resumes"]


embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Ensure the vector store is initialized with the correct collection and embedding model
vectorstore = MongoDBAtlasVectorSearch(collection=collection, embedding=embedding, index_name="vector_index")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    # google_api_key=google_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
matcher = ResumeMatcher(vectorstore, llm)

# ------------------ Database Management ------------------

def delete_all_resumes_from_db():
    """
    Deletes all existing documents from the MongoDB collection.
    """
    print("Deleting all existing resumes from the database...")
    try:
        result = collection.delete_many({})
        print(f"Successfully deleted {result.deleted_count} documents.")
        # Using a simple gray text for the output message
        return f"<p style='color: #333; text-align: center;'>Successfully deleted {result.deleted_count} resumes from the database.</p>"
    except Exception as e:
        print(f"An error occurred while deleting documents: {e}")
        # Using a distinct red for error messages
        return f"<p style='color: #ef4444; text-align: center;'>Error: {e}</p>"

# ------------------ Gradio Processing Logic ------------------

def process_resumes(resume_files, job_description):
    """
    Processes uploaded resumes, embeds them, and runs the matching pipeline.
    Now waits for the documents to be fully indexed using a marker document strategy.
    """
    if not resume_files:
        return "<p style='color: #ef4444; font-size: 1.25rem;'>Please upload at least one resume.</p>"

    import uuid
    from langchain_core.documents import Document

    temp_dir = tempfile.mkdtemp()
    temp_resume_paths = []

    try:
        delete_all_resumes_from_db()
        # Copy uploaded files to a temporary directory for processing
        for file_obj in resume_files:
            src_path = file_obj.name
            dest_path = os.path.join(temp_dir, os.path.basename(src_path))
            shutil.copy(src_path, dest_path)
            temp_resume_paths.append(dest_path)

        # Load, split resumes into chunks
        chunks = load_and_split_resumes(temp_resume_paths)

        # Add marker document to check when indexing is complete
        marker_text = f"INDEX_MARKER_{uuid.uuid4()}"
        marker_doc = Document(page_content=marker_text, metadata={"index_marker": True})
        chunks.append(marker_doc)

        # Save to vector store
        MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embedding,
            collection=collection,
            index_name="vector_index"
        )
        print("Resumes and marker document embedded and saved.")

        # --- Wait for marker to be indexed ---
        print("Waiting for indexing to complete...")
        timeout = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                results = vectorstore.similarity_search(marker_text, k=1)
                if results and marker_text in results[0].page_content:
                    print("Indexing complete.")
                    break
            except Exception as e:
                print(f"Waiting for index... {e}")
            time.sleep(1)
        else:
            return "<p style='color: #ef4444; text-align: center;'>Timeout waiting for resumes to be indexed. Please try again.</p>"

        # Run the resume matching pipeline
        results = matcher.run_pipeline(job_description)

        if not results:
            return "<p style='color: #333; font-size: 1.25rem; text-align: center;'>No matches found for the provided job description.</p>"

        # -------------------- BUILD RESULTS TABLE --------------------
        display_table = "<h3 style='font-size: 1.5rem; font-weight: bold; color: #333; text-align: center;'>Top Resume Matches</h3><br>"
        for res in results:
            evaluation = res.get('evaluation', 'No evaluation found.')

            # Extract JSON block from LLM response
            json_match = re.search(r'\{.*\}', evaluation, re.DOTALL)
            evaluation_json = {}
            if json_match:
                json_string = json_match.group(0)
                try:
                    evaluation_json = json.loads(json_string)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
                    evaluation_json = {'summary': 'Could not parse JSON response.', 'criteria': []}
            else:
                evaluation_json = {'summary': 'No JSON found in LLM response.', 'criteria': []}

            summary = evaluation_json.get('summary', 'Summary not available.')
            criteria = evaluation_json.get('criteria', [])

            candidate_name = 'N/A'  # Placeholder (can be extracted later if needed)

            display_table += f"""
            <div style='background-color: #f8f8f8; border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 8px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                    <p style='font-size: 1.25rem; font-weight: bold; color: #555;'>Resume ID: {res.get('resume_id', 'N/A')}</p>
                    
                    <p style='font-size: 1.25rem; font-weight: bold; color: #555;'>Cosine Similarity: {res.get('cosine_similarity', 'N/A')}</p>
                </div>

                <h4 style='font-size: 1.15rem; font-weight: bold; color: #333; margin-top: 0;'>Summary:</h4>
                <p style='font-size: 1.1rem; color: #555;'>{summary}</p><br>

                <h4 style='font-size: 1.15rem; font-weight: bold; color: #333;'>Criteria Evaluation:</h4>
                <table style='width:100%; border-collapse:collapse; background-color: #eee;'>
                    <thead>
                        <tr>
                            <th style='background-color: #e0e0e0; color: black; padding: 10px; font-size: 1.1rem; font-weight: bold; text-align: left;'>Criterion</th>
                            <th style='background-color: #e0e0e0; color: black; padding: 10px; font-size: 1.1rem; font-weight: bold;'>Score</th>
                            <th style='background-color: #e0e0e0; color: black; padding: 10px; font-size: 1.1rem; font-weight: bold; text-align: left;'>Justification</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for c in criteria:
                display_table += f"""
                        <tr>
                            <td style='border:1px solid #ccc; padding:10px; font-size:1.05rem; color: #333;'>{c.get('name', 'N/A')}</td>
                            <td style='border:1px solid #ccc; padding:10px; font-size:1.05rem; color: #333; text-align: center;'>{c.get('score', 'N/A')}</td>
                            <td style='border:1px solid #ccc; padding:10px; font-size:1.05rem; color: #333;'>{c.get('justification', 'N/A')}</td>
                        </tr>
                """
            display_table += "</tbody></table></div>"

        return display_table

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# ------------------ Gradio UI ------------------

# Custom CSS for a more polished look
custom_css = """
body {
    font-family: 'Inter', sans-serif;
    background-color: white; /* Changed from dark gray to white */
    color: black; /* Changed from light gray to black */
}
.gradio-container {
    background-color: white; /* Changed from dark gray to white */
    border-radius: 1rem;
    padding: 2rem;
}
.markdown-heading h1 {
    font-size: 2.5rem;
    color: black; /* Changed from white to black */
    text-align: center;
}
.gr-button {
    background-color: #eee !important; /* Changed from gray to light gray */
    color: black !important; /* Changed from white to black */
    border-radius: 0.5rem !important;
    font-weight: bold !important;
    padding: 0.75rem 1.5rem !important;
    border: 1px solid #ccc; /* Added a light border */
}
.gr-button.secondary {
    background-color: #f8f8f8 !important; /* Lighter background for the secondary button */
}
.loader {
    border: 4px solid #eee; /* Changed from gray to light gray */
    border-top: 4px solid #555; /* Changed from gray to a darker gray */
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* New CSS for input components */
.gradio-input-text, .gradio-input-file {
    background-color: white !important; /* Changed from dark gray to white */
    color: black !important; /* Changed from light gray to black */
    border: 1px solid #ccc !important; /* Changed border to light gray */
    border-radius: 0.5rem !important;
}
.gradio-input-text::placeholder {
    color: #888 !important; /* Changed from light gray to a darker gray for visibility */
}
"""

def show_loader():
    """Returns the Gradio HTML component for the loader."""
    return gr.HTML("<div class='loader'></div>", visible=True)

def hide_loader():
    """Returns the Gradio HTML component to hide the loader."""
    return gr.HTML(visible=False)

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🧠 Resume Matcher")
    with gr.Row():
        resumes = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload Resumes", elem_classes="gradio-input-file")
    job_desc = gr.Textbox(label="Paste Job Description", lines=10, placeholder="Enter job description here...", elem_classes="gradio-input-text")
    
    with gr.Row():
        calculate_btn = gr.Button("Calculate", scale=2)
        # delete_btn = gr.Button("Delete Old Resumes", scale=1, variant="secondary")

    loader_output = gr.HTML(value="", visible=False)
    delete_status_output = gr.HTML(value="")
    output = gr.HTML()

    # The calculate button chain
    calculate_btn.click(
        fn=show_loader,
        outputs=loader_output,
    ).then(
        fn=process_resumes,
        inputs=[resumes, job_desc],
        outputs=output,
    ).then(
        fn=hide_loader,
        outputs=loader_output,
    )
    

demo.launch(share=True)
