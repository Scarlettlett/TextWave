from flask import Flask, request, jsonify
import sys
import os
import pandas as pd
from pipeline import Pipeline
from PIL import Image
import io
import time


app = Flask(__name__)

# Initialize the pipeline
pipeline = Pipeline(index_type='HNSW', rerank_type="hybrid", temperature=0.6, generator_model="mistral-large-latest")


@app.route('/')
def home():
    return "Welcome to the question answering (Retrieval Augmented Generation) API! Use the /identify endpoint to start the authentication process."


# Endpoint to trigger the question-answering system
@app.route('/qa', methods=['POST'])
def question_answering():
    try:
        # Get the question from the request JSON
        data = request.json
        question = data.get('question')
        k = data.get('k', 20)  # Default value for k is 5

        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Start measuring latency for the full pipeline
        pipeline_start_time = time.time()

        # Retrieve top-k documents and generate an answer
        retrieved_docs = pipeline.search_neighbors(query=question, k=k, reporting=False)
        generated_answer = pipeline.generate_answer(query=question, context=retrieved_docs, rerank=True, reporting=False)

        # End measuring full pipeline latency
        pipeline_end_time = time.time()
        total_pipeline_latency = pipeline_end_time - pipeline_start_time

        # Return the generated answer
        return jsonify({
            "question": question,
            "generated_answer": generated_answer,
            "retrieved_documents": retrieved_docs,
            "latency_metrics": f"{total_pipeline_latency:.2f} seconds"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint to add documents to the corpus
@app.route('/add_document', methods=['POST'])
def add_document():
    try:
        # Get the document from the request JSON
        data = request.json
        document_id = data.get('document_id')
        document_text = data.get('document_text')

        if not document_id or not document_text:
            return jsonify({"error": "Both document_id and document_text are required"}), 400

        # Add the document to the corpus
        pipeline.add_to_corpus(document_id=document_id, document_text=document_text)

        # Return success response
        return jsonify({"message": f"Document with ID {document_id} added successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    
if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

    # To run, input below in terminal:
    # python app.py
