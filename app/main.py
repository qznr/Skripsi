from flask import Blueprint, jsonify, request, render_template
from db import get_db
import asyncio
from services import find_experts, run_pairwise_search, record_preference

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/db_check')
def db_check():
    """
    Checks the database connection and confirms pgvector is enabled.
    """
    try:
        conn = get_db()
        cursor = conn.cursor()

        # 1. Check the PostgreSQL server version
        cursor.execute('SELECT version();')
        pg_version = cursor.fetchone()[0]

        # 2. Check the pgvector extension version
        # The 'init-db' command should have already run CREATE EXTENSION.
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        pgvector_result = cursor.fetchone()

        if pgvector_result is None:
            raise Exception("pgvector extension is not installed in the database.")
        
        pgvector_version = pgvector_result[0]

        cursor.close()

        return jsonify({
            "status": "ok",
            "message": "Database connection successful.",
            "postgres_version": pg_version,
            "pgvector_version": pgvector_version
        })

    except Exception as e:
        error_message = f"Database connection failed: {e}"
        print(error_message)
        return jsonify({
            "status": "error",
            "message": error_message
        }), 500

@bp.route('/query', methods=['POST'])
def query_experts():
    """
    API endpoint for querying experts. Handles request/response and calls the service layer.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query_text = data['query']
    dims = data.get('dims', 256) 
    
    if not isinstance(dims, int) or not 0 < dims <= 768:
        return jsonify({"error": "Invalid 'dims'. Must be an integer between 1 and 768."}), 400

    try:
        # Call the business logic from the service layer
        results = asyncio.run(find_experts(query_text, dims))
        return jsonify(results)

    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred during the query process: {e}")
        return jsonify({
            "status": "error",
            "message": "An internal error occurred."
        }), 500
    
@bp.route('/search', methods=['POST'])
def search_experts_pairwise():
    """
    API endpoint for pairwise expert search. Handles request/response and calls the service layer.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query_text = data['query']

    try:
        results = asyncio.run(run_pairwise_search(query_text))
        return jsonify(results)

    except Exception as e:
        print(f"An error occurred during pairwise search: {e}")
        return jsonify({
            "status": "error",
            "message": f"An internal error occurred: {str(e)}"
        }), 500
    
@bp.route('/preference', methods=['POST'])
def record_user_preference():
    """
    API endpoint for recording the user's pairwise preference (A, B, or Draw).
    """
    data = request.get_json()
    
    required_fields = ['evaluation_id', 'choice']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields: evaluation_id or choice"}), 400

    evaluation_id = data['evaluation_id']
    choice = data['choice'].lower() # 'a', 'b', or 'draw'

    if choice not in ['a', 'b', 'draw']:
        return jsonify({"error": "Invalid choice. Must be 'a', 'b', or 'draw'"}), 400

    try:
        # Call service layer to update the database
        record_preference(evaluation_id, choice)
        return jsonify({"status": "success", "message": f"Preference '{choice}' recorded for evaluation {evaluation_id}"})

    except Exception as e:
        print(f"An error occurred while recording preference: {e}")
        return jsonify({
            "status": "error",
            "message": "An internal error occurred while saving preference."
        }), 500
