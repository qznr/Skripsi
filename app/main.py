import os
import csv
import io
from functools import wraps
from flask import Blueprint, jsonify, request, render_template, Response
from db import get_db
import asyncio
from services import find_experts, run_pairwise_search, record_preference
from psycopg2.extras import RealDictCursor

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


# ==========================================
# ADMIN SECURE AREA
# ==========================================

def check_auth(username, password):
    """Check if a username / password combination is valid."""
    admin_user = os.environ.get('ADMIN_USER', 'admin')
    admin_pass = os.environ.get('ADMIN_PASS', 'admin')
    return username == admin_user and password == admin_pass

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Admin Area"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def get_denormalized_evaluations():
    """Helper function to fetch the flattened evaluation data."""
    conn = get_db()
    query = """
        SELECT 
            e.evaluation_id,
            q.query_text,
            ma.model_name AS model_a,
            mb.model_name AS model_b,
            CAST(e.latency_a->>'total_search_time' AS FLOAT) AS latency_a_sec,
            CAST(e.latency_b->>'total_search_time' AS FLOAT) AS latency_b_sec,
            e.results_identical,
            e.preference_submitted,
            CASE 
                WHEN e.preference_submitted = FALSE THEN 'Not Voted'
                WHEN e.winner_model_id IS NULL THEN 'Draw'
                ELSE mw.model_name 
            END as winner,
            e.created_at
        FROM evaluation_results e
        JOIN queries q ON e.query_id = q.query_id
        JOIN models ma ON e.model_a_id = ma.model_id
        JOIN models mb ON e.model_b_id = mb.model_id
        LEFT JOIN models mw ON e.winner_model_id = mw.model_id
        ORDER BY e.created_at DESC;
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query)
        return cursor.fetchall()

@bp.route('/admin')
@requires_auth
def admin_dashboard():
    """Renders the HTML table of the evaluation data."""
    evaluations = get_denormalized_evaluations()
    
    total_votes = sum(1 for e in evaluations if e['preference_submitted'])
    
    return render_template('admin.html', evaluations=evaluations, total_votes=total_votes)

@bp.route('/admin/export')
@requires_auth
def export_csv():
    """Generates and downloads the data as a CSV file."""
    evaluations = get_denormalized_evaluations()
    
    # Create an in-memory string buffer
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Write the header
    cw.writerow([
        'Evaluation ID', 'Created At', 'Query', 'Model A', 'Model B', 
        'Latency A (s)', 'Latency B (s)', 'Results Identical', 
        'Preference Submitted', 'Winner'
    ])
    
    # Write the data rows
    for e in evaluations:
        cw.writerow([
            e['evaluation_id'],
            e['created_at'],
            e['query_text'],
            e['model_a'],
            e['model_b'],
            round(e['latency_a_sec'], 4) if e['latency_a_sec'] else '',
            round(e['latency_b_sec'], 4) if e['latency_b_sec'] else '',
            e['results_identical'],
            e['preference_submitted'],
            e['winner']
        ])
    
    # Return the generated CSV as a downloadable file
    output = Response(si.getvalue(), mimetype='text/csv')
    output.headers["Content-Disposition"] = "attachment; filename=evaluations_export.csv"
    return output