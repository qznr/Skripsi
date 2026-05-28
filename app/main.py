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
    conn = get_db()
    query = """
        SELECT 
            e.evaluation_id, q.query_text,
            ma.model_name AS model_a, mb.model_name AS model_b,
            ma.model_id AS model_a_id, mb.model_id AS model_b_id,
            
            -- LATENCY STEPS (A)
            CAST(e.latency_a->>'step1_embedding' AS FLOAT) AS a_step1,
            CAST(e.latency_a->>'step2_shortlisting' AS FLOAT) AS a_step2,
            CAST(e.latency_a->>'step3_reranking' AS FLOAT) AS a_step3,
            CAST(e.latency_a->>'step4_fetch_details' AS FLOAT) AS a_step4,
            CAST(e.latency_a->>'step5_score_aggregation' AS FLOAT) AS a_step5,
            CAST(e.latency_a->>'step6_final_format' AS FLOAT) AS a_step6,
            CAST(e.latency_a->>'total_search_time' AS FLOAT) AS a_total,
            
            -- LATENCY STEPS (B)
            CAST(e.latency_b->>'step1_embedding' AS FLOAT) AS b_step1,
            CAST(e.latency_b->>'step2_shortlisting' AS FLOAT) AS b_step2,
            CAST(e.latency_b->>'step3_reranking' AS FLOAT) AS b_step3,
            CAST(e.latency_b->>'step4_fetch_details' AS FLOAT) AS b_step4,
            CAST(e.latency_b->>'step5_score_aggregation' AS FLOAT) AS b_step5,
            CAST(e.latency_b->>'step6_final_format' AS FLOAT) AS b_step6,
            CAST(e.latency_b->>'total_search_time' AS FLOAT) AS b_total,

            e.results_identical, e.preference_submitted, e.evaluator_type, e.evaluator_model, e.reasoning,
            CASE 
                WHEN e.preference_submitted = FALSE THEN 'Not Voted'
                WHEN e.winner_model_id IS NULL THEN 'Draw'
                ELSE mw.model_name 
            END as winner,
            e.created_at,
            
            -- BUNDLED EXPERT RESULTS (JSONB)
            (
                SELECT json_agg(json_build_object('rank', er.rank_position, 'name', au.full_name, 'model_id', er.model_id))
                FROM expert_results er
                JOIN authors au ON er.author_id = au.author_id
                WHERE er.evaluation_id = e.evaluation_id
            ) AS experts_list
            
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
    """Renders the HTML table, separated by evaluator type."""
    all_evals = get_denormalized_evaluations()
    
    human_evals = [e for e in all_evals if e['evaluator_type'] == 'human']
    llm_evals = [e for e in all_evals if e['evaluator_type'] == 'llm']
    
    stats = {
        "human_votes": sum(1 for e in human_evals if e['preference_submitted']),
        "llm_votes": sum(1 for e in llm_evals if e['preference_submitted'])
    }
    
    return render_template('admin.html', 
                           human_evals=human_evals, 
                           llm_evals=llm_evals, 
                           stats=stats)

@bp.route('/admin/export')
@requires_auth
def export_csv():
    """Generates and downloads the data as a CSV file with full latency breakdown."""
    evaluations = get_denormalized_evaluations()
    
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Write the header (Extremely detailed for SPSS/ANOVA)
    cw.writerow([
        'Evaluation ID', 'Created At', 'Query', 'Model A', 'Model B', 
        'A_Step1_Embedding', 'A_Step2_Shortlist', 'A_Step3_Rerank', 
        'A_Step4_FetchDB', 'A_Step5_Aggregate', 'A_Step6_Format', 'A_Total',
        'B_Step1_Embedding', 'B_Step2_Shortlist', 'B_Step3_Rerank', 
        'B_Step4_FetchDB', 'B_Step5_Aggregate', 'B_Step6_Format', 'B_Total',
        'Results Identical', 'Evaluator Type', 'Preference Submitted', 'Winner', 'Reasoning'
    ])
    
    def r(val):
        return round(val, 6) if val is not None else ''

    for e in evaluations:
        cw.writerow([
            e['evaluation_id'], e['created_at'], e['query_text'], e['model_a'], e['model_b'],
            r(e['a_step1']), r(e['a_step2']), r(e['a_step3']), r(e['a_step4']), r(e['a_step5']), r(e['a_step6']), r(e['a_total']),
            r(e['b_step1']), r(e['b_step2']), r(e['b_step3']), r(e['b_step4']), r(e['b_step5']), r(e['b_step6']), r(e['b_total']),
            e['results_identical'], e['evaluator_type'], e['preference_submitted'], e['winner'], e['reasoning']
        ])
    
    output = Response(si.getvalue(), mimetype='text/csv')
    output.headers["Content-Disposition"] = "attachment; filename=evaluations_full_metrics.csv"
    return output