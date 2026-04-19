import os
import datetime
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from google.cloud import firestore
except ImportError:
    firestore = None

app = Flask(__name__)
CORS(app)

# Initialize Firestore DB
# This relies on the GOOGLE_APPLICATION_CREDENTIALS environment variable
# or default service account permissions when deployed to Google Cloud.
if firestore is None:
    print("Firestore library is not installed. Using in-memory storage.")
    db = None
else:
    try:
        if os.path.exists("service-account.json"):
            db = firestore.Client.from_service_account_json("service-account.json")
            print("Firestore Client initialized using service-account.json")
        else:
            db = firestore.Client()
            print("Firestore Client initialized using default credentials")
    except Exception as e:
        print(f"Warning: Failed to initialize Firestore client: {e}")
        db = None

PLANT_HISTORY_COLLECTION = "plant_history"
DEFAULT_EVENTS_LIMIT = 25
MAX_EVENTS_LIMIT = 100

_memory_store_lock = Lock()
_memory_events: List[Dict[str, Any]] = []


def _coerce_sprayed(payload: Dict[str, Any]) -> Optional[bool]:
    sprayed_raw = payload.get("sprayed")
    if isinstance(sprayed_raw, bool):
        return sprayed_raw

    history_type = payload.get("history_type")
    if isinstance(history_type, str):
        normalized = history_type.strip().lower()
        if normalized == "sprayed":
            return True
        if normalized == "scan":
            return False

    return None


def _serialize_event(event_id: str, record: Dict[str, Any], timestamp_iso: str) -> Dict[str, Any]:
    return {
        "id": event_id,
        "disease_name": record["disease_name"],
        "sprayed": record["sprayed"],
        "timestamp": timestamp_iso,
    }


def _save_event(record: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_iso = now.isoformat()

    if db is None:
        event_id = uuid4().hex
        event = _serialize_event(event_id, record, timestamp_iso)
        with _memory_store_lock:
            _memory_events.append(event)
        return event

    firestore_record = {
        "disease_name": record["disease_name"],
        "sprayed": record["sprayed"],
        "timestamp": now,
    }

    doc_ref = db.collection(PLANT_HISTORY_COLLECTION).document()
    doc_ref.set(firestore_record)
    return _serialize_event(doc_ref.id, record, timestamp_iso)


def _parse_limit(limit_raw: str | None) -> int:
    if limit_raw is None:
        return DEFAULT_EVENTS_LIMIT

    try:
        parsed_limit = int(limit_raw)
    except ValueError:
        return DEFAULT_EVENTS_LIMIT

    if parsed_limit < 1:
        return 1
    if parsed_limit > MAX_EVENTS_LIMIT:
        return MAX_EVENTS_LIMIT
    return parsed_limit


def _fetch_events(limit: int) -> List[Dict[str, Any]]:
    if db is None:
        with _memory_store_lock:
            return list(reversed(_memory_events[-limit:]))

    docs = (
        db.collection(PLANT_HISTORY_COLLECTION)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )

    events: List[Dict[str, Any]] = []
    for doc in docs:
        data = doc.to_dict()
        ts = data.get("timestamp")
        if isinstance(ts, datetime.datetime):
            timestamp_iso = ts.astimezone(datetime.timezone.utc).isoformat()
        else:
            timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        events.append(
            {
                "id": doc.id,
                "disease_name": data.get("disease_name", "Unknown"),
                "sprayed": bool(data.get("sprayed", False)),
                "timestamp": timestamp_iso,
            }
        )

    return events


@app.route('/api/log_disease', methods=['POST'])
@app.route('/api/disease-events', methods=['POST'])
def log_disease():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "No JSON payload provided"}), 400

    disease_name_raw = data.get("disease_name")
    sprayed = _coerce_sprayed(data)

    if not isinstance(disease_name_raw, str) or not disease_name_raw.strip():
        return jsonify({"error": "Missing or invalid 'disease_name' in payload"}), 400

    if sprayed is None:
        return jsonify({"error": "Missing or invalid 'sprayed' boolean in payload"}), 400

    record = {
        "disease_name": disease_name_raw.strip(),
        "sprayed": sprayed,
    }

    try:
        event = _save_event(record)
        return jsonify({
            "message": "Record saved successfully",
            "event": event,
            "id": event["id"],
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/disease-events', methods=['GET'])
def list_disease_events():
    limit = _parse_limit(request.args.get("limit"))
    try:
        events = _fetch_events(limit)
        return jsonify({"events": events, "count": len(events)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "storage_mode": "firestore" if db is not None else "memory",
    }), 200

if __name__ == '__main__':
    # Run locally (usually on port 8080 as per Google Cloud run default)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
