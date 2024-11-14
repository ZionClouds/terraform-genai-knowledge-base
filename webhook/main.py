# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools, json, logging, multiprocessing, os, re
from datetime import datetime
from google.cloud import aiplatform, documentai, firestore, storage
from google.api_core.client_options import ClientOptions
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from cloudevents.http import CloudEvent
from collections.abc import Iterator
from retry import retry
import functions_framework

# Global constants and initializations
DOCAI_LOCATION = os.getenv("DOCAI_LOCATION", "us")
vertexai.init(location=os.getenv("VERTEXAI_LOCATION", "us-central1"))
aiplatform.init(location=os.getenv("VERTEXAI_LOCATION", "us-central1"))

@functions_framework.cloud_event
def on_cloud_event(event: CloudEvent) -> None:
    """Cloud Function triggered by an Eventarc event to process new document uploads."""
    try:
        process_document({
            "event_id": event.data["id"],
            "input_bucket": event.data["bucket"],
            "filename": event.data["name"],
            "mime_type": event.data["contentType"],
            "time_uploaded": datetime.fromisoformat(event.data["timeCreated"]),
            "docai_processor_id": os.environ["DOCAI_PROCESSOR"],
            "database": os.environ["DATABASE"],
            "output_bucket": os.environ["OUTPUT_BUCKET"],
            "index_id": os.environ["INDEX_ID"],
        })
    except Exception as e:
        logging.exception(e, stack_info=True)

def process_document(args: dict) -> None:
    """Process a new document by extracting, indexing, and generating Q&A data.

    Args:
        args: Dictionary containing the parameters for document processing.
    """
    # Initialize Firestore database and prepare data for processing
    db = firestore.Client(database=args["database"])
    doc = db.document("documents", args["filename"].replace("/", "-"))
    event_data, docai_config = {k: args[k] for k in ("event_id", "input_bucket", "filename", "mime_type", "time_uploaded")}, {k: args[k] for k in ("input_bucket", "filename", "mime_type", "docai_processor_id", "output_bucket")}

    # Check if event is already processed
    if is_event_processed(doc, args["event_id"]): return
    doc.create(event_data) if not doc.get().exists else doc.update(event_data)

    # Get document text and update Firestore
    pages = list(get_document_text_pages(docai_config))
    doc.update({"pages": pages})

    # Index pages and generate Q&As, then save and write dataset
    index_pages(args["index_id"], args["filename"], pages)
    save_qa_entries(db, generate_qa_for_pages(args["filename"], pages))
    write_tuning_dataset(db, args["output_bucket"])

def is_event_processed(doc, event_id: str) -> bool:
    """Check if an event has already been processed by examining Firestore entries."""
    entry = doc.get().to_dict() or {}
    return (entry.get("event_id") == event_id)

def get_document_text_pages(docai_config: dict) -> list:
    """Retrieve OCR-processed text pages of the document using Document AI.

    Args:
        docai_config: Dictionary with Document AI configurations.

    Returns:
        A list containing text content of each page.
    """
    uri = f"gs://{docai_config['input_bucket']}/{docai_config['filename']}"
    return list(get_document_text(uri, docai_config["mime_type"], docai_config["docai_processor_id"], docai_config["output_bucket"]))

def generate_qa_for_pages(filename: str, pages: list) -> list:
    """Generate Q&A entries for each page of a document.

    Args:
        filename: Document file name.
        pages: List of document pages.

    Returns:
        A list of dictionaries, each containing questions and answers.
    """
    with multiprocessing.Pool(len(pages)) as pool:
        return list(itertools.chain.from_iterable(pool.map(process_page, [{"filename": filename, "page_number": i, "text": page} for i, page in enumerate(pages)])))

def save_qa_entries(db, entries: list):
    """Save generated Q&A entries to Firestore.

    Args:
        db: Firestore client instance.
        entries: List of Q&A dictionaries.
    """
    for entry in entries:
        doc = db.document("dataset", entry["question"].replace("/", " "))
        doc.create(entry) if not doc.get().exists else doc.update(entry)

def process_page(event_page: dict) -> list[dict[str, str]]:
    """Generate Q&A for a single document page using AI-based question generation.

    Args:
        event_page: Dictionary with page details and text.

    Returns:
        A list of Q&A dictionaries for the page.
    """
    entries = generate_questions(event_page["text"])
    return [{"question": e["question"], "answer": e["answer"], "filename": event_page["filename"], "page_number": event_page["page_number"]} for e in entries] if entries else []

def get_document_text(input_file: str, mime_type: str, processor_id: str, temp_bucket: str) -> Iterator[str]:
    """Perform OCR on a document using Document AI and retrieve text by page.

    Args:
        input_file: Document file URI in GCS.
        mime_type: Document MIME type.
        processor_id: Document AI processor ID.
        temp_bucket: Temporary storage bucket for OCR output.

    Returns:
        An iterator over the text content of each page.
    """
    # Initialize Document AI client and perform batch processing
    client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=f"{DOCAI_LOCATION}-documentai.googleapis.com"))
    operation = client.batch_process_documents(request=documentai.BatchProcessRequest(name=processor_id, input_documents=documentai.BatchDocumentsInputConfig(gcs_documents=documentai.GcsDocuments(documents=[documentai.GcsDocument(gcs_uri=input_file, mime_type=mime_type)])), document_output_config=documentai.DocumentOutputConfig(gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=f"gs://{temp_bucket}/ocr/{input_file.split('gs://')[-1]}"))))
    operation.result()

    # Retrieve processed text from Cloud Storage
    output_bucket, output_prefix = documentai.BatchProcessMetadata(operation.metadata).individual_process_statuses[0].output_gcs_destination.removeprefix("gs://").split("/", 1)
    for blob in storage.Client().list_blobs(output_bucket, prefix=output_prefix):
        for page in documentai.Document.from_json(blob.download_as_bytes(), ignore_unknown_fields=True).pages:
            yield "\n".join([documentai.Document.from_json(blob.download_as_bytes(), ignore_unknown_fields=True).text[s:e] for s, e in [(seg.start_index, seg.end_index) for seg in page.layout.text_anchor.text_segments]])

def index_pages(index_id: str, filename: str, pages: list[str]) -> None:
    """Index document pages in Vertex AI for search.

    Args:
        index_id: Vertex AI index ID.
        filename: Document file name.
        pages: List of document text pages.
    """
    model, points = TextEmbeddingModel.from_pretrained("textembedding-gecko@003"), [IndexDatapoint(datapoint_id=f"{filename}:{i}", feature_vector=embedding.values) for i, embedding in enumerate([v for b in itertools.batched(pages, 5) for v in model.get_embeddings(b)])]
    index = aiplatform.MatchingEngineIndex(index_id)
    index.remove_datapoints(["null"])
    index.upsert_datapoints(points).wait()

@retry(tries=3)
def generate_questions(text: str) -> list[dict[str, str]]:
    """Generate a list of questions and answers from text using a large language model.

    Args:
        text: Text content for Q&A generation.

    Returns:
        A list of dictionaries, each containing a question and an answer.
    """
    model = GenerativeModel(model_name="gemini-1.0-pro", system_instruction=[ 'Respond with a JSON list of {"question", "answer"} objects.', "Use simple language and words that are easy to understand.", "Avoid technical terms in the answers.", f"TEXT: {text}"])
    response = model.generate_content("Give me 20 self-contained questions and answers that can be answered from the text").text
    start = response.find("```")
    return json.loads("\n".join(response[start:].splitlines()[1:-1])) if start != -1 else json.loads(response)

def write_tuning_dataset(db: firestore.Client, output_bucket: str) -> int:
    """Write tuning dataset to Cloud Storage for model training.

    Args:
        db: Firestore client.
        output_bucket: GCS bucket for saving the tuning dataset.

    Returns:
        The size of the written dataset.
    """
    storage_client = storage.Client()
    documents = [doc.to_dict() or {} for doc in db.collection("documents").stream()]
    doc_pages = {doc["filename"]: doc["pages"] for doc in documents}
    size = 0
    with storage_client.get_bucket(output_bucket).blob("dataset.jsonl").open("w") as f:
        for doc in db.collection("dataset").stream():
            entry = doc.to_dict() or {}
            document_page_text = doc_pages[entry["filename"]][entry["page_number"]]
            messages = {
                "messages": [
                    {"role": "system", "content": document_page_text},
                    {"role": "user", "content": entry["question"]},
                    {"role": "model", "content": entry["answer"]},
                ]
            }
            f.write(f"{json.dumps(messages)}\n")
            size += 1
    return size
