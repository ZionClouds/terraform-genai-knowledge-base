/**
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

module "project-services" {
  # source = "https://github.com/terraform-google-modules/terraform-google-project-factory/tree/master/modules/project-services"
  # source  = "terraform-google-modules/project-factory/google//modules/project-services"
  # source  = "terraform-google-modules/project-factory/google//modules/project_services"
  source = "./modules/project_services"
  # version = "15.0.1"

  project_id                  = var.project_id
  disable_services_on_destroy = var.disable_services_on_destroy

  activate_apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "config.googleapis.com",
    "documentai.googleapis.com",
    "eventarc.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "serviceusage.googleapis.com",
    "storage-api.googleapis.com",
    "storage.googleapis.com",
  ]
}

# Add a delay of 30 seconds after API activation
# resource "time_sleep" "wait_for_apis" {
#   depends_on = [module.project-services]
#   create_duration = "10s"
# }


resource "random_id" "unique_id" {
  byte_length = 3
}

locals {
  bucket_main_name         = var.unique_names ? "zionai-kb-bucket-${var.project_id}-${random_id.unique_id.hex}" : "knowledge-base-bucket-${var.project_id}"
  bucket_docs_name         = var.unique_names ? "zionai-kb-docs-${var.project_id}-${random_id.unique_id.hex}" : "knowledge-base-docs-${var.project_id}"
  webhook_name             = var.unique_names ? "zionai-kb-webhook-${random_id.unique_id.hex}" : "knowledge-base-webhook"
  webhook_sa_name          = var.unique_names ? "zionai-kb-webhook-sa-${random_id.unique_id.hex}" : "knowledge-base-webhook-sa"
  trigger_name             = var.unique_names ? "zionai-kb-trigger-${random_id.unique_id.hex}" : "knowledge-base-trigger"
  trigger_sa_name          = var.unique_names ? "zionai-kb-trigger-sa-${random_id.unique_id.hex}" : "knowledge-base-trigger-sa"
  artifact_repo_name       = var.unique_names ? "zionai-kb-repo-${random_id.unique_id.hex}" : "knowledge-base-repo"
  ocr_processor_name       = var.unique_names ? "zionai-kb-ocr-processor-${random_id.unique_id.hex}" : "knowledge-base-ocr-processor"
  docs_index_name          = var.unique_names ? "zionai-kb-index-${random_id.unique_id.hex}" : "knowledge-base-index"
  docs_index_endpoint_name = var.unique_names ? "zionai-kb-index-endpoint-${random_id.unique_id.hex}" : "knowledge-base-index-endpoint"
  firestore_name           = var.unique_names ? "zionai-kb-database-${random_id.unique_id.hex}" : "knowledge-base-database"
}

#-- Cloud Storage buckets --#
resource "google_storage_bucket" "main" {
  project                     = module.project-services.project_id
  name                        = local.bucket_main_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
  labels                      = var.labels
  #depends_on = [time_sleep.wait_for_apis]
}

resource "google_storage_bucket" "docs" {
  project                     = module.project-services.project_id
  name                        = local.bucket_docs_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
  labels                      = var.labels
  #depends_on = [time_sleep.wait_for_apis]
}

#-- Cloud Function webhook --#
resource "google_cloudfunctions2_function" "webhook" {
  project  = module.project-services.project_id
  name     = local.webhook_name
  location = var.region
  labels   = var.labels
  #depends_on = [time_sleep.wait_for_apis]

  build_config {
    runtime           = "python312"
    entry_point       = "on_cloud_event"
    docker_repository = google_artifact_registry_repository.webhook_images.id
    source {
      storage_source {
        bucket = google_storage_bucket.main.name
        object = google_storage_bucket_object.webhook_staging.name
      }
    }
  }

  service_config {
    available_memory      = "4G"
    timeout_seconds       = 300 # 5 minutes
    service_account_email = google_service_account.webhook.email
    environment_variables = {
      PROJECT_ID        = module.project-services.project_id
      VERTEXAI_LOCATION = var.region
      OUTPUT_BUCKET     = google_storage_bucket.main.name
      DOCAI_PROCESSOR   = google_document_ai_processor.ocr.id
      DOCAI_LOCATION    = google_document_ai_processor.ocr.location
      DATABASE          = google_firestore_database.main.name
      INDEX_ID          = google_vertex_ai_index.docs.id
      LOG_EXECUTION_ID  = true
    }
  }
}

resource "google_project_iam_member" "webhook" {
  project = module.project-services.project_id
  member  = google_service_account.webhook.member
  for_each = toset([
    "roles/aiplatform.serviceAgent", # https://cloud.google.com/iam/docs/service-agents
    "roles/datastore.user",          # https://cloud.google.com/datastore/docs/access/iam
    "roles/documentai.apiUser",      # https://cloud.google.com/document-ai/docs/access-control/iam-roles
  ])
  role = each.key
  #depends_on = [time_sleep.wait_for_apis]
}
resource "google_service_account" "webhook" {
  project      = module.project-services.project_id
  account_id   = local.webhook_sa_name
  display_name = "Cloud Functions webhook service account"
  #depends_on = [time_sleep.wait_for_apis]
}

resource "google_artifact_registry_repository" "webhook_images" {
  project       = module.project-services.project_id
  location      = var.region
  repository_id = local.artifact_repo_name
  format        = "DOCKER"
  labels        = var.labels
  #depends_on = [time_sleep.wait_for_apis]
}

data "archive_file" "webhook_staging" {
  type        = "zip"
  source_dir  = "${path.module}/webhook"
  output_path = "${path.module}/.terraform/webhook.zip"
  excludes = [
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "env",
  ]
  #depends_on = [time_sleep.wait_for_apis]
}

resource "google_storage_bucket_object" "webhook_staging" {
  name   = "webhook-staging/${data.archive_file.webhook_staging.output_base64sha256}.zip"
  bucket = google_storage_bucket.main.name
  source = data.archive_file.webhook_staging.output_path
  #depends_on = [time_sleep.wait_for_apis]
}

#-- Eventarc trigger --#
resource "google_eventarc_trigger" "trigger" {
  project         = module.project-services.project_id
  location        = var.region
  name            = local.trigger_name
  service_account = google_service_account.trigger.email
  labels          = var.labels

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }
  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.docs.name
  }

  destination {
    cloud_run_service {
      service = google_cloudfunctions2_function.webhook.name
      region  = var.region
    }
  }
  #depends_on = [time_sleep.wait_for_apis]
}

resource "google_project_iam_member" "trigger" {
  project = module.project-services.project_id
  member  = google_service_account.trigger.member
  for_each = toset([
    "roles/eventarc.eventReceiver", # https://cloud.google.com/eventarc/docs/access-control
    "roles/run.invoker",            # https://cloud.google.com/run/docs/reference/iam/roles
  ])
  role = each.key
  #depends_on = [time_sleep.wait_for_apis]
}
resource "google_service_account" "trigger" {
  project      = module.project-services.project_id
  account_id   = local.trigger_sa_name
  display_name = "Eventarc trigger service account"
  #depends_on = [time_sleep.wait_for_apis]
}

#-- Cloud Storage Eventarc agent --#
resource "google_project_iam_member" "gcs_account" {
  project = module.project-services.project_id
  member  = "serviceAccount:${data.google_storage_project_service_account.gcs_account.email_address}"
  role    = "roles/pubsub.publisher" # https://cloud.google.com/pubsub/docs/access-control
  #depends_on = [time_sleep.wait_for_apis]
}
data "google_storage_project_service_account" "gcs_account" {
  project = module.project-services.project_id
  #depends_on = [time_sleep.wait_for_apis]
}

resource "google_project_iam_member" "eventarc_agent" {
  project = module.project-services.project_id
  member  = "serviceAccount:${google_project_service_identity.eventarc_agent.email}"
  for_each = toset([
    "roles/eventarc.serviceAgent",             # https://cloud.google.com/iam/docs/service-agents
    "roles/serviceusage.serviceUsageConsumer", # https://cloud.google.com/service-usage/docs/access-control
  ])
  role = each.key
  #depends_on = [time_sleep.wait_for_apis]
}
resource "google_project_service_identity" "eventarc_agent" {
  provider = google-beta
  project  = module.project-services.project_id
  service  = "eventarc.googleapis.com"
  #depends_on = [time_sleep.wait_for_apis]
}

#-- Document AI --#
resource "google_document_ai_processor" "ocr" {
  project      = module.project-services.project_id
  location     = var.documentai_location
  display_name = local.ocr_processor_name
  type         = "OCR_PROCESSOR"
  #depends_on = [time_sleep.wait_for_apis]
}

#-- Firestore --#
resource "google_firestore_database" "main" {
  project         = module.project-services.project_id
  name            = local.firestore_name
  location_id     = var.firestore_location
  type            = "FIRESTORE_NATIVE"
  deletion_policy = "DELETE"
  #depends_on = [time_sleep.wait_for_apis]
}

#-- Vertex AI Vector Search --#
resource "google_vertex_ai_index" "docs" {
  project             = module.project-services.project_id
  region              = var.region
  display_name        = local.docs_index_name
  index_update_method = "STREAM_UPDATE"
  labels              = var.labels
  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.main.name}/vector-search-index"
    config {
      dimensions                  = 768
      approximate_neighbors_count = 150
      shard_size                  = "SHARD_SIZE_SMALL"
      distance_measure_type       = "SQUARED_L2_DISTANCE"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 1000
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }
  depends_on = [google_storage_bucket_object.index_initial]
}

resource "google_vertex_ai_index_endpoint" "docs" {
  project                 = module.project-services.project_id
  region                  = var.region
  display_name            = local.docs_index_endpoint_name
  public_endpoint_enabled = true
  labels                  = var.labels
  #depends_on = [time_sleep.wait_for_apis]
}

# # Deploy the Index to the Endpoint
# resource "google_vertex_ai_deployed_index" "docs" {
#   project        = module.project-services.project_id
#   region         = var.region
#   index_endpoint = google_vertex_ai_index_endpoint.docs.id
#   display_name   = "deployed-index-${local.docs_index_name}"
#   index          = google_vertex_ai_index.docs.id
#   labels         = var.labels

#   dedicated_resources {
#     machine_spec {
#       machine_type = "n1-standard-4"  # Adjust based on your workload
#     }
#     min_replica_count = 1  # Adjust replicas based on traffic
#   }

#   depends_on = [google_vertex_ai_index.docs, google_vertex_ai_index_endpoint.docs]
# }

resource "google_storage_bucket_object" "index_initial" {
  bucket = google_storage_bucket.main.name
  name   = "vector-search-index/initial-index.json"
  source = "${path.module}/initial-index.json"
  #depends_on = [time_sleep.wait_for_apis]
}
