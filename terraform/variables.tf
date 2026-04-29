variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The GCP region to deploy resources (e.g., us-central1)"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone to deploy the VM (e.g., us-central1-a)"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "The GCP Machine Type (e.g., e2-medium or n1-standard-4 for heavier AI workloads)"
  type        = string
  default     = "e2-standard-4" # Using a stronger machine by default since YOLO/Mediapipe need resources
}
