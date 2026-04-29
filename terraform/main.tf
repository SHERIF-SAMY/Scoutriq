terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Create a firewall rule to allow traffic to the FastAPI server (Port 8000)
resource "google_compute_firewall" "allow_fastapi" {
  name    = "allow-fastapi-scoutai"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"] # Open to the internet
  target_tags   = ["scoutai-api"]
}

# Create the VM instance
resource "google_compute_instance" "scoutai_vm" {
  name         = "scoutai-api-vm"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["scoutai-api"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30 # Default 10GB might be too small for the models and video processing
    }
  }

  network_interface {
    network = "default"
    access_config {
      # This block gives the VM an external Public IP address
    }
  }
  
  # The startup script runs as root when the Instance boots
  metadata_startup_script = file("${path.module}/startup-script.sh")

}

output "instance_ip" {
  value       = google_compute_instance.scoutai_vm.network_interface[0].access_config[0].nat_ip
  description = "The Public IP of the ScoutAI API server. API is at http://<this_IP>:8000/docs"
}
