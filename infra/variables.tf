variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for all resources (keep short: ALB/target group names max 32 chars)"
  type        = string
  default     = "churn-api"
}

variable "image_tag" {
  description = "ECR image tag the ECS service runs"
  type        = string
  default     = "latest"
}

variable "desired_count" {
  description = "Baseline number of running tasks (also the autoscaling minimum)"
  type        = number
  default     = 2
}

variable "max_count" {
  description = "Autoscaling maximum number of tasks"
  type        = number
  default     = 4
}
