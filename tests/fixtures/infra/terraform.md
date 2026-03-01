# Terraform Infrastructure as Code

## Provider Configuration

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}
```

## Module Structure

Organize Terraform code into reusable modules:

```
modules/
├── vpc/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── ecs-service/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── rds/
    ├── main.tf
    ├── variables.tf
    └── outputs.tf
```

## State Management

- Always use remote state backend (S3, GCS, Terraform Cloud)
- Enable state locking with DynamoDB
- Never manually edit state files
- Use `terraform import` for existing resources

```bash
terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan
```

## Common Patterns

```hcl
resource "aws_instance" "web" {
  count         = var.instance_count
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type

  tags = merge(var.common_tags, {
    Name = "web-${count.index + 1}"
  })
}

output "instance_ips" {
  value = aws_instance.web[*].public_ip
}
```
