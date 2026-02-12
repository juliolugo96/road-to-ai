# From Elastic Beanstalk to EC2: Automation Files

**These files are whitelabel versions of the actual scripts used in production.** They have been sanitized and generalized for educational purposes. Use with caution and **test thoroughly before using in any production environment**.

## Disclaimer

- All organization-specific references, credentials, and identifiers have been removed
- Placeholder values (e.g., `my-deployments`, `myapp`, `my-app-asg`) must be replaced with your actual configuration
- These scripts were designed for Amazon Linux 2023 — adjustments may be needed for other operating systems
- Always review and understand each script before executing it on real infrastructure

## Structure

```
from-elastic-beanstalk-to-ec2/
├── setup-scripts/           # Modular, idempotent instance setup (run in order)
│   ├── 01-env.sh            # Load env vars from AWS Secrets Manager
│   ├── 02-packages.sh       # Install system packages
│   ├── 03-redis.sh          # Redis via Docker
│   ├── 04-nodejs.sh         # Node.js + Yarn for asset pipeline
│   ├── 05-ruby.sh           # Ruby + OpenSSL 1.1 (compiled from source)
│   ├── 06-pdf-tools.sh      # wkhtmltopdf for PDF generation
│   ├── 07-directories.sh    # Application directory layout
│   ├── 08-deploy.sh         # Download from S3 + atomic symlink deploy
│   └── 09-monitoring.sh     # Start Grafana/Loki/Promtail stack
├── systemd/                 # Service definitions
│   ├── puma.service          # Rails app server
│   └── sidekiq.service       # Background job processor
├── nginx/                   # Reverse proxy configuration
│   └── app.conf              # Puma upstream + static assets + security headers
├── monitoring/              # Observability stack
│   ├── docker-compose.yml    # Grafana + Loki + Promtail
│   ├── loki-config.yml       # Log storage and retention (31 days)
│   ├── promtail-config.yml   # Log scraping targets
│   └── grafana-provisioning/ # Auto-configured Loki datasource
│       └── datasources/
│           └── loki.yml
├── deployment/              # CI/CD and deployment orchestration
│   ├── build-and-upload.sh   # Build tarball + upload to S3
│   ├── trigger-deploy.sh     # Start ASG instance refresh (blue/green)
│   ├── rollback.sh           # Atomic rollback to previous release
│   └── bootstrap.sh          # EC2 user-data entry point
└── .github/workflows/
    └── deploy.yml            # GitHub Actions: test → build → deploy
```

## How It Works

1. **CI/CD** (`deploy.yml`): On push to `main`, tests run, then the app is packaged and uploaded to S3
2. **ASG Instance Refresh** (`trigger-deploy.sh`): New instances launch with fresh code; old ones drain gracefully
3. **Bootstrap** (`bootstrap.sh`): Each new EC2 instance runs the 9 setup scripts in order
4. **Atomic Deploy** (`08-deploy.sh`): Code extracted to timestamped dir, symlink switched atomically
5. **Monitoring** (`09-monitoring.sh`): Grafana/Loki/Promtail start collecting logs immediately

## Before You Use These

- [ ] Replace all placeholder values (`my-deployments`, `myapp`, `my-app-asg`, etc.)
- [ ] Set up AWS Secrets Manager with your application's environment variables
- [ ] Create the S3 bucket for deployment artifacts
- [ ] Configure IAM roles for EC2 instances (S3 read, Secrets Manager read, ASG permissions)
- [ ] Set up the ALB target group with health check pointing to `/health`
- [ ] Test the full bootstrap flow on a throwaway EC2 instance first
- [ ] Review security group rules and network configuration
