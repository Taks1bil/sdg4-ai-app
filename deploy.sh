#!/bin/bash

# SDG 4 AI Application Deployment Script
# This script automates the deployment process on Ubuntu servers

set -e  # Exit on any error

echo "🚀 Starting SDG 4 AI Application Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as ubuntu user."
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
print_status "Installing required packages..."
sudo apt install -y python3 python3-pip python3-venv nginx supervisor ufw certbot python3-certbot-nginx

# Create application directory
print_status "Setting up application directory..."
sudo mkdir -p /var/www/sdg4-ai-app
sudo chown ubuntu:ubuntu /var/www/sdg4-ai-app
cd /var/www/sdg4-ai-app

# Copy application files (assuming they're in the current directory)
print_status "Copying application files..."
cp ~/sdg4_ai_app.py .
cp ~/requirements.txt .

# Create virtual environment and install dependencies
print_status "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure supervisor
print_status "Configuring supervisor..."
sudo cp ~/supervisor_config.conf /etc/supervisor/conf.d/sdg4-ai-app.conf
sudo supervisorctl reread
sudo supervisorctl update

# Configure nginx
print_status "Configuring nginx..."
sudo cp ~/nginx_config /etc/nginx/sites-available/sdg4-ai-app

# Prompt for domain name
read -p "Enter your domain name (e.g., example.com): " DOMAIN_NAME

if [ -z "$DOMAIN_NAME" ]; then
    print_warning "No domain provided. Using default configuration for IP access."
    # Modify nginx config for IP access
    sudo sed -i 's/your-domain.com/'"$(curl -s ifconfig.me)"'/g' /etc/nginx/sites-available/sdg4-ai-app
    sudo sed -i '/ssl_certificate/d' /etc/nginx/sites-available/sdg4-ai-app
    sudo sed -i '/ssl_certificate_key/d' /etc/nginx/sites-available/sdg4-ai-app
    sudo sed -i '/listen 443 ssl http2/c\    listen 80;' /etc/nginx/sites-available/sdg4-ai-app
    sudo sed -i '/return 301 https/d' /etc/nginx/sites-available/sdg4-ai-app
else
    # Replace domain in nginx config
    sudo sed -i 's/your-domain.com/'"$DOMAIN_NAME"'/g' /etc/nginx/sites-available/sdg4-ai-app
fi

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/sdg4-ai-app /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
print_status "Testing nginx configuration..."
sudo nginx -t

# Configure firewall
print_status "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'

# Start services
print_status "Starting services..."
sudo systemctl restart nginx
sudo supervisorctl start sdg4-ai-app

# Setup SSL certificate if domain is provided
if [ ! -z "$DOMAIN_NAME" ]; then
    print_status "Setting up SSL certificate..."
    read -p "Enter your email for SSL certificate: " EMAIL
    if [ ! -z "$EMAIL" ]; then
        sudo certbot --nginx -d $DOMAIN_NAME -d www.$DOMAIN_NAME --email $EMAIL --agree-tos --non-interactive
    else
        print_warning "No email provided. Skipping SSL setup. You can run 'sudo certbot --nginx' later."
    fi
fi

# Create log rotation
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/sdg4-ai-app > /dev/null <<EOF
/var/log/sdg4-ai-app.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        supervisorctl restart sdg4-ai-app
    endscript
}
EOF

# Create monitoring script
print_status "Creating monitoring script..."
cat > /home/ubuntu/monitor_app.sh << 'EOF'
#!/bin/bash
# Simple monitoring script for SDG 4 AI Application

APP_URL="http://localhost:8501"
LOG_FILE="/var/log/sdg4-ai-app-monitor.log"

# Check if application is responding
if curl -f -s $APP_URL > /dev/null; then
    echo "$(date): Application is running normally" >> $LOG_FILE
else
    echo "$(date): Application is not responding, restarting..." >> $LOG_FILE
    sudo supervisorctl restart sdg4-ai-app
fi
EOF

chmod +x /home/ubuntu/monitor_app.sh

# Add monitoring to crontab
print_status "Setting up monitoring cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/ubuntu/monitor_app.sh") | crontab -

# Final status check
print_status "Checking application status..."
sleep 5
if sudo supervisorctl status sdg4-ai-app | grep -q RUNNING; then
    print_status "✅ Application is running successfully!"
else
    print_error "❌ Application failed to start. Check logs with: sudo supervisorctl tail sdg4-ai-app"
    exit 1
fi

# Display access information
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Access Information:"
if [ ! -z "$DOMAIN_NAME" ]; then
    echo "   🌐 URL: https://$DOMAIN_NAME"
    echo "   🌐 Alternative: http://$DOMAIN_NAME"
else
    SERVER_IP=$(curl -s ifconfig.me)
    echo "   🌐 URL: http://$SERVER_IP"
fi
echo ""
echo "📊 Management Commands:"
echo "   • Check status: sudo supervisorctl status sdg4-ai-app"
echo "   • View logs: sudo supervisorctl tail sdg4-ai-app"
echo "   • Restart app: sudo supervisorctl restart sdg4-ai-app"
echo "   • Nginx status: sudo systemctl status nginx"
echo ""
echo "📁 Important Paths:"
echo "   • Application: /var/www/sdg4-ai-app/"
echo "   • Logs: /var/log/sdg4-ai-app.log"
echo "   • Nginx config: /etc/nginx/sites-available/sdg4-ai-app"
echo ""
print_status "Your SDG 4 AI Application is now live! 🚀"

