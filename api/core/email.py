import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.config import EMAIL_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

def send_verification_email(email: str, first_name: str, verification_code: str) -> bool:
    """
    Send a verification email to the patient
    
    Args:
        email: Patient's email address
        first_name: Patient's first name
        verification_code: Verification code to include in the email
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        message = MIMEMultipart()
        message['From'] = EMAIL_CONFIG['from_email']
        message['To'] = email
        message['Subject'] = "Verify Your Power of Patient Account"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>Hello {first_name},</h2>
            <p>Thank you for joining Power of Patient. Please use the following verification code to confirm your identity:</p>
            <div style="background-color: #f2f2f2; padding: 15px; text-align: center; font-size: 24px; letter-spacing: 5px;">
                <strong>{verification_code}</strong>
            </div>
            <p>This code will expire in {EMAIL_CONFIG['verification_expiry_hours']} hours.</p>
            <p>If you did not request this verification, please ignore this email.</p>
            <p>Best regards,<br>The Power of Patient Team</p>
        </body>
        </html>
        """
        
        message.attach(MIMEText(body, 'html'))
        
        # Send the email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['smtp_user'], EMAIL_CONFIG['smtp_password'])
        server.send_message(message)
        server.quit()
        
        logger.info(f"Verification email sent to {email}")
        return True
        
    except Exception as e:
        logger.exception(f"Error sending verification email: {str(e)}")
        return False

def send_welcome_email(email: str, first_name: str) -> bool:
    """
    Send a welcome email to a newly verified patient
    
    Args:
        email: Patient's email address
        first_name: Patient's first name
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        message = MIMEMultipart()
        message['From'] = EMAIL_CONFIG['from_email']
        message['To'] = email
        message['Subject'] = "Welcome to Power of Patient!"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>Welcome to Power of Patient, {first_name}!</h2>
            <p>Your account has been successfully verified, and you now have full access to all our features.</p>
            <p>With Power of Patient, you can:</p>
            <ul>
                <li>Track your TBI symptoms and recovery progress</li>
                <li>Chat with Sallie, your personal TBI assistant</li>
                <li>Access personalized resources for your recovery journey</li>
                <li>Connect with healthcare providers</li>
            </ul>
            <p>If you have any questions, please don't hesitate to contact our support team.</p>
            <p>Best regards,<br>The Power of Patient Team</p>
        </body>
        </html>
        """
        
        message.attach(MIMEText(body, 'html'))
        
        # Send the email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['smtp_user'], EMAIL_CONFIG['smtp_password'])
        server.send_message(message)
        server.quit()
        
        logger.info(f"Welcome email sent to {email}")
        return True
        
    except Exception as e:
        logger.exception(f"Error sending welcome email: {str(e)}")
        return False