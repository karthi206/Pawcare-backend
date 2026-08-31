import os
import resend
import logging

resend.api_key = os.environ.get("RESEND_API_KEY")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")
FROM_EMAIL = "PawCare <onboarding@resend.dev>"  # swap once domain verified

logger = logging.getLogger(__name__)

def send_vet_registration_email(vet):
    try:
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [ADMIN_EMAIL],
            "subject": "New Vet Registration Pending Verification",
            "html": f"""
                <p>A new vet has registered and needs verification:</p>
                <ul>
                    <li>Name: {vet.username}</li>
                    <li>Email: {vet.email}</li>
                    <li>License #: {vet.license_number}</li>
                    <li>Clinic: {vet.clinic_name}</li>
                </ul>
                <p>Review in the admin panel.</p>
            """
        })
    except Exception as e:
        logger.error(f"Failed to send vet registration email: {e}")


def send_vet_decision_email(vet, approved: bool):
    try:
        status_text = "approved" if approved else "rejected"
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [vet.email],
            "subject": f"Your PawCare Vet Verification was {status_text.capitalize()}",
            "html": f"""
                <p>Hi {vet.username},</p>
                <p>Your vet verification request has been <strong>{status_text}</strong>.</p>
                {"<p>You can now log in and access vet features.</p>" if approved else "<p>If you believe this is a mistake, please contact support.</p>"}
            """
        })
    except Exception as e:
        logger.error(f"Failed to send vet decision email: {e}")