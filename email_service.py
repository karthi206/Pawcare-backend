import os
import html
import resend
import logging

resend.api_key = os.environ.get("RESEND_API_KEY")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")
FROM_EMAIL = "PawCare <onboarding@resend.dev>"  # swap once domain verified

logger = logging.getLogger(__name__)

def send_ngo_notification_email(ngo, message: str) -> bool:
    """Email an NGO about a reported case. Returns True if the send was
    attempted successfully, False if skipped (no email on file) or failed."""
    if not ngo.email:
        logger.warning(f"NGO {ngo.id} ({ngo.name}) has no email on file — notification not sent.")
        return False
    try:
        safe_name = html.escape(ngo.name)
        safe_message = html.escape(message)
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [ngo.email],
            "subject": "PawCare: A case near you needs attention",
            "html": f"""
                <p>Hi {safe_name},</p>
                <p>PawCare has a case that may need your attention:</p>
                <p>{safe_message}</p>
                <p>This message was sent by a PawCare admin via the notify feature.</p>
            """
        })
        return True
    except Exception as e:
        logger.error(f"Failed to send NGO notification email to {ngo.email}: {e}")
        return False

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