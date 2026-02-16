import smtplib
from email.message import EmailMessage
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import json
from typing import Optional, List, Tuple
from flask import current_app


def send_email(to: str, subject: str, body: str, html: bool = False, attachments: Optional[List[Tuple[str, bytes, str]]] = None) -> bool:
    """Send an email using simple SMTP settings from config.
    Returns True on success, False on failure. Falls back to printing to stdout.
    """
    cfg = current_app.config if current_app else {}
    server = cfg.get("MAIL_SERVER")
    port = int(cfg.get("MAIL_PORT", 25))
    username = cfg.get("MAIL_USERNAME")
    password = cfg.get("MAIL_PASSWORD")
    use_tls = bool(cfg.get("MAIL_USE_TLS"))
    use_ssl = bool(cfg.get("MAIL_USE_SSL"))
    sender = cfg.get("MAIL_DEFAULT_SENDER", "noreply@local")

    if not server:
        print(f"[EMAIL:DRYRUN] to={to} subject={subject}\n{body}\n")
        return True

    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        if html:
            msg.add_alternative(body, subtype="html")
        else:
            msg.set_content(body)

        # Attach files if provided
        if attachments:
            for filename, content, mime in attachments:
                maintype, _, subtype = (mime or "application/octet-stream").partition("/")
                msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)

        if use_ssl:
            with smtplib.SMTP_SSL(server, port) as smtp:
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(server, port) as smtp:
                if use_tls:
                    smtp.starttls()
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"[EMAIL:ERROR] {e}")
        return False


def send_chat_message(text: str) -> bool:
    """Send a simple text payload to Google Chat via incoming webhook.
    Returns True on success, False on failure; prints on failure.
    """
    cfg = current_app.config if current_app else {}
    url: Optional[str] = cfg.get("CHAT_WEBHOOK_URL")
    if not url:
        print(f"[CHAT:DRYRUN] {text}")
        return True
    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as _:
            return True
    except (HTTPError, URLError) as e:
        print(f"[CHAT:ERROR] {e}")
        return False


def notify_user(email: Optional[str], subject: str, body: str, html: bool = False, attachments: Optional[List[Tuple[str, bytes, str]]] = None) -> None:
    """Try email first; if not configured or fails, try chat webhook; else fallback print."""
    ok = False
    if email:
        ok = send_email(email, subject, body, html=html, attachments=attachments)
    if not ok:
        send_chat_message(f"{subject}: {body}")
