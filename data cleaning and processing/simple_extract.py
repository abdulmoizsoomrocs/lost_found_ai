import mailbox
import pandas as pd
from bs4 import BeautifulSoup

# 👉 Your actual file path
mbox_path = r"C:\Users\DELL\OneDrive - Higher Education Commission\lost and found dataset\dataset.mbox"

mbox = mailbox.mbox(mbox_path)

data = []

for message in mbox:
    subject = message['subject'] or ""
    body = ""

    try:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True)
                    if body:
                        body = body.decode(errors='ignore')
                        break
                elif part.get_content_type() == "text/html":
                    html = part.get_payload(decode=True)
                    if html:
                        soup = BeautifulSoup(html, "html.parser")
                        body = soup.get_text()
                        break
        else:
            payload = message.get_payload(decode=True)
            if payload:
                body = payload.decode(errors='ignore')

        if subject.strip() or body.strip():
            data.append({
                "subject": subject,
                "body": body[:500]
            })

    except:
        continue

df = pd.DataFrame(data)

df.to_csv("emails_output.csv", index=False)

print(f"✅ Extracted {len(data)} emails")