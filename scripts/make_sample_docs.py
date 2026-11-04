"""Make sample (made-up) documents in fda/docs for trying "ask_documents": a billing policy (Markdown), an office
handbook (Word), an inventory (Excel), contacts (CSV) and safety rules (PDF, 2 pages). All content is fictional.

    .venv\\Scripts\\python scripts\\make_sample_docs.py
"""
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)
NOTE = "SAMPLE DOCUMENT - made-up content for testing FDA."


def pdf(path: Path, pages: list[list[str]]):
    """A tiny valid PDF with one text line per row (no extra libraries)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        esc = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
        stream = "BT /F1 12 Tf 72 760 Td 16 TL " + " ".join(f"({t}) Tj T*" for t in esc) + " ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_no = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_no} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


(DOCS / "sample-billing-policy.md").write_text(f"""# Billing policy (sample)
{NOTE}

## 4.1 Monthly plans
Monthly plans renew every month and can be cancelled at any time. There is no refund for the current month.

## 4.2 Annual plans
Annual plans are refundable pro rata within the first 60 days only. After 60 days there is no refund.

## 4.3 Late payments
A late fee of 200 pesos is added when a bill is more than 15 days overdue.
""", encoding="utf-8")

import docx                                                     # noqa: E402
d = docx.Document()
d.add_heading("Lab handbook (sample)", 0)
d.add_paragraph(NOTE)
d.add_heading("Wi-Fi", 1)
d.add_paragraph("The guest Wi-Fi network is called LabGuest and the password is printed on the fridge door.")
d.add_heading("Opening hours", 1)
d.add_paragraph("The lab is open from 8 am to 7 pm on weekdays, 9 am to 1 pm on Saturdays, and closed on Sundays.")
d.add_heading("Borrowing equipment", 1)
d.add_paragraph("Write your name in the logbook. Equipment must be returned within 3 days.")
d.save(DOCS / "sample-lab-handbook.docx")

import openpyxl                                                 # noqa: E402
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Inventory"
ws.append(["Item", "Quantity", "Location"])
for row in [("Raspberry Pi 5 8GB", 4, "Cabinet A"), ("USB microphone", 6, "Drawer 2"), ("USB speaker", 5, "Drawer 2"),
            ("7 inch display", 2, "Cabinet B"), ("Active cooler", 8, "Cabinet A"), ("27W USB-C power supply", 4, "Drawer 1")]:
    ws.append(row)
wb.save(DOCS / "sample-inventory.xlsx")

(DOCS / "sample-contacts.csv").write_text("Name,Role,Phone\nDr. Santos (sample),Adviser,0900 000 0101\n"
                                          "IT desk (sample),Support,0900 000 0199\nLab assistant (sample),Equipment,0900 000 0150\n",
                                          encoding="utf-8")

pdf(DOCS / "sample-safety-rules.pdf", [
    ["Safety rules (sample)", NOTE, "", "Unplug the Pi before changing the HAT.",
     "Never block the fan or the active cooler.", "Keep drinks away from the workbench."],
    ["Emergency (sample)", "Fire exits are at the end of the east hallway.",
     "The first aid kit is beside the sink.", "In an emergency call the IT desk or the building guard."],
])
print("made:", ", ".join(sorted(p.name for p in DOCS.iterdir())))
