"""Generate REPORT.pdf from REPORT.md using a lightweight Markdown -> HTML -> PDF pipeline.
Requires: xhtml2pdf (already in requirements). This is a minimal converter (not full Markdown spec).
"""
from pathlib import Path
import re
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
md_path = ROOT / "REPORT.md"
pdf_path = ROOT / "REPORT.pdf"

def md_to_html(md: str) -> str:
    lines = md.splitlines()
    html_lines = []
    for line in lines:
        if line.startswith('# '):
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith('## '):
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith('### '):
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith('- '):
            # Simple bullet list handling: group consecutive - items
            if not html_lines or not html_lines[-1].startswith('<ul'):  # start list
                html_lines.append('<ul>')
            html_lines.append(f"<li>{line[2:].strip()}</li>")
        elif line.strip() == '' and html_lines and html_lines[-1].startswith('<li'):
            # Close list on blank line
            html_lines.append('</ul>')
        elif line.startswith('```'):
            # Ignore fenced code blocks start/end for simplicity
            continue
        else:
            # Inline bold **text**
            processed = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            html_lines.append(f"<p>{processed}</p>")
    if html_lines and html_lines[-1] == '</ul>':
        pass
    # Ensure any dangling list is closed
    open_ul = False
    for l in html_lines:
        if l == '<ul>':
            open_ul = True
        elif l == '</ul>':
            open_ul = False
    if open_ul:
        html_lines.append('</ul>')
    style = """
    <style>
      body { font-family: Arial, sans-serif; font-size: 12px; line-height: 1.4; }
      h1,h2,h3 { color: #2d3e50; margin-bottom: 6px; }
      h1 { border-bottom: 2px solid #667eea; padding-bottom: 4px; }
      ul { margin: 4px 0 8px 20px; }
      li { margin-bottom: 4px; }
      table { border-collapse: collapse; width:100%; margin:8px 0; }
      th, td { border: 1px solid #ccc; padding: 4px; text-align:left; }
      th { background:#f5f5f5; }
      code { background:#f0f0f0; padding:2px 4px; border-radius:4px; }
      .foot { margin-top:24px; font-size:10px; color:#555; }
    </style>
    """
    return f"<html><head>{style}</head><body>" + "\n".join(html_lines) + "<div class='foot'>Generated via generate_report.py</div></body></html>"


def main():
    if not md_path.exists():
        print(f"REPORT.md not found at {md_path}")
        return
    md = md_path.read_text(encoding='utf-8')
    html = md_to_html(md)
    with pdf_path.open('wb') as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        print("PDF generation failed")
    else:
        size_kb = pdf_path.stat().st_size / 1024
        print(f"REPORT.pdf created ({size_kb:.1f} KB) -> {pdf_path}")

if __name__ == '__main__':
    main()
