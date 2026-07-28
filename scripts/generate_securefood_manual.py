#!/usr/bin/env python3
"""Generate the SecureFood-only quick user manual shipped with the app."""

from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, Table, TableStyle
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
OUTPUT = STATIC / "GROCERYsim_SecureFood_Scenario_Walkthrough_ClimateChange_Dairy.pdf"

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = colors.HexColor("#06292E")
TEAL = colors.HexColor("#177A80")
TEAL_DARK = colors.HexColor("#0E6268")
TEAL_LIGHT = colors.HexColor("#E8F6F4")
GOLD = colors.HexColor("#E1A552")
CREAM = colors.HexColor("#F8F5EE")
INK = colors.HexColor("#173238")
MUTED = colors.HexColor("#667B80")
LINE = colors.HexColor("#D8D0C3")
WHITE = colors.white


def register_fonts() -> None:
    fonts = STATIC / "fonts"
    pdfmetrics.registerFont(TTFont("Manual", str(fonts / "LiberationSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Manual-Bold", str(fonts / "LiberationSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Manual-Italic", str(fonts / "LiberationSans-Italic.ttf")))


def style(name: str, **kwargs) -> ParagraphStyle:
    defaults = dict(
        fontName="Manual",
        fontSize=9.2,
        leading=12.0,
        textColor=INK,
        spaceAfter=0,
        spaceBefore=0,
        alignment=TA_LEFT,
    )
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)


BODY = style("Body")
SMALL = style("Small", fontSize=8.2, leading=10.2)
CAPTION = style("Caption", fontSize=8.8, leading=11.0, textColor=MUTED)
H2 = style("H2", fontName="Manual-Bold", fontSize=17.5, leading=21, textColor=NAVY)
H3 = style("H3", fontName="Manual-Bold", fontSize=12.8, leading=15.5, textColor=NAVY)
TABLE_HEAD = style("TableHead", fontName="Manual-Bold", fontSize=8.4, leading=10.2, textColor=WHITE)
TABLE_CELL = style("TableCell", fontSize=8.1, leading=10.1)
TABLE_CELL_BOLD = style("TableCellBold", fontName="Manual-Bold", fontSize=8.1, leading=10.1)
STEP_TITLE = style("StepTitle", fontName="Manual-Bold", fontSize=9.0, leading=10.8)
STEP_BODY = style("StepBody", fontSize=8.7, leading=10.8)


def para(c: canvas.Canvas, text: str, x: float, y_top: float, width: float, pstyle=BODY) -> float:
    p = Paragraph(text, pstyle)
    _, height = p.wrap(width, PAGE_H)
    p.drawOn(c, x, y_top - height)
    return y_top - height


def body_header(c: canvas.Canvas, page: int) -> None:
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 16 * mm, PAGE_W, 16 * mm, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, PAGE_H - 16 * mm, 5 * mm, 16 * mm, fill=1, stroke=0)
    c.setFont("Manual-Bold", 7.7)
    c.setFillColor(WHITE)
    c.drawString(10 * mm, PAGE_H - 10.2 * mm, "GROCERYsim SecureFood - Quick User Manual")
    c.setFont("Manual-Italic", 7.7)
    c.setFillColor(GOLD)
    c.drawRightString(PAGE_W - 14 * mm, PAGE_H - 10.2 * mm, "Finland Dairy Supply Chain")

    c.setStrokeColor(LINE)
    c.line(MARGIN_X, 14 * mm, PAGE_W - MARGIN_X, 14 * mm)
    c.setFont("Manual", 7.1)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_X, 8.2 * mm, "SecureFood project | GROCERYsim ABM v2.0")
    c.drawRightString(PAGE_W - MARGIN_X, 8.2 * mm, f"Page {page}")


def section_title(c: canvas.Canvas, number: str, title: str, y: float) -> float:
    box = 13 * mm
    c.setFillColor(GOLD)
    c.rect(MARGIN_X, y - box, box, box, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Manual-Bold", 12)
    c.drawCentredString(MARGIN_X + box / 2, y - 8.7 * mm, number)
    para(c, title, MARGIN_X + box + 4 * mm, y - 2 * mm, CONTENT_W - box - 4 * mm, H2)
    return y - box - 3.2 * mm


def step_box(c: canvas.Canvas, number: str, title: str, text: str, y: float) -> float:
    row_h = 14.5 * mm
    c.setFillColor(CREAM)
    c.setStrokeColor(LINE)
    c.rect(MARGIN_X, y - row_h, CONTENT_W, row_h, fill=1, stroke=1)
    c.setFillColor(TEAL)
    c.rect(MARGIN_X, y - row_h, 16 * mm, row_h, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Manual-Bold", 12)
    c.drawCentredString(MARGIN_X + 8 * mm, y - 9.5 * mm, number)
    text_x = MARGIN_X + 20 * mm
    para(c, title, text_x, y - 3.4 * mm, CONTENT_W - 24 * mm, STEP_TITLE)
    para(c, text, text_x, y - 7.9 * mm, CONTENT_W - 24 * mm, STEP_BODY)
    return y - row_h - 2.4 * mm


def info_box(c: canvas.Canvas, title: str, text: str, y: float, height: float = 23 * mm) -> float:
    c.setFillColor(TEAL_LIGHT)
    c.roundRect(MARGIN_X, y - height, CONTENT_W, height, 3 * mm, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#15988E"))
    c.roundRect(MARGIN_X, y - height, 2.5 * mm, height, 1.2 * mm, fill=1, stroke=0)
    para(c, title, MARGIN_X + 7 * mm, y - 4.5 * mm, CONTENT_W - 12 * mm, H3)
    para(c, text, MARGIN_X + 7 * mm, y - 11.5 * mm, CONTENT_W - 12 * mm, BODY)
    return y - height - 3 * mm


def draw_table(c: canvas.Canvas, data, widths, y: float, font_size: float = 8.1) -> float:
    converted = []
    for row_index, row in enumerate(data):
        converted.append([
            Paragraph(str(cell), TABLE_HEAD if row_index == 0 else (
                TABLE_CELL_BOLD if col_index == 0 else TABLE_CELL
            ))
            for col_index, cell in enumerate(row)
        ])
    table = Table(converted, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM]),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    _, h = table.wrap(CONTENT_W, PAGE_H)
    table.drawOn(c, MARGIN_X, y - h)
    return y - h - 4 * mm


def page_one(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, PAGE_H - 15 * mm, PAGE_W, 3.5 * mm, fill=1, stroke=0)
    c.setStrokeColor(GOLD)
    c.rect(18 * mm, 28 * mm, PAGE_W - 36 * mm, PAGE_H - 48 * mm, fill=0, stroke=1)

    logos = [
        (STATIC / "SecureFood.png", 27 * mm, PAGE_H - 52 * mm, 55 * mm, 20 * mm),
        (STATIC / "GROCERYsim.png", 86 * mm, PAGE_H - 49 * mm, 42 * mm, 16 * mm),
        (STATIC / "Logo_lab.png", 145 * mm, PAGE_H - 51 * mm, 39 * mm, 18 * mm),
    ]
    for path, x, y, w, h in logos:
        source = str(path)
        if path.name == "GROCERYsim.png":
            cropped = PILImage.open(path).convert("RGBA")
            cropped = cropped.crop(cropped.getbbox())
            source = BytesIO()
            cropped.save(source, format="PNG")
            source.seek(0)
        img = Image(source, width=w, height=h, kind="proportional")
        img.drawOn(c, x, y)

    c.setFillColor(GOLD)
    c.setFont("Manual-Bold", 10)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 100 * mm, "QUICK USER MANUAL")
    c.setFillColor(WHITE)
    c.setFont("Manual-Bold", 28)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 116 * mm, "SecureFood Scenario Simulator")
    subtitle = Paragraph(
        "Navigate the Finland dairy supply chain scenario, run the policy-free default analysis, "
        "optionally test policy interventions, and download analysis-ready results.",
        style("CoverSub", fontSize=12, leading=16, textColor=colors.HexColor("#DCE8E8"), alignment=TA_CENTER),
    )
    _, sh = subtitle.wrap(155 * mm, 40 * mm)
    subtitle.drawOn(c, (PAGE_W - 155 * mm) / 2, PAGE_H - 140 * mm - sh)

    strip_y = PAGE_H - 170 * mm
    c.setFillColor(CREAM)
    c.rect(18.5 * mm, strip_y, PAGE_W - 37 * mm, 16 * mm, fill=1, stroke=0)
    labels = ["Open case study", "Run default scenario", "Optional policy analysis", "Download PDF and CSV"]
    for i, label in enumerate(labels):
        x = 18.5 * mm + (i + 0.5) * ((PAGE_W - 37 * mm) / 4)
        c.setFont("Manual-Bold", 7.6)
        c.setFillColor(NAVY)
        c.drawCentredString(x, strip_y + 6.7 * mm, label)
        if i < 3:
            c.setFont("Manual-Bold", 14)
            c.setFillColor(GOLD)
            c.drawString(x + 20 * mm, strip_y + 5.2 * mm, ">")

    c.setFillColor(TEAL_LIGHT)
    c.rect(17.5 * mm, PAGE_H - 208 * mm, PAGE_W - 35 * mm, 17 * mm, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(17.5 * mm, PAGE_H - 208 * mm, 2 * mm, 17 * mm, fill=1, stroke=0)
    para(c, "<b>Scope</b>", 22 * mm, PAGE_H - 196 * mm, 31 * mm, SMALL)
    para(c, "This guide covers only the SecureFood Scenario Simulator. It does not explain the wider "
         "GROCERYsim analysis, calibration, validation, or export workspaces.",
         58 * mm, PAGE_H - 196 * mm, 126 * mm, SMALL)

    link = "https://finland.streamlit.app/"
    c.setFillColor(GOLD)
    c.setFont("Manual", 11)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 225 * mm, "Open the live application: finland.streamlit.app")
    c.linkURL(link, (55 * mm, PAGE_H - 230 * mm, 155 * mm, PAGE_H - 220 * mm), relative=0)

    c.setFillColor(TEAL)
    c.rect(0, 0, PAGE_W, 20 * mm, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#D9ECEB"))
    c.setFont("Manual", 7.3)
    c.drawCentredString(PAGE_W / 2, 10 * mm, "SecureFood | Horizon Europe | IAMO XR Lab | July 2026")


def page_two(c: canvas.Canvas) -> None:
    body_header(c, 2)
    y = PAGE_H - 26 * mm
    y = section_title(c, "1", "Enter the SecureFood scenario", y)
    y = para(c, "The shortest path from the public landing page to the dedicated scenario workspace.", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, CAPTION) - 4 * mm
    y = step_box(c, "1", "Open the application", "Go to finland.streamlit.app. Select a language if needed.", y)
    y = step_box(c, "2", "Open the case studies", "Select <b>Explore case studies</b> on the landing page.", y)
    y = step_box(c, "3", "Choose Finland", "Find <b>Finland - Dairy Supply Chain</b> and select <b>Launch simulation</b>.", y)
    y = step_box(c, "4", "Open the scenario workspace", "Skip the guided tour if it appears, then select the green <b>Scenario Simulator</b> button.", y)

    y -= 2 * mm
    y = para(c, "What you will see", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, H3) - 3 * mm
    y = draw_table(c, [
        ["Page area", "Purpose"],
        ["Default scenario report", "Fixed, policy-free climate-disruption preset with parameters shown in an expandable panel."],
        ["Default Scenario Analysis", "Editable operational shock settings and a policy-free supply-chain simulation."],
        ["Additional Policy Analysis", "Separate counterfactual module. It is off and hidden by default until explicitly enabled."],
    ], [51 * mm, CONTENT_W - 51 * mm], y)

    y = para(c, "Fastest workflow: use the default preset", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, H3) - 3 * mm
    y = step_box(c, "1", "Review the preset", "Expand <b>View preset parameters</b>. The preset does not change when controls below are edited.", y)
    y = step_box(c, "2", "Generate", "Select <b>Generate Default Scenario Report</b>. The app runs the paired default conditions.", y)
    y = step_box(c, "3", "Download", "Choose the PDF report, Daily Results CSV, or Product Results CSV.", y)

    info_box(c, "Key separation rule", "The default report never includes a policy intervention. Enabling or editing the optional policy module does not alter the fixed default report.", y, 22 * mm)


def page_three(c: canvas.Canvas) -> None:
    body_header(c, 3)
    y = PAGE_H - 26 * mm
    y = section_title(c, "2", "Run an optional policy analysis", y)
    y = para(c, "Use this module only when the stakeholder question concerns an intervention. It compares the selected policy against the same crisis without policy.", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, CAPTION) - 4 * mm
    y = step_box(c, "1", "Open the policy tab", "Select <b>Additional Policy Analysis</b> beside the default-analysis tab.", y)
    y = step_box(c, "2", "Enable the module", "Tick <b>Enable additional policy analysis</b>. This reveals the counterfactual controls.", y)
    y = step_box(c, "3", "Define the shared crisis", "Set crisis severity, logistics and inventory, and behavioural assumptions for the paired comparison.", y)
    y = step_box(c, "4", "Select at least one policy lever", "The run and report buttons stay disabled until an intervention is active.", y)
    y = step_box(c, "5", "Generate policy outputs", "Select <b>Run Policy Simulation</b>, review the results, then select <b>Generate Report from This Analysis</b> for the matching PDF and CSV package.", y)

    y -= 2 * mm
    y = para(c, "Available policy controls", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, H3) - 3 * mm
    y = draw_table(c, [
        ["Control group", "Mechanisms available"],
        ["Access and communication", "Purchase rationing; government communication strategy; communication intensity."],
        ["Prices and affordability", "Domestic, organic, or combined product subsidy; fat-content surcharge and threshold."],
        ["Information and preferences", "Nutritional-labelling start day; health-preference and organic-preference boosts."],
        ["Counterfactual context", "Crisis timing and duration, inflation, disruption, lead time, reorder/restock targets, panic, and hoarding."],
    ], [53 * mm, CONTENT_W - 53 * mm], y)

    y = info_box(c, "What the comparison means", "The policy report contains a paired crisis-without-policy condition and a selected-policy condition using the same seed. It does not replace the default scenario report.", y, 25 * mm)
    info_box(c, "Scientific caution", "Panic, hoarding, communication, and labelling effects remain exploratory assumptions unless separately calibrated. Interpret differences as modelled counterfactuals, not validated causal estimates.", y, 25 * mm)


def page_four(c: canvas.Canvas) -> None:
    body_header(c, 4)
    y = PAGE_H - 26 * mm
    y = section_title(c, "3", "Download and interpret the results", y)
    y = para(c, "Each report action produces a PDF and two CSV files from the same simulation conditions.", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, CAPTION) - 4 * mm
    y = draw_table(c, [
        ["Download", "Best use", "Key contents"],
        ["PDF Report", "Presentation and review", "Inputs, charts, paired comparisons, limitations, and methodological notes."],
        ["Daily Results CSV", "Time-series analysis", "One row per day and condition with aggregate revenue, inventory, welfare, access, panic, waste, and policy outcomes."],
        ["Product Results CSV", "SKU/category analysis", "One row per product, day, and condition. This file is larger and can take longer to prepare."],
    ], [34 * mm, 42 * mm, CONTENT_W - 76 * mm], y)

    y = para(c, "Scenario labels in the exports", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, H3) - 3 * mm
    y = draw_table(c, [
        ["Label", "Meaning"],
        ["Baseline", "No crisis. Operational and welfare reference."],
        ["Crisis", "Policy-free crisis used in the default scenario analysis."],
        ["Crisis (No Policy)", "Counterfactual crisis inside the optional policy report, with all policy levers disabled."],
        ["Crisis (Selected Policy)", "The same optional-policy crisis with the selected intervention configuration."],
    ], [48 * mm, CONTENT_W - 48 * mm], y)

    y = para(c, "Choose the correct report button", MARGIN_X + 3 * mm, y, CONTENT_W - 3 * mm, H3) - 3 * mm
    y = draw_table(c, [
        ["Question", "Button to use"],
        ["Use the fixed stakeholder preset", "Generate Default Scenario Report"],
        ["Evaluate a selected policy against no policy", "Run Policy Simulation, then Generate Report from This Analysis"],
    ], [70 * mm, CONTENT_W - 70 * mm], y)

    y = info_box(c, "Five-point live demo checklist", "1. State whether the run is default or optional policy. &nbsp;&nbsp; 2. Show the included parameters. &nbsp;&nbsp; 3. Name the comparison conditions. &nbsp;&nbsp; 4. Keep unmet demand in units and check CSV column definitions. &nbsp;&nbsp; 5. Download the CSVs for reproducibility.", y, 29 * mm)
    y = info_box(c, "If the app is slow", "Streamlit Community Cloud may temporarily throttle CPU. Wait for the page to finish loading, avoid repeated clicks, and generate one report at a time. Completed artifacts remain available in the current session.", y, 24 * mm)

    c.setFont("Manual-Bold", 8.4)
    c.setFillColor(TEAL_DARK)
    c.drawString(MARGIN_X + 3 * mm, max(19 * mm, y - 1 * mm), "Live application: https://finland.streamlit.app/")
    c.linkURL("https://finland.streamlit.app/", (MARGIN_X, max(16 * mm, y - 4 * mm), MARGIN_X + 85 * mm, max(23 * mm, y + 3 * mm)), relative=0)


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    c.setTitle("GROCERYsim SecureFood Quick User Manual")
    c.setSubject("Quick guide to the SecureFood scenario simulator and optional policy analysis")
    c.setAuthor("IAMO XR Lab - SecureFood")
    for draw_page in (page_one, page_two, page_three, page_four):
        draw_page(c)
        c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
