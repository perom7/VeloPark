"""Synchronous utility tasks – no Celery/Redis dependency."""

import os
from datetime import datetime
from io import BytesIO
from .models import User, Reservation, ParkingLot, ParkingSpot, Role
from .database import db
from .notify import notify_user

try:
    from xhtml2pdf import pisa
except Exception:
    pisa = None

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.graphics.shapes import Drawing, Line
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.widgets.markers import makeMarker
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


def generate_admin_report_pdf():
    """Generate a comprehensive admin report PDF with charts and save it to the reports directory.
    Returns the file path on success, None on failure.
    """
    if not HAS_REPORTLAB:
        return None

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    now = datetime.utcnow()
    fname = f"admin-report-{now.strftime('%Y-%m-%d_%H%M%S')}.pdf"
    fpath = os.path.join(reports_dir, fname)

    doc = SimpleDocTemplate(
        fpath,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#0d6efd"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=16,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )

    elements = []

    # ── Title ─────────────────────────────────────────────
    elements.append(Paragraph("VeloPark — Admin Report", title_style))
    elements.append(Paragraph(
        f"Generated on {now.strftime('%B %d, %Y at %H:%M UTC')}",
        subtitle_style,
    ))

    # ── Summary Stats ─────────────────────────────────────
    total_users = User.query.count()
    total_lots = ParkingLot.query.count()
    total_spots = ParkingSpot.query.count()
    total_reservations = Reservation.query.count()
    completed = Reservation.query.filter(Reservation.end_time.isnot(None)).count()
    active = Reservation.query.filter(Reservation.end_time.is_(None)).count()
    total_revenue = db.session.query(db.func.coalesce(db.func.sum(Reservation.parking_cost), 0.0)).scalar() or 0.0

    elements.append(Paragraph("System Overview", heading_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total Users", str(total_users)],
        ["Total Parking Lots", str(total_lots)],
        ["Total Parking Spots", str(total_spots)],
        ["Total Reservations", str(total_reservations)],
        ["Active Reservations", str(active)],
        ["Completed Reservations", str(completed)],
        ["Total Revenue", f"\u20b9{total_revenue:.2f}"],
    ]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # ── Per-Lot Breakdown ─────────────────────────────────
    lots = ParkingLot.query.order_by(ParkingLot.created_at.desc()).all()
    if lots:
        elements.append(Paragraph("Parking Lot Details", heading_style))
        lot_header = ["Lot Name", "Price/hr", "Total Spots", "Occupied", "Available", "Reservations", "Revenue"]
        lot_rows = [lot_header]
        for lot in lots:
            occ = sum(1 for s in lot.spots if s.status == "O")
            avail = len(lot.spots) - occ
            # Get reservation count and revenue for this lot
            spot_ids = [s.id for s in lot.spots]
            lot_res_count = Reservation.query.filter(Reservation.spot_id.in_(spot_ids)).count() if spot_ids else 0
            lot_revenue = db.session.query(
                db.func.coalesce(db.func.sum(Reservation.parking_cost), 0.0)
            ).filter(Reservation.spot_id.in_(spot_ids)).scalar() if spot_ids else 0.0
            lot_rows.append([
                lot.prime_location_name,
                f"\u20b9{lot.price_per_hour:.2f}",
                str(len(lot.spots)),
                str(occ),
                str(avail),
                str(lot_res_count),
                f"\u20b9{lot_revenue:.2f}",
            ])
        lot_table = Table(lot_rows, colWidths=[140, 65, 70, 65, 65, 80, 80])
        lot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(lot_table)
        elements.append(Spacer(1, 12))

    # ── Per-User Revenue ──────────────────────────────────
    user_revenue_rows = (
        db.session.query(
            User.username,
            db.func.count(Reservation.id).label("count"),
            db.func.coalesce(db.func.sum(Reservation.parking_cost), 0.0).label("revenue"),
        )
        .join(Reservation, Reservation.user_id == User.id)
        .group_by(User.id, User.username)
        .order_by(db.func.sum(Reservation.parking_cost).desc())
        .all()
    )
    if user_revenue_rows:
        elements.append(Paragraph("Revenue by User", heading_style))
        ur_header = ["Username", "Reservations", "Total Revenue"]
        ur_data = [ur_header]
        for row in user_revenue_rows:
            ur_data.append([row.username, str(row.count), f"\u20b9{float(row.revenue):.2f}"])
        ur_table = Table(ur_data, colWidths=[200, 120, 120])
        ur_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#faf5ff"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(ur_table)
        elements.append(Spacer(1, 12))

    # ── Most Booked Lots Chart ────────────────────────────
    # Query for booking counts per lot
    lot_data = (
        db.session.query(
            ParkingLot.prime_location_name,
            db.func.count(Reservation.id)
        )
        .join(ParkingSpot, ParkingSpot.lot_id == ParkingLot.id)
        .join(Reservation, Reservation.spot_id == ParkingSpot.id)
        .group_by(ParkingLot.id, ParkingLot.prime_location_name)
        .order_by(db.func.count(Reservation.id).asc())  # Ascending for horizontal chart (bottom to top)
        .all()
    )

    if lot_data:
        elements.append(Paragraph("Most Booked Lots", heading_style))
        elements.append(Paragraph(
            "Total number of bookings per parking lot.",
            body_style,
        ))
        elements.append(Spacer(1, 10))

        names = [r[0] for r in lot_data]
        counts = [r[1] for r in lot_data]

        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import HorizontalBarChart

        drawing = Drawing(550, 200)
        bc = HorizontalBarChart()
        bc.x = 100
        bc.y = 20
        bc.height = 160
        bc.width = 400
        bc.data = [counts]
        bc.strokeColor = colors.white
        bc.valueAxis.valueMin = 0
        bc.valueAxis.gridStrokeColor = colors.HexColor("#e2e8f0")
        bc.valueAxis.gridStrokeWidth = 0.5
        
        # Determine strict integer steps for the axis
        max_val = max(counts) if counts else 0
        if max_val <= 10:
             bc.valueAxis.valueMax = max_val + 1
             bc.valueAxis.valueStep = 1
        
        bc.categoryAxis.labels.boxAnchor = 'e'
        bc.categoryAxis.categoryNames = names
        bc.categoryAxis.labels.fontName = 'Helvetica'
        bc.categoryAxis.labels.fontSize = 10
        bc.categoryAxis.labels.dx = -5
        
        bc.bars[0].fillColor = colors.HexColor("#6366f1")
        
        drawing.add(bc)
        elements.append(drawing)
        elements.append(Spacer(1, 20))

    # ── Recent Reservations ───────────────────────────────
    recent = Reservation.query.order_by(Reservation.start_time.desc()).limit(30).all()
    if recent:
        elements.append(Paragraph("Recent Reservations (last 30)", heading_style))
        res_header = ["ID", "User", "Lot", "Spot", "Start", "End", "Cost"]
        res_data = [res_header]
        for r in recent:
            username = r.user.username if r.user else "N/A"
            lot_name = r.spot.lot.prime_location_name if r.spot and r.spot.lot else "N/A"
            spot_no = r.spot.index_number if r.spot else "N/A"
            res_data.append([
                str(r.id),
                username,
                lot_name,
                str(spot_no),
                r.start_time.strftime("%Y-%m-%d %H:%M"),
                r.end_time.strftime("%Y-%m-%d %H:%M") if r.end_time else "Active",
                f"\u20b9{r.parking_cost:.2f}" if r.parking_cost else "\u2014",
            ])
        res_table = Table(res_data, colWidths=[40, 80, 110, 45, 110, 110, 65])
        res_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f9ff"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(res_table)

    # ── Build PDF ─────────────────────────────────────────
    try:
        doc.build(elements)
        return fpath
    except Exception as e:
        print(f"[PDF] Error building report: {e}")
        return None


def export_csv_sync(user_id: int) -> str:
    """Generate CSV content for a user's reservations synchronously."""
    import csv
    from io import StringIO

    rows = Reservation.query.filter_by(user_id=user_id).all()
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["reservation_id", "spot_id", "lot_id", "lot_name", "index_number",
                      "start_time", "end_time", "cost"])
    for r in rows:
        writer.writerow([
            r.id,
            r.spot_id,
            r.spot.lot_id,
            r.spot.lot.prime_location_name,
            r.spot.index_number,
            r.start_time.isoformat(),
            r.end_time.isoformat() if r.end_time else "",
            r.parking_cost or 0,
        ])
    return buf.getvalue()


def notify_new_lot_created(lot_id: int):
    """Broadcast a notification to all non-admin users when a new lot is created."""
    lot = ParkingLot.query.get(lot_id)
    if not lot:
        return
    users = User.query.filter(User.role != Role.ADMIN.value).all()
    for u in users:
        to = u.email or f"{u.username}@example.local"
        subject = "New Parking Lot Available"
        body = f"A new parking lot '{lot.prime_location_name}' has been added. Spots: {lot.number_of_spots}. Price/hr: {lot.price_per_hour}."
        notify_user(to, subject, body, html=False)
