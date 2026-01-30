"""
Export utilities for reports and data.
"""

import csv
import io
from datetime import datetime, UTC
from typing import List, Dict
from sqlalchemy.orm import Session
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from db import AnalysisResult


def export_to_csv(db: Session, route_id: str, output_path: str = None) -> str:
    """Export route analysis data to CSV."""
    results = db.query(AnalysisResult).filter(
        AnalysisResult.route_id.like(f"{route_id}%")
    ).order_by(AnalysisResult.timestamp.desc()).all()
    
    if output_path:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'route_id', 'travel_time_min', 'delay_min',
                'distance_km', 'congestion_ratio', 'cost', 'ml_prediction'
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'timestamp': r.timestamp.isoformat() if r.timestamp else '',
                    'route_id': r.route_id,
                    'travel_time_min': round(r.travel_time_s / 60, 2) if r.travel_time_s else '',
                    'delay_min': round(r.delay_s / 60, 2) if r.delay_s else '',
                    'distance_km': round(r.length_m / 1000, 2) if r.length_m else '',
                    'congestion_ratio': round(r.travel_time_s / r.no_traffic_s, 2) if r.no_traffic_s and r.no_traffic_s > 0 else '',
                    'cost': round(r.calculated_cost, 2) if r.calculated_cost else '',
                    'ml_prediction': round(r.ml_predicted, 2) if r.ml_predicted else ''
                })
        return output_path
    else:
        # Return as string
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'timestamp', 'route_id', 'travel_time_min', 'delay_min',
            'distance_km', 'congestion_ratio', 'cost', 'ml_prediction'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'timestamp': r.timestamp.isoformat() if r.timestamp else '',
                'route_id': r.route_id,
                'travel_time_min': round(r.travel_time_s / 60, 2) if r.travel_time_s else '',
                'delay_min': round(r.delay_s / 60, 2) if r.delay_s else '',
                'distance_km': round(r.length_m / 1000, 2) if r.length_m else '',
                'congestion_ratio': round(r.travel_time_s / r.no_traffic_s, 2) if r.no_traffic_s and r.no_traffic_s > 0 else '',
                'cost': round(r.calculated_cost, 2) if r.calculated_cost else '',
                'ml_prediction': round(r.ml_predicted, 2) if r.ml_predicted else ''
            })
        return output.getvalue()


def export_to_excel(db: Session, route_id: str, output_path: str) -> str:
    """Export route analysis data to Excel."""
    results = db.query(AnalysisResult).filter(
        AnalysisResult.route_id.like(f"{route_id}%")
    ).order_by(AnalysisResult.timestamp.desc()).all()
    
    data = []
    for r in results:
        data.append({
            'Timestamp': r.timestamp.isoformat() if r.timestamp else '',
            'Route ID': r.route_id,
            'Travel Time (min)': round(r.travel_time_s / 60, 2) if r.travel_time_s else '',
            'Delay (min)': round(r.delay_s / 60, 2) if r.delay_s else '',
            'Distance (km)': round(r.length_m / 1000, 2) if r.length_m else '',
            'Congestion Ratio': round(r.travel_time_s / r.no_traffic_s, 2) if r.no_traffic_s and r.no_traffic_s > 0 else '',
            'Cost': round(r.calculated_cost, 2) if r.calculated_cost else '',
            'ML Prediction': round(r.ml_predicted, 2) if r.ml_predicted else ''
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    return output_path


def export_to_pdf(db: Session, route_id: str, output_path: str, title: str = "Route Analysis Report") -> str:
    """Export route analysis data to PDF."""
    results = db.query(AnalysisResult).filter(
        AnalysisResult.route_id.like(f"{route_id}%")
    ).order_by(AnalysisResult.timestamp.desc()).limit(100).all()
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_para = Paragraph(title, styles['Title'])
    elements.append(title_para)
    elements.append(Spacer(1, 0.2*inch))
    
    # Summary
    summary_text = f"<b>Route ID:</b> {route_id}<br/>"
    summary_text += f"<b>Total Records:</b> {len(results)}<br/>"
    summary_text += f"<b>Generated:</b> {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    summary_para = Paragraph(summary_text, styles['Normal'])
    elements.append(summary_para)
    elements.append(Spacer(1, 0.3*inch))
    
    # Table data
    table_data = [['Timestamp', 'Travel Time (min)', 'Delay (min)', 'Distance (km)', 'Congestion', 'Cost']]
    
    for r in results[:50]:  # Limit to 50 rows for PDF
        table_data.append([
            r.timestamp.strftime('%Y-%m-%d %H:%M') if r.timestamp else '',
            f"{round(r.travel_time_s / 60, 1)}" if r.travel_time_s else '',
            f"{round(r.delay_s / 60, 1)}" if r.delay_s else '',
            f"{round(r.length_m / 1000, 1)}" if r.length_m else '',
            f"{round(r.travel_time_s / r.no_traffic_s, 2)}x" if r.no_traffic_s and r.no_traffic_s > 0 else 'N/A',
            f"{round(r.calculated_cost, 2)}" if r.calculated_cost else ''
        ])
    
    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    elements.append(table)
    doc.build(elements)
    return output_path
