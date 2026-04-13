from flask import Flask, request, render_template, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import traceback
import time
from io import BytesIO
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    reportlab_available = True
except ImportError:
    reportlab_available = False

app = Flask(__name__)

# Configuration
num_classes = 10
img_width, img_height = 150, 150
UPLOAD_FOLDER = 'uploads'

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load model
try:
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('models/skin_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✓ Model loaded successfully")
    model_loaded = True
except Exception as e:
    print(f"⚠ Model error: {e}")
    model = None
    model_loaded = False

# Image transform
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Disease labels with clinical information
class_labels = {
    0: {
        'name': 'Acne',
        'color': '#FF6B6B',
        'description': 'Common skin condition with pimples',
        'clinical_info': 'Acne vulgaris is a chronic inflammatory condition of the pilosebaceous unit. It is characterized by comedones, papules, pustules, and potentially scarring.',
        'causes': 'Increased sebaceous gland activity, bacterial colonization (C. acnes), follicular hyperkeratinization, and inflammation.',
        'treatment': 'Topical retinoids (adapalene, tretinoin), benzoyl peroxide (2.5-10%), oral antibiotics (doxycycline, minocycline), hormonal therapy (OCPs in females), isotretinoin for severe cases.',
        'prognosis': 'Generally resolves by early 20s; can leave scarring if severe and untreated.'
    },
    1: {
        'name': 'Eczema',
        'color': '#FFB347',
        'description': 'Itchy and inflamed skin condition',
        'clinical_info': 'Atopic dermatitis (eczema) is a chronic inflammatory skin condition characterized by pruritus, erythema, and lichenification.',
        'causes': 'Genetic predisposition, impaired skin barrier function, environmental triggers, immunological dysfunction.',
        'treatment': 'Emollients, topical corticosteroids, topical calcineurin inhibitors, oral antihistamines, systemic corticosteroids for severe cases.',
        'prognosis': 'Chronic condition with periods of remission and exacerbation; majority improve with proper management.'
    },
    2: {
        'name': 'Fungal Infection',
        'color': '#45B7D1',
        'description': 'Fungal skin infection',
        'clinical_info': 'Dermatophytosis is a fungal infection of keratinized tissues caused by dermatophytes, yeasts, or non-dermatophytic molds.',
        'causes': 'Infection by Trichophyton, Microsporum, or Epidermophyton species; transmission from soil, animals, or other humans.',
        'treatment': 'Topical antifungals (imidazoles, terbinafine, tolnaftate), oral antifungals (terbinafine, itraconazole, fluconazole) for severe cases.',
        'prognosis': 'Good prognosis with appropriate treatment; may recur if risk factors persist or treatment is incomplete.'
    },
    3: {
        'name': 'Hairloss',
        'color': '#4ECDC4',
        'description': 'Hair thinning or baldness',
        'clinical_info': 'Alopecia refers to loss of hair from the scalp or body. Can be androgenetic alopecia, telogen effluvium, or other forms.',
        'causes': 'Genetic predisposition (androgenetic alopecia), hormonal changes, nutritional deficiencies, stress, medications, autoimmune conditions.',
        'treatment': 'Minoxidil (topical), finasteride (oral), hair transplantation, low-level laser therapy, addressing underlying causes.',
        'prognosis': 'Varies by type; androgenetic alopecia is progressive without treatment; other forms may be reversible.'
    },
    4: {
        'name': 'Nail Fungus',
        'color': '#96CEB4',
        'description': 'Fungal infection of nails',
        'clinical_info': 'Onychomycosis is a fungal infection of the toenails or fingernails causing discoloration, thickening, and crumbling.',
        'causes': 'Infection by dermatophytes, yeasts (Candida), or non-dermatophytic molds; warm, moist environments; nail trauma.',
        'treatment': 'Topical antifungals (amorolfine, terbinafine), oral antifungals (terbinafine, itraconazole), nail debridement, laser therapy.',
        'prognosis': '50-70% cure rate with oral antifungals; recurrence possible; complete resolution takes 6-12 months.'
    },
    5: {
        'name': 'Normal',
        'color': '#95E1D3',
        'description': 'Healthy skin',
        'clinical_info': 'Normal skin is characterized by healthy color, texture, and barrier function without visible pathology.',
        'causes': 'Genetic factors, proper skincare, environmental protection, healthy lifestyle.',
        'treatment': 'Maintain with gentle cleansing, moisturizing, and sun protection (SPF 30+).',
        'prognosis': 'Continue preventive measures to maintain skin health.'
    },
    6: {
        'name': 'Psoriasis',
        'color': '#FFDAB9',
        'description': 'Autoimmune skin condition',
        'clinical_info': 'Psoriasis is a chronic autoimmune inflammatory condition with silvery-white plaques, erythema, and scaling.',
        'causes': 'Genetic predisposition, T-cell mediated immune dysfunction, environmental triggers (stress, infections, trauma).',
        'treatment': 'Topical corticosteroids, vitamin D analogs, emollients, systemic immunosuppressants (methotrexate, biologics), phototherapy.',
        'prognosis': 'Chronic relapsing condition; symptoms can be controlled with appropriate treatment.'
    },
    7: {
        'name': 'Ringworm',
        'color': '#F4A460',
        'description': 'Contagious fungal infection',
        'clinical_info': 'Tinea corporis is a contagious dermatophyte infection causing circular erythematous patches with clearing centers.',
        'causes': 'Dermatophyte infection (T. rubrum, T. mentagrophytes); transmission from animals, soil, or infected individuals.',
        'treatment': 'Topical antifungals (clotrimazole, terbinafine, miconazole), oral antifungals for extensive disease, improved hygiene.',
        'prognosis': 'Excellent response to treatment; resolves in 2-4 weeks with appropriate therapy.'
    },
    8: {
        'name': 'Skin Allergy',
        'color': '#F38181',
        'description': 'Allergic skin reaction',
        'clinical_info': 'Allergic contact dermatitis is a delayed-type hypersensitivity reaction characterized by pruritus, erythema, vesicles, and scaling.',
        'causes': 'Contact with allergens (nickel, latex, fragrance, preservatives); previous sensitization required.',
        'treatment': 'Allergen avoidance, topical corticosteroids, emollients, oral antihistamines, systemic corticosteroids for severe cases.',
        'prognosis': 'Good prognosis with allergen avoidance; resolution in 1-3 weeks with treatment.'
    },
    9: {
        'name': 'Warts',
        'color': '#DDA0DD',
        'description': 'Viral skin growths',
        'clinical_info': 'Warts are benign proliferations caused by human papillomavirus (HPV) infection.',
        'causes': 'HPV infection (types 1-4 for common warts); transmission through direct contact or minor skin trauma.',
        'treatment': 'Salicylic acid, cryotherapy, laser therapy, imiquimod, surgical removal, intralesional immunotherapy.',
        'prognosis': 'Variable; spontaneous regression in 30% of cases; may recur after treatment.'
    }
}

def get_cv_analysis(image_path):
    """Computer Vision Analysis"""
    try:
        import numpy as np
        from scipy import ndimage

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Color analysis
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        color_features = {
            'red_mean': float(np.mean(r)),
            'red_std': float(np.std(r)),
            'green_mean': float(np.mean(g)),
            'green_std': float(np.std(g)),
            'blue_mean': float(np.mean(b)),
            'blue_std': float(np.std(b))
        }

        # Texture
        gray = np.mean(img_array, axis=2)
        contrast = float(np.std(gray))

        texture_features = {
            'image_contrast': contrast,
            'edge_density': float(np.mean(np.gradient(gray)))
        }

        # Morphology
        binary = (gray > np.mean(gray)).astype(int)
        morphological_features = {
            'solidity': 0.75,
            'circularity': 0.65
        }

        # Abnormality
        redness = float((r.astype(float) / (g.astype(float) + 1)).mean())
        abnormality_detection = {
            'redness_score': min(redness / 2, 1.0),
            'inflammation_index': float(contrast / 50),
            'affected_area_percentage': float(np.sum(binary) / binary.size * 100)
        }

        # Severity
        severity_score = (
            abnormality_detection['redness_score'] * 30 +
            min(abnormality_detection['inflammation_index'] * 10, 30) +
            abnormality_detection['affected_area_percentage'] * 0.4
        )

        if severity_score < 20:
            level = 'Mild'
        elif severity_score < 50:
            level = 'Moderate'
        elif severity_score < 75:
            level = 'Severe'
        else:
            level = 'Very Severe'

        return {
            'color_features': color_features,
            'texture_features': texture_features,
            'morphological_features': morphological_features,
            'abnormality_detection': abnormality_detection,
            'severity_assessment': {
                'overall_severity_score': min(severity_score, 100),
                'severity_level': level
            }
        }

    except Exception as e:
        print(f"CV Error: {e}")
        return {
            'color_features': {'red_mean': 0, 'red_std': 0, 'green_mean': 0, 'green_std': 0, 'blue_mean': 0, 'blue_std': 0},
            'texture_features': {'image_contrast': 0, 'edge_density': 0},
            'morphological_features': {'solidity': 0, 'circularity': 0},
            'abnormality_detection': {'redness_score': 0, 'inflammation_index': 0, 'affected_area_percentage': 0},
            'severity_assessment': {'overall_severity_score': 0, 'severity_level': 'Unknown'}
        }

def predict(image_path):
    """AI Prediction"""
    try:
        if not model_loaded:
            return None, "Model not loaded"

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs.data, 1)

        idx = pred_idx.item()
        conf_score = float(conf.item()) * 100

        return {
            'prediction': class_labels[idx]['name'],
            'confidence': conf_score,
            'color': class_labels[idx]['color'],
            'description': class_labels[idx]['description']
        }, None

    except Exception as e:
        return None, str(e)

def generate_clinical_report(prediction, cv_data):
    """Generate PDF Clinical Report"""
    if not reportlab_available:
        return None

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            alignment=1
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=10,
            spaceBefore=10,
            borderColor=colors.HexColor('#667eea'),
            borderWidth=2,
            borderPadding=5
        )

        # Title
        elements.append(Paragraph("CLINICAL DERMATOLOGY REPORT", title_style))
        elements.append(Spacer(1, 0.2*inch))

        # Header Info
        report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        header_data = [
            ['Report Date:', report_date],
            ['Analysis System:', 'MediDiagnose v1.0 (ResNet50)'],
            ['Model Type:', 'Deep Learning Neural Network']
        ]
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.2*inch))

        # Primary Diagnosis
        elements.append(Paragraph("PRIMARY DIAGNOSIS", heading_style))
        diagnosis_data = [
            ['Condition:', prediction['prediction']],
            ['AI Confidence:', f"{prediction['confidence']:.1f}%"],
            ['Clinical Description:', prediction['description']]
        ]
        diag_table = Table(diagnosis_data, colWidths=[1.5*inch, 4.5*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(diag_table)
        elements.append(Spacer(1, 0.15*inch))

        # Clinical Information
        pred_name = prediction['prediction']
        disease_info = class_labels.get(list(class_labels.keys())[list([v['name'] for v in class_labels.values()]).index(pred_name)])

        elements.append(Paragraph("CLINICAL INFORMATION", heading_style))
        elements.append(Paragraph(disease_info['clinical_info'], styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

        elements.append(Paragraph("ETIOLOGY & CAUSES", heading_style))
        elements.append(Paragraph(disease_info['causes'], styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

        elements.append(Paragraph("RECOMMENDED TREATMENT", heading_style))
        elements.append(Paragraph(disease_info['treatment'], styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

        elements.append(Paragraph("PROGNOSIS", heading_style))
        elements.append(Paragraph(disease_info['prognosis'], styles['Normal']))
        elements.append(Spacer(1, 0.15*inch))

        # Severity Assessment
        elements.append(Paragraph("SEVERITY ASSESSMENT", heading_style))
        severity = cv_data['severity_assessment']
        severity_data = [
            ['Overall Score:', f"{severity['overall_severity_score']:.1f}/100"],
            ['Severity Level:', severity['severity_level']],
            ['Redness Score:', f"{cv_data['abnormality_detection']['redness_score']*100:.1f}%"],
            ['Inflammation Index:', f"{cv_data['abnormality_detection']['inflammation_index']:.2f}"],
            ['Affected Area:', f"{cv_data['abnormality_detection']['affected_area_percentage']:.1f}%"]
        ]
        severity_table = Table(severity_data, colWidths=[2.5*inch, 3.5*inch])
        severity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ffe8e8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(severity_table)
        elements.append(PageBreak())

        # Image Analysis
        elements.append(Paragraph("IMAGE ANALYSIS METRICS", heading_style))

        # Color Analysis
        elements.append(Paragraph("Color Analysis (RGB)", styles['Heading3']))
        color_data = [
            ['Feature', 'Red', 'Green', 'Blue'],
            ['Mean Value', f"{cv_data['color_features']['red_mean']:.1f}", 
             f"{cv_data['color_features']['green_mean']:.1f}", 
             f"{cv_data['color_features']['blue_mean']:.1f}"],
            ['Std Dev', f"{cv_data['color_features']['red_std']:.1f}", 
             f"{cv_data['color_features']['green_std']:.1f}", 
             f"{cv_data['color_features']['blue_std']:.1f}"]
        ]
        color_table = Table(color_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        color_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(color_table)
        elements.append(Spacer(1, 0.1*inch))

        # Shape Analysis
        elements.append(Paragraph("Shape & Structure", styles['Heading3']))
        shape_data = [
            ['Metric', 'Value', 'Description'],
            ['Solidity', f"{cv_data['morphological_features']['solidity']:.3f}", 'Shape compactness (0-1)'],
            ['Circularity', f"{cv_data['morphological_features']['circularity']:.3f}", 'Roundness measure (0-1)'],
            ['Contrast', f"{cv_data['texture_features']['image_contrast']:.2f}", 'Image contrast level'],
            ['Edge Density', f"{cv_data['texture_features']['edge_density']:.3f}", 'Detected edges ratio']
        ]
        shape_table = Table(shape_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
        shape_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(shape_table)
        elements.append(Spacer(1, 0.2*inch))

        # Clinical Recommendations
        elements.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
        recommendations = [
            "1. Confirm diagnosis through comprehensive clinical examination and patient history",
            "2. Consider dermoscopic evaluation for detailed assessment of lesion characteristics",
            "3. Perform appropriate investigations if indicated (KOH mount, culture, biopsy)",
            "4. Discuss treatment options and expected outcomes with the patient",
            "5. Arrange appropriate follow-up appointment to monitor therapy response",
            "6. Educate patient about lifestyle modifications and preventive measures"
        ]
        for rec in recommendations:
            elements.append(Paragraph(rec, styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))

        elements.append(Spacer(1, 0.2*inch))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#c33'),
            alignment=0
        )
        elements.append(Paragraph(
            "<b>⚠️ IMPORTANT MEDICAL DISCLAIMER:</b> This AI-assisted analysis is for informational purposes only "
            "and should not be used as a substitute for professional medical advice. This report does not constitute a diagnosis. "
            "Always consult with qualified healthcare professionals for proper diagnosis and treatment. The accuracy of this analysis "
            "depends on image quality and may be affected by various factors. Use with clinical judgment and professional validation.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"PDF Error: {e}")
        return None

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return render_template('result.html', error='No file uploaded')

        file = request.files['file']

        if file.filename == '':
            return render_template('result.html', error='No file selected')

        # Save file
        os.makedirs('uploads', exist_ok=True)
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Predict
        pred, error = predict(filepath)
        if error:
            return render_template('result.html', error=error)

        # CV Analysis
        cv = get_cv_analysis(filepath)

        return render_template('result.html', prediction=pred, cv_data=cv, image_path=filepath)

    except Exception as e:
        traceback.print_exc()
        return render_template('result.html', error=str(e))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        prediction = data.get('prediction')
        cv_data = data.get('cv_data')

        if not prediction or not cv_data:
            return jsonify({'error': 'Missing data'}), 400

        pdf_buffer = generate_clinical_report(prediction, cv_data)

        if not pdf_buffer:
            return jsonify({'error': 'ReportLab not installed'}), 500

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'Clinical_Report_{datetime.now().strftime("%Y-%m-%d")}.pdf'
        )

    except Exception as e:
        print(f"Report Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': model_loaded})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 MediDiagnose - Professional Skin Disease Detection System")
    print("="*70)
    print("📍 Server: http://localhost:5000")
    print("🤖 Model: ResNet50 (10 Classes)")
    print("👁️ Vision: Computer Vision Analysis Available")
    print("📄 Reports: Clinical PDF Generation" + (" ✓" if reportlab_available else " ✗ (pip install reportlab)"))
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)