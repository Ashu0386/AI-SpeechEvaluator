import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 800, "AI Speech Evaluator — Complete Study & Revision Guide")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 792, 541, 792)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 35, footer_text)
        self.drawString(54, 35, "Self-Study & Viva Preparation Document")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 541, 48)
        self.restoreState()

def create_study_guide_pdf(filename="AI_Speech_Evaluator_Study_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1A202C')
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2C5282')
    )

    story = []

    # Title block
    story.append(Paragraph("AI Speech Evaluator", title_style))
    story.append(Paragraph("Detailed Self-Study & Revision Notes | Architecture, Models, and Viva Q&amp;A", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=14))

    # Section 1: Overview
    story.append(Paragraph("1. Project Overview &amp; The Core Problem", h1_style))
    story.append(Paragraph(
        "In technical interviews (software engineering, system design), candidates are judged on two distinct dimensions: "
        "<b>Delivery Fluency</b> (clarity, minimal verbal hesitations like 'um'/'uh') and <b>Technical Substance</b> (proper engineering depth like indexing, caching, latency). "
        "The AI Speech Evaluator automates this evaluation by converting spoken answers into text, analyzing linguistic metrics, computing semantic embeddings, and classifying responses as <b>GOOD</b> or <b>NEEDS IMPROVEMENT</b>.",
        body_style
    ))

    # Callout Box: 4-Step Pipeline Summary
    pipeline_box_data = [[
        Paragraph(
            "<b>The 4-Step Architecture Pipeline:</b><br/>"
            "<b>1. Audio Transcription:</b> OpenAI Whisper converts speech into verbatim text without stripping filler words.<br/>"
            "<b>2. Feature Extraction:</b> Linguistic analysis checks filler count, hedging words, technical keyword density, and vocabulary richness.<br/>"
            "<b>3. Semantic Embeddings:</b> SentenceTransformer (all-MiniLM-L6-v2) converts the transcript into a dense 384-dimensional vector.<br/>"
            "<b>4. ML Classification:</b> Logistic Regression / Random Forest predicts GOOD vs. NEEDS IMPROVEMENT (~86% accuracy).",
            callout_style
        )
    ]]
    pipeline_box = Table(pipeline_box_data, colWidths=[487])
    pipeline_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EDF2F7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(pipeline_box)
    story.append(Spacer(1, 10))

    # Section 2: Whisper
    story.append(Paragraph("2. Speech-to-Text: Why OpenAI Whisper?", h1_style))
    story.append(Paragraph(
        "A common viva question is why we chose Whisper over alternatives like Google Cloud Speech API, Vosk, or CMU Sphinx.",
        body_style
    ))
    story.append(Paragraph("• <b>Retains Filler Words (The #1 reason):</b> Cloud APIs automatically 'clean up' transcripts by deleting 'um', 'uh', and repeated words. Since our system exists specifically to detect speech hesitation, Whisper's word-for-word verbatim output is critical.", bullet_style))
    story.append(Paragraph("• <b>Handles Technical Terms &amp; Diverse Accents:</b> Whisper was trained on 680,000 hours of multilingual web audio, recognizing terms like 'Kubernetes', 'Postgres', 'Kafka', and handling non-native/Indian accents smoothly.", bullet_style))
    story.append(Paragraph("• <b>100% Free &amp; Offline Privacy:</b> Runs locally on machine CPU/GPU using the 'base' model without cloud API bills or sending candidate voice data externally.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 3: Feature Extraction
    story.append(Paragraph("3. Feature Extraction: Linguistic Diagnostics (extract_features.py)", h1_style))
    story.append(Paragraph(
        "The transcript is processed through regular expressions and lexical matching to extract actionable delivery statistics:",
        body_style
    ))
    story.append(Paragraph("• <b>Tokenization &amp; Lowercasing:</b> Strips punctuation and converts text to lowercase using regex (\\b[\\w']+\\b).", bullet_style))
    story.append(Paragraph("• <b>Filler Word Count:</b> Checks words against predefined disfluencies (um, uh, like, basically, actually, you know).", bullet_style))
    story.append(Paragraph("• <b>Hedging Count:</b> Identifies uncertainty phrases (maybe, I think, I guess, sort of, kind of).", bullet_style))
    story.append(Paragraph("• <b>Filler Density (%):</b> Formula: <i>(filler_count / total_words) * 100</i>. Normalizes hesitation across varying answer lengths.", bullet_style))
    story.append(Paragraph("• <b>Technical Keyword Count &amp; Density:</b> Scans against 60+ engineering terms (caching, latency, sharding, Docker, O(n), ACID, etc.).", bullet_style))
    story.append(Paragraph("• <b>Lexical Diversity (Type-Token Ratio):</b> <i>unique_words / total_words</i>. Measures vocabulary richness.", bullet_style))
    story.append(Spacer(1, 6))

    # Callout: Why hardcoded lists vs full NLP
    nlp_box_data = [[
        Paragraph(
            "<b>Study Note: Is Feature Extraction 'Hardcoded' or 'NLP'?</b><br/>"
            "The linguistic features use <i>lexicon-based rule matching</i> (hardcoded dictionaries) because filler sounds are a closed set in English, and direct lookup runs in 0.001s with zero overhead. However, the <b>main decision intelligence is true Deep NLP</b> (BERT-based SentenceTransformer).<br/>"
            "<i>Production Upgrade:</i> We can replace dictionaries with <b>KeyBERT</b> for dynamic keywords and <b>spaCy POS tagging</b> to differentiate 'I like Python' (verb) vs 'It was, like, slow' (filler).",
            callout_style
        )
    ]]
    nlp_box = Table(nlp_box_data, colWidths=[487])
    nlp_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEFCBF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#ECC94B')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(nlp_box)
    story.append(Spacer(1, 10))

    # Section 4: Sentence Embeddings
    story.append(Paragraph("4. Deep Semantic Embeddings (all-MiniLM-L6-v2)", h1_style))
    story.append(Paragraph(
        "To understand sentence meaning, grammatical structure, and technical depth beyond keyword counts, we use <b>SentenceTransformer('all-MiniLM-L6-v2')</b>.",
        body_style
    ))
    story.append(Paragraph("1. <b>Token Embeddings:</b> Text is tokenized into sub-word tokens, generating vector representations for each piece.", bullet_style))
    story.append(Paragraph("2. <b>Self-Attention Mechanism:</b> 6 Transformer layers compute relationship weights between words (e.g., linking 'reduce' with 'latency').", bullet_style))
    story.append(Paragraph("3. <b>Mean Pooling:</b> Averages all token vectors into a single <b>384-dimensional dense vector</b> representing the whole answer.", bullet_style))
    story.append(Paragraph("4. <b>L2 Normalization:</b> Scales vector length to 1.0 so length variations don't distort spatial distances.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 5: Classification
    story.append(Paragraph("5. ML Classification: How GOOD vs. NEEDS IMPROVEMENT is Decided", h1_style))
    story.append(Paragraph(
        "The classifier model receives the 384 numbers from the embedding and determines the candidate's rating.",
        body_style
    ))
    story.append(Paragraph("<b>Logistic Regression (The Weighted Calculator):</b>", h2_style))
    story.append(Paragraph(
        "• The model learned 384 weights during training. It computes the dot product: <i>z = (w₁x₁ + w₂x₂ + ... + w₃₈₄x₃₈₄) + b</i>.<br/>"
        "• It passes <i>z</i> through the <b>Sigmoid Function</b>: <i>P(GOOD) = 1 / (1 + e⁻ᶻ)</i>.<br/>"
        "• If probability &ge; 50%, output is <b>GOOD</b> (e.g., 88% confidence). If &lt; 50%, output is <b>NEEDS IMPROVEMENT</b>.<br/>"
        "• <i>Why it's preferred:</i> Trains in 0.24 seconds, uses minimal memory, and delivers high accuracy on normalized embeddings.",
        body_style
    ))
    story.append(Paragraph("<b>Random Forest (The Committee of 40 Trees):</b>", h2_style))
    story.append(Paragraph(
        "• Builds 40 independent Decision Trees (max depth 10) checking embedding value thresholds.<br/>"
        "• Each tree votes. If 35 trees vote GOOD and 5 vote NEEDS IMPROVEMENT, verdict is GOOD with 87.5% confidence.<br/>"
        "• Good for capturing non-linear patterns, but 13x slower to train than Logistic Regression.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Comparison Table
    table_data = [
        ["Model", "Paradigm", "5-Fold CV Acc", "Test Acc", "Training Time"],
        ["Logistic Regression", "Linear Probabilistic", "86.5% ± 0.8%", "86.0%", "0.24 s (Fastest)"],
        ["Linear SVM", "Max-Margin Hyperplane", "86.5% ± 0.8%", "86.0%", "69.26 s"],
        ["Random Forest", "Ensemble Bagging (40 trees)", "86.5% ± 0.9%", "86.0%", "3.23 s"]
    ]
    model_table = Table(table_data, colWidths=[110, 120, 90, 75, 92])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 10))

    # Section 6: Why extracted features matter if embeddings go to classifier
    story.append(Paragraph("6. Dual-Track Role: Why Feature Extraction Still Matters", h1_style))
    story.append(Paragraph(
        "If the classifier model only takes the 384 embeddings, what is the role of feature extraction? <b>Two crucial reasons:</b>",
        body_style
    ))
    story.append(Paragraph("1. <b>Diagnostic Transparency (Feedback for the User):</b> Deep learning embeddings are 'black box' numbers. The candidate needs concrete, human-readable numbers (e.g., 'You had a 14% filler density with 5 ums and only 1 tech term'). The feature extractor powers this report.", bullet_style))
    story.append(Paragraph("2. <b>Dataset Construction (Ground Truth):</b> In create_dataset.py, these exact features were used to label the 100,000 samples into good vs. needs_improvement.", bullet_style))
    story.append(Paragraph("3. <b>Future Scope (Feature Concatenation / Early Fusion):</b> In production, we can concatenate the 384 embeddings with the 4 numerical features to create a <b>388-dimensional vector</b>, combining semantic and lexical signals directly inside the classifier.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 7: Viva & Interview Cheat-Sheet
    story.append(Paragraph("7. Viva &amp; Interview Quick-Answers (For Sir)", h1_style))
    
    viva_qa = [
        ("Q1: Project ka overall working 1 minute mein batao?",
         "Sir, hamara system technical interviews ko evaluate karta hai 4 steps mein: (1) OpenAI Whisper speech ko word-to-word text mein convert karta hai, (2) Script filler density aur tech keywords count karti hai, (3) SentenceTransformer (all-MiniLM-L6-v2) text ko 384-dimensional semantic vector banata hai, aur (4) Logistic Regression classifier sigmoid probability ke through GOOD ya NEEDS IMPROVEMENT classify karta hai (~86% accuracy)."),
        
        ("Q2: Whisper kyun use kiya, Google Speech API kyun nahi?",
         "Sir, Google API hesitation words ('um', 'uh') ko auto-clean (delete) kar deta hai, jabki hamare project ko hesitations measure karni hoti hain. Whisper speech ko exact verbatim rakhta hai, technical terms acche se samajhta hai, aur 100% free/offline chalta hai."),
         
        ("Q3: Accuracy 100% kyun nahi hai, 86% kyun hai?",
         "Sir, real human speech mein 100% accuracy artificial overfitting hoti hai. Real life mein strong engineers bhi kabhi 'basically' bolte hain aur fluent log bina technical terms ke bolte hain. 86% realistic aur scientifically sound human-level accuracy hai."),
         
        ("Q4: Feature extraction ke features ka model se kya connection hai?",
         "Sir, embeddings classifier ko decision lene mein help karti hain, jabki extracted features candidate ko screen par actionable diagnostics (filler count, tech terms, density %) explain karne ke kaam aate hain.")
    ]

    for q, a in viva_qa:
        qa_data = [[
            Paragraph(f"<b>{q}</b><br/>{a}", callout_style)
        ]]
        qa_table = Table(qa_data, colWidths=[487])
        qa_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(qa_table)
        story.append(Spacer(1, 6))

    # Section 8: File-by-File Mapping
    story.append(Spacer(1, 6))
    story.append(Paragraph("8. Codebase File Reference", h1_style))
    file_map_data = [
        ["File Name", "Core Responsibility"],
        ["transcribe.py", "Audio (.mp3) -> Text (transcript.txt) using OpenAI Whisper (base model)."],
        ["extract_features.py", "Extracts filler count, hedging, filler density %, and technical keywords."],
        ["create_dataset.py", "Generates the 100,000-row realistic training dataset (training_data.csv)."],
        ["train_model.py", "Encodes embeddings, runs 5-fold CV, trains classifier, saves interview_classifier.pkl."],
        ["compare_models.py", "Benchmarks Logistic Regression, Linear SVM, and Random Forest."],
        ["predict.py", "Takes transcript.txt, extracts features, runs inference, prints final diagnostic report."],
        ["demo.py", "Interactive CLI demo to test custom candidate answers live."]
    ]
    file_table = Table(file_map_data, colWidths=[130, 357])
    file_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4A5568')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(file_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    create_study_guide_pdf()
