import streamlit as st
import boto3
from openai import OpenAI
import google.generativeai as genai
import time

# =========================================================
# 🔴 1. ส่วนตั้งค่า API KEYS (ใส่ข้อมูลของคุณที่นี่)
# =========================================================
AWS_ACCESS_KEY = "AKIAxxxxxxxxxxxxxxxx"       
AWS_SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
KB_ID = "XXXXXXXXXX"  
REGION = "us-east-1"

DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
GEMINI_API_KEY = "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# =========================================================
# ⚙️ 2. ตั้งค่าหน้าเว็บ (Page Config)
# =========================================================
st.set_page_config(
    page_title="ระบบเปรียบเทียบ AI - ศาลปกครอง",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 3. ตกแต่ง CSS (ธีมศาลปกครอง: น้ำเงิน/ทอง/ฟอนต์สารบรรณ)
# =========================================================
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Sarabun จาก Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');

    /* บังคับใช้ฟอนต์ทั้งหน้าเว็บ */
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }

    /* หัวข้อหลักด้านบน */
    .main-header {
        background-color: #002D62; /* สีน้ำเงินเข้มราชการ */
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #FFD700; /* สีทอง */
        font-weight: 700;
        margin: 0;
        font-size: 28px;
    }
    .main-header p {
        color: #E0E0E0;
        margin-top: 5px;
        font-size: 16px;
    }

    /* การ์ดแสดงผลลัพธ์ */
    .result-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
    }
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        color: white;
        margin-bottom: 15px;
    }
    .badge-aws { background-color: #232F3E; } /* สี AWS */
    .badge-deepseek { background-color: #4B0082; } /* สีม่วง */
    .badge-gemini { background-color: #1E88E5; } /* สีฟ้า Google */

    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #ddd;
    }
    
    /* กล่อง Source */
    .source-box {
        background-color: #F0F4F8;
        border-left: 4px solid #002D62;
        padding: 10px;
        font-size: 14px;
        margin-top: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🔧 4. Setup Clients & Logic
# =========================================================

# --- Clients Setup ---
@st.cache_resource
def get_aws_client():
    return boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

@st.cache_resource
def get_deepseek_client():
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

genai.configure(api_key=GEMINI_API_KEY)

# --- Models List ---
MODELS = {
    "Claude 3.5 Sonnet (AWS)": {"type": "bedrock", "id": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0", "color": "badge-aws"},
    "Claude 3 Haiku (AWS)": {"type": "bedrock", "id": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0", "color": "badge-aws"},
    "DeepSeek V3 (API)": {"type": "deepseek", "id": "deepseek-chat", "color": "badge-deepseek"},
    "Gemini 1.5 Flash (Google)": {"type": "gemini", "id": "gemini-1.5-flash", "color": "badge-gemini"},
    "Gemini 1.5 Pro (Google)": {"type": "gemini", "id": "gemini-1.5-pro", "color": "badge-gemini"},
}

# --- Shared Functions ---
def get_retrieved_context(prompt):
    aws_client = get_aws_client()
    try:
        retrieval = aws_client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={'text': prompt},
            retrievalConfiguration={'vectorSearchConfiguration': {'numberOfResults': 5}}
        )
        context_text = ""
        citations = []
        if 'retrievalResults' in retrieval:
            for result in retrieval['retrievalResults']:
                text = result['content']['text']
                uri = result['location']['s3Location']['uri']
                context_text += f"- {text}\n"
                citations.append({
                    'retrievedReferences': [{
                        'content': {'text': text},
                        'location': {'s3Location': {'uri': uri}}
                    }]
                })
        return context_text, citations
    except Exception as e:
        return None, str(e)

# --- AI Functions ---
def ask_bedrock(prompt, model_arn):
    client = get_aws_client()
    try:
        response = client.retrieve_and_generate(
            input={'text': prompt},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KB_ID,
                    'modelArn': model_arn
                }
            }
        )
        return response['output']['text'], response.get('citations', [])
    except Exception as e:
        return f"Error: {str(e)}", []

def ask_deepseek(prompt, model_name):
    ds_client = get_deepseek_client()
    context, cites = get_retrieved_context(prompt)
    if context is None: return f"Search Error: {cites}", []
    if not context: return "ไม่พบข้อมูลในเอกสาร", []
    
    try:
        res = ds_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": f"ตอบคำถามจากข้อมูลนี้: {context}"}, {"role": "user", "content": prompt}],
            stream=False
        )
        return res.choices[0].message.content, cites
    except Exception as e:
        if "402" in str(e): return "⚠️ DeepSeek Credit หมด (Error 402)", cites
        return f"Error: {str(e)}", []

def ask_gemini(prompt, model_name):
    context, cites = get_retrieved_context(prompt)
    if context is None: return f"Search Error: {cites}", []
    if not context: return "ไม่พบข้อมูลในเอกสาร", []

    try:
        model = genai.GenerativeModel(model_name)
        res = model.generate_content(f"Context: {context}\n\nQuestion: {prompt}")
        return res.text, cites
    except Exception as e:
        return f"Error: {str(e)}", []

def query_router(prompt, model_key):
    config = MODELS[model_key]
    if config["type"] == "bedrock": return ask_bedrock(prompt, config["id"])
    elif config["type"] == "deepseek": return ask_deepseek(prompt, config["id"])
    elif config["type"] == "gemini": return ask_gemini(prompt, config["id"])

# =========================================================
# 🖥️ 5. ส่วนแสดงผล (UI Layout)
# =========================================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Emblem_of_the_Administrative_Court_of_Thailand.svg/200px-Emblem_of_the_Administrative_Court_of_Thailand.svg.png", width=100) # Placeholder Logo
    st.markdown("### ⚙️ ตั้งค่าการประมวลผล")
    st.info("เลือกรุ่นปัญญาประดิษฐ์เพื่อเปรียบเทียบผลลัพธ์")
    
    model_left = st.selectbox("🤖 โมเดลฝั่งซ้าย", list(MODELS.keys()), index=0)
    model_right = st.selectbox("🦁 โมเดลฝั่งขวา", list(MODELS.keys()), index=3)
    
    st.markdown("---")
    if st.button("ล้างประวัติการสนทนา", type="primary"):
        st.session_state.history = []
        st.rerun()

# --- Main Content ---
# Header สวยงาม
st.markdown("""
<div class="main-header">
    <h1>⚖️ ระบบสนับสนุนการค้นหาข้อมูลศาลปกครอง</h1>
    <p>AI-Powered Knowledge Retrieval & Comparison System</p>
</div>
""", unsafe_allow_html=True)

# Session State
if "history" not in st.session_state: st.session_state.history = []

# แสดงผลการค้นหา
for chat in st.session_state.history:
    st.markdown(f"#### 🗣️ คำถาม: {chat['question']}")
    
    col1, col2 = st.columns(2)
    
    # Card ฝั่งซ้าย
    with col1:
        config = MODELS[chat['m1']]
        st.markdown(f"""
        <div class="result-card">
            <div class="model-badge {config['color']}">{chat['m1']}</div>
            <div style="line-height: 1.6; color: #333;">{chat['a1']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Source (Separate Expander for cleaner look)
        if chat['c1']:
            with st.expander("📄 เอกสารอ้างอิง (Source)"):
                seen = set()
                for c in chat['c1']:
                    uri = c['retrievedReferences'][0]['location']['s3Location']['uri'].split('/')[-1]
                    if uri not in seen:
                        st.markdown(f"<div class='source-box'>📎 {uri}</div>", unsafe_allow_html=True)
                        seen.add(uri)

    # Card ฝั่งขวา
    with col2:
        config = MODELS[chat['m2']]
        st.markdown(f"""
        <div class="result-card">
            <div class="model-badge {config['color']}">{chat['m2']}</div>
            <div style="line-height: 1.6; color: #333;">{chat['a2']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if chat['c2']:
            with st.expander("📄 เอกสารอ้างอิง (Source)"):
                seen = set()
                for c in chat['c2']:
                    uri = c['retrievedReferences'][0]['location']['s3Location']['uri'].split('/')[-1]
                    if uri not in seen:
                        st.markdown(f"<div class='source-box'>📎 {uri}</div>", unsafe_allow_html=True)
                        seen.add(uri)
    
    st.markdown("---")

# --- Input Area (ด้านล่างสุด) ---
prompt = st.chat_input("พิมพ์คำถามเพื่อสืบค้นข้อมูลระเบียบหรือคำพิพากษา...")

if prompt:
    # Logic การเรียก AI (เหมือนเดิม แต่เพิ่ม Loading สวยๆ)
    with st.spinner("⏳ กำลังสืบค้นข้อมูลจากฐานข้อมูลองค์กร..."):
        a1, c1 = query_router(prompt, model_left)
        a2, c2 = query_router(prompt, model_right)
    
    st.session_state.history.append({
        "question": prompt,
        "m1": model_left, "a1": a1, "c1": c1,
        "m2": model_right, "a2": a2, "c2": c2
    })
    st.rerun()
