import streamlit as st
import boto3

# ---------------------------------------------------------
# 1. ตั้งค่า Config (แก้ตรงนี้ให้เป็นข้อมูลของคุณ)
# ---------------------------------------------------------
KB_ID = 'XXXXXXXXXX'  # << เอา Knowledge Base ID มาใส่ตรงนี้
MODEL_ARN = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0' # หรือ Model ที่คุณเลือกใช้
REGION = 'us-east-1' # Region ที่คุณสร้าง Bedrock (เช่น us-east-1)

# ---------------------------------------------------------
# 2. เชื่อมต่อ AWS Bedrock
# ---------------------------------------------------------
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=REGION
    )

bedrock_client = get_bedrock_client()

# ---------------------------------------------------------
# 3. สร้างหน้าเว็บ
# ---------------------------------------------------------
st.set_page_config(page_title="Company AI Assistant")
st.title("🤖 Chatbot ถาม-ตอบ ข้อมูลองค์กร")

# เก็บประวัติการคุย (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการคุยเก่าบนหน้าจอ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 4. ส่วนรับข้อความและตอบกลับ
# ---------------------------------------------------------
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    
    # แสดงคำถามผู้ใช้
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ส่งให้ AI คิด
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลในเอกสาร..."):
            try:
                response = bedrock_client.retrieve_and_generate(
                    input={'text': prompt},
                    retrieveAndGenerateConfiguration={
                        'type': 'KNOWLEDGE_BASE',
                        'knowledgeBaseConfiguration': {
                            'knowledgeBaseId': KB_ID,
                            'modelArn': MODEL_ARN
                        }
                    }
                )
                answer = response['output']['text']
                
                # แสดงคำตอบ
                st.markdown(answer)
                
                # (Optional) แสดง Reference
                citations = response.get('citations', [])
                if citations:
                    with st.expander("ดูแหล่งอ้างอิง"):
                        for cit in citations:
                            uri = cit['retrievedReferences'][0]['location']['s3Location']['uri']
                            st.write(f"- {uri}")

                # บันทึกคำตอบลงประวัติ
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")