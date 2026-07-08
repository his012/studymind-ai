from openai import OpenAI
import os
from memory import get_relevant_memories, save_memory

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def chat_with_memory(user_id: str, user_message: str) -> str:
    past_memories = get_relevant_memories(user_id, user_message)

    memory_context = ""
    if past_memories:
        formatted = "\n".join([f"- {mem}" for mem in past_memories])
        memory_context = f"""
Here is what you already know about this student from previous sessions:
{formatted}
Use this information to personalize your response.
"""

    system_prompt = f"""You are StudyMind AI, a smart and caring personal study assistant with persistent memory.
You remember everything about your student across all sessions — their subjects, weak topics, quiz scores, learning style, and upcoming deadlines.
You help students from ALL fields — medical, engineering, arts, commerce, law, and more.
When explaining concepts, always give practical examples. For technical/coding topics, provide working code examples with clear explanations. For other subjects, use real-world analogies.
Keep explanations simple, friendly, and thorough.
{memory_context}
Your behavior:
- Always refer to past context naturally
- Celebrate progress and improvements
- Gently push on weak areas
- Give quizzes when asked
- Keep responses encouraging and friendly"""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=600
    )

    ai_response = response.choices[0].message.content
    _auto_save_memories(user_id, user_message, ai_response)
    return ai_response

def _auto_save_memories(user_id: str, user_message: str, ai_response: str):
    msg_lower = user_message.lower()

    subjects = ["math", "physics", "chemistry", "biology", "english", "urdu",
                "dsa", "data structures", "algorithms", "programming", "java",
                "python", "calculus", "statistics", "history", "geography"]
    for subject in subjects:
        if subject in msg_lower:
            save_memory(user_id, f"Student is studying: {subject}", memory_type="subject", importance=7)
            break

    weak_keywords = ["don't understand", "confused", "struggling", "difficult",
                     "hard", "can't solve", "help me with", "explain", "samajh nahi"]
    for keyword in weak_keywords:
        if keyword in msg_lower:
            save_memory(user_id, f"Student struggled with: {user_message[:100]}", memory_type="weak_topic", importance=9)
            break

    score_keywords = ["got", "scored", "marks", "grade", "result", "exam", "test", "quiz"]
    for keyword in score_keywords:
        if keyword in msg_lower:
            save_memory(user_id, f"Score info: {user_message[:100]}", memory_type="score", importance=8)
            break

    deadline_keywords = ["deadline", "due", "tomorrow", "next week", "exam on", "kal", "parson"]
    for keyword in deadline_keywords:
        if keyword in msg_lower:
            save_memory(user_id, f"Deadline: {user_message[:100]}", memory_type="deadline", importance=10)
            break

    save_memory(user_id, f"Conversation: {user_message[:120]}", memory_type="general", importance=3)

def generate_quiz(user_id: str, topic: str) -> dict:
    past_memories = get_relevant_memories(user_id, topic)
    memory_context = ""
    if past_memories:
        memory_context = "Student history:\n" + "\n".join([f"- {m}" for m in past_memories])

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": f"""You are StudyMind AI. Generate exactly 3 MCQ quiz questions.
{memory_context}
Return ONLY valid JSON, no extra text, in this exact format:
{{
  "questions": [
    {{
      "q": "Question text here?",
      "options": {{"A": "option1", "B": "option2", "C": "option3", "D": "option4"}},
      "answer": "A",
      "explanation": "A is correct because... B is wrong because... C is wrong because... D is wrong because..."
    }}
  ]
}}"""},
            {"role": "user", "content": f"Quiz on: {topic}"}
        ],
        max_tokens=1000
    )
    
    import json
    text = response.choices[0].message.content
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        return {"questions": []}