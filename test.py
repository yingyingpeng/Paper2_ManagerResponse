from openai import OpenAI

client = OpenAI(api_key="sk-aa274158bd114579a3f61a82d592fb4b", base_url="https://api.deepseek.com")

def translate_de_to_en(texts):
  """将多段德语文本翻译成英语"""
  translations = []
  for text in texts:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates German to English."},
            {"role": "user", "content": f"Translate the following German text to English: {text}"},
        ],
        stream=False
    )
    translations.append(response.choices[0].message.content)
  return translations

# 测试翻译
texts_de = [
    "Hallo Welt!",  # 德语文本1
    "Wie geht es dir?",  # 德语文本2
    "Ich bin gut, danke.",  # 德语文本3
]
translations = translate_de_to_en(texts_de)

# 打印结果
for i, text in enumerate(texts_de):
  print(f"原文 {i+1}: {text}")
  print(f"翻译 {i+1}: {translations[i]}")