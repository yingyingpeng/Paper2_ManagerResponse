import pandas as pd
from openai import OpenAI

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

client = OpenAI(api_key="sk-aa274158bd114579a3f61a82d592fb4b", base_url="https://api.deepseek.com")

data = pd.read_csv('Artikelrezensionen_EW_2018_2023.csv', sep="|")

# 删除Content或Comment列有NaN的行
data = data.dropna(subset=['Content', 'Comment'], how='any')

content_list = data['Content'].tolist()
comment_list = data['Comment'].tolist()

content_translation = translate_de_to_en(content_list)
comment_translation = translate_de_to_en(comment_list)

data['Content'] = content_translation
data['Comment'] = comment_translation

# 保存翻译后的数据
data.to_csv('Artikelrezensionen_EW_2018_2023_translated.csv', sep="|", index=False)
